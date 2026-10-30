// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-direct dispatch: the router's final hop is the engine's own gRPC,
//! hosted inside the frontend process. No sidecar process, no request plane.
//!
//! Each engine is a vLLM Mocker behind the vLLM gRPC contract. `DirectEngines`
//! starts the sidecar engine client in-process, registers it in the frontend's
//! own discovery as an ordinary worker, and serves it through a
//! `StreamingDispatch` so `PushRouter` (and therefore `RoutingHost`) selects,
//! dispatches, and fault-detects exactly as it does over the request plane.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::direct::DirectEngines;
use dynamo_backend_common::{
    BackendError, DisaggregationMode, DynamoError, ErrorType, FinishReason, LLMEngine,
    PreprocessedRequest,
};

use dynamo_runtime::error::match_error_chain;
use dynamo_runtime::pipeline::{AsyncEngine, Context, PushRouter, RouterMode};
use dynamo_sidecar_testkit::{
    assert::terminal,
    bounded,
    control::{Controller, Event, RequestHandle, RequestPlan},
    fixtures,
};

#[allow(dead_code)]
#[path = "support/process.rs"]
mod process;
#[allow(dead_code)]
mod support;

use process::{Environment, Frontend, outputs};
use serde_json::Value;
use support::{FixtureConfig, ProcessFixture, SidecarFixture, WireFixture, vllm};

/// One stock engine as the frontend sees it: its mock gRPC server, the
/// controller that scripts and observes it, and the worker id discovery gave it.
struct Engine {
    fixture: vllm::Fixture,
    control: Controller<vllm::Adapter>,
    id: u64,
}

async fn add_engine(env: &Environment, direct: &DirectEngines) -> Engine {
    let control = Controller::default();
    let fixture = vllm::Fixture::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            ..Default::default()
        },
    )
    .await;
    let engine: Arc<dyn LLMEngine> = Arc::new(fixture.engine().await);
    let id = bounded("direct engine start + registration", direct.add(engine))
        .await
        .unwrap();
    Engine {
        fixture,
        control,
        id,
    }
}

fn request(env: &Environment, id: &str) -> (Context<PreprocessedRequest>, PreprocessedRequest) {
    let mut request = fixtures::request(&env.model, vec![11, 22, 33, 44], 3);
    // The same sampling and logprob fields the request-plane scenarios pin on the wire.
    <vllm::Fixture as ProcessFixture>::configure_request(&mut request);
    (
        Context::with_id_and_metadata(request.clone(), id.to_string(), Default::default()),
        request,
    )
}

async fn released(handle: &RequestHandle<vllm::Adapter>) {
    bounded("native request release", handle.wait(Event::Dropped)).await;
}

#[tokio::test]
async fn direct_engines_register_as_distinct_workers_with_model_cards() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let env = Environment::new().await;
        let endpoint = env.endpoint("backend");
        let direct = DirectEngines::new(endpoint.clone(), DisaggregationMode::Aggregated);

        let a = add_engine(&env, &direct).await;
        let b = add_engine(&env, &direct).await;
        assert_ne!(
            a.id, b.id,
            "two engines in one process must not share the process connection id"
        );

        let client = endpoint.client().await.unwrap();
        let instances = bounded("direct worker discovery", client.wait_for_instances())
            .await
            .unwrap();
        let discovered: BTreeSet<u64> = instances.iter().map(|instance| instance.id()).collect();
        assert_eq!(discovered, BTreeSet::from([a.id, b.id]));

        let cards = env.cards().await;
        assert_eq!(cards.len(), 2, "one model card per direct engine");
        assert!(cards.iter().all(|card| card.name() == env.model));
        assert!(cards.iter().all(|card| card.tokenizer.is_some()));

        drop((a, b));
    })
    .await
    .expect("direct registration scenario exceeded its overall deadline");
}

#[tokio::test]
async fn direct_request_reaches_the_selected_engine_over_grpc() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let env = Environment::new().await;
        let endpoint = env.endpoint("backend");
        let direct = DirectEngines::new(endpoint.clone(), DisaggregationMode::Aggregated);
        let engine = add_engine(&env, &direct).await;

        let client = endpoint.client().await.unwrap();
        bounded("direct worker discovery", client.wait_for_instances())
            .await
            .unwrap();
        let router = PushRouter::from_client_with_dispatch(
            client,
            RouterMode::RoundRobin,
            direct.dispatch(),
        )
        .await
        .unwrap();

        let handle = engine.control.request("direct-1", RequestPlan::default());
        let (context, expected) = request(&env, "direct-1");
        let stream = bounded("direct dispatch", router.generate(context))
            .await
            .unwrap();
        let result = outputs(stream).await;

        assert_eq!(
            handle.tokens().len(),
            3,
            "the mock engine generated the request"
        );
        vllm::Fixture::assert_stream(&handle, &expected, &result);
        terminal(result, &handle.tokens(), 4, FinishReason::Length);
        released(&handle).await;

        drop(engine);
    })
    .await
    .expect("direct dispatch scenario exceeded its overall deadline");
}

#[tokio::test]
async fn lost_direct_engine_is_a_transport_fault_and_leaves_the_avail_set() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let env = Environment::new().await;
        let endpoint = env.endpoint("backend");
        let direct = DirectEngines::new(endpoint.clone(), DisaggregationMode::Aggregated);
        let healthy = add_engine(&env, &direct).await;
        let mut lost = add_engine(&env, &direct).await;

        let client = endpoint.client().await.unwrap();
        bounded("direct worker discovery", async {
            while client.instance_ids().len() < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        let router = PushRouter::from_client_with_dispatch(
            client.clone(),
            RouterMode::RoundRobin,
            direct.dispatch(),
        )
        .await
        .unwrap();

        lost.fixture.shutdown().await;

        // Round robin visits both workers within two requests. The one that
        // lands on the dead engine must fail the way the request plane fails:
        // the outermost error is a top-level transport `ErrorType`, so
        // `PushRouter`'s fault detection reports the worker down. The sidecar
        // engine's own `Backend(CannotConnect)` stays underneath as the cause;
        // on its own it would not be recognised.
        let mut fault = None;
        for attempt in 0..2 {
            let id = format!("direct-fault-{attempt}");
            let handle = healthy.control.request(&id, RequestPlan::default());
            let (context, _) = request(&env, &id);
            match bounded("direct dispatch", router.generate(context)).await {
                Ok(stream) => {
                    let result = outputs(stream).await;
                    terminal(result, &handle.tokens(), 4, FinishReason::Length);
                    released(&handle).await;
                }
                Err(error) => {
                    let outer = error
                        .downcast_ref::<DynamoError>()
                        .unwrap_or_else(|| panic!("typed transport fault expected: {error:#}"))
                        .error_type();
                    assert!(
                        matches!(
                            outer,
                            ErrorType::CannotConnect
                                | ErrorType::Disconnected
                                | ErrorType::ConnectionTimeout
                        ),
                        "lost engine must surface as a top-level transport fault, got {outer:?}: {error:#}"
                    );
                    // The engine client's own report stays underneath as the
                    // cause. Which one depends on timing: a refused connect, or a
                    // stream cancelled by the connection going away.
                    assert!(
                        match_error_chain(
                            error.as_ref(),
                            &[
                                ErrorType::Backend(BackendError::CannotConnect),
                                ErrorType::Backend(BackendError::Disconnected),
                                ErrorType::Backend(BackendError::ConnectionTimeout),
                                ErrorType::Backend(BackendError::Cancelled),
                            ],
                            &[],
                        ),
                        "the engine client's own fault stays in the chain as the cause: {error:#}"
                    );
                    fault = Some(error);
                }
            }
        }
        assert!(
            fault.is_some(),
            "one of two round-robin requests must hit the lost engine"
        );

        bounded("lost engine leaves the avail set", async {
            while client.instance_ids_avail().contains(&lost.id) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(client.instance_ids_avail().contains(&healthy.id));

        // The healthy engine keeps serving.
        let handle = healthy
            .control
            .request("direct-after-fault", RequestPlan::default());
        let (context, _) = request(&env, "direct-after-fault");
        let stream = bounded("direct dispatch after fault", router.generate(context))
            .await
            .unwrap();
        terminal(
            outputs(stream).await,
            &handle.tokens(),
            4,
            FinishReason::Length,
        );
        released(&handle).await;

        drop((healthy, lost));
    })
    .await
    .expect("direct fault scenario exceeded its overall deadline");
}

#[tokio::test]
async fn direct_engine_serves_http_completions_through_the_frontend() {
    tokio::time::timeout(Duration::from_secs(60), async {
        let env = Environment::new().await;
        let endpoint = env.endpoint("backend");
        let direct = DirectEngines::new(endpoint.clone(), DisaggregationMode::Aggregated);
        let engine = add_engine(&env, &direct).await;

        // The stock HTTP frontend: model watcher, preprocessor, router. It must
        // reach the direct engine without a request plane in between.
        let frontend = Frontend::start(&env).await;

        let id = "00000000-0000-4000-8000-000000000101";
        let handle = engine.control.request(id, RequestPlan::default());
        let response = bounded(
            "HTTP completion through the frontend",
            frontend.generate(&env.model, id, false).send(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
        assert_eq!(body["usage"]["completion_tokens"], 3, "{body}");
        assert_eq!(handle.tokens().len(), 3, "the engine generated the request");
        released(&handle).await;

        frontend.shutdown().await;
        drop(engine);
    })
    .await
    .expect("direct HTTP scenario exceeded its overall deadline");
}

// ---------------------------------------------------------------------------
// Engines discovered from pods by label selector.
//
// The frontend observes stock engine pods instead of waiting for them to
// register: a `Ready`, non-terminating pod becomes a worker keyed by
// `hash_pod_name`, and leaves when it stops serving. The pods here are the
// objects a Kubernetes watch would deliver; their address is a mock vLLM gRPC
// server on loopback.
// ---------------------------------------------------------------------------

mod pods {
    use super::*;

    use dynamo_backend_common::direct::pods::{Connect, PORT_ANNOTATION, PodEngines};
    use dynamo_runtime::discovery::hash_pod_name;
    use dynamo_vllm_sidecar::VllmSidecarEngine;
    use futures::FutureExt;
    use k8s_openapi::api::core::v1::Pod;
    use serde_json::json;

    /// The connector the runner uses: a vLLM gRPC client for `host:port`.
    fn connect() -> Connect {
        Arc::new(|grpc_endpoint: String| {
            async move {
                let argv = vec![
                    "dynamo-vllm-sidecar".to_string(),
                    "--grpc-endpoint".to_string(),
                    grpc_endpoint,
                    "--grpc-startup-deadline-secs".to_string(),
                    "2".to_string(),
                    "--grpc-connect-attempt-timeout-secs".to_string(),
                    "1".to_string(),
                ];
                let (engine, _) =
                    tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
                        .await
                        .expect("engine argv parse task")?;
                Ok(Arc::new(engine) as Arc<dyn LLMEngine>)
            }
            .boxed()
        })
    }

    /// `(ip, port)` of a mock server's `http://ip:port` endpoint.
    fn address(fixture: &vllm::Fixture) -> (String, u16) {
        let endpoint = fixture.server.endpoint();
        let host_port = endpoint.trim_start_matches("http://");
        let (ip, port) = host_port.rsplit_once(':').unwrap();
        (ip.to_string(), port.parse().unwrap())
    }

    fn pod(name: &str, (ip, port): (String, u16), ready: bool) -> Pod {
        serde_json::from_value(json!({
            "metadata": {
                "name": name,
                "labels": { "app": "vllm" },
                "annotations": { PORT_ANNOTATION: port.to_string() },
            },
            "status": {
                "podIP": ip,
                "conditions": [{ "type": "Ready", "status": if ready { "True" } else { "False" } }],
            },
        }))
        .unwrap()
    }

    async fn start_fixture(env: &Environment) -> (vllm::Fixture, Controller<vllm::Adapter>) {
        let control = Controller::default();
        let fixture = vllm::Fixture::start(
            control.clone(),
            FixtureConfig {
                model: env.model.clone(),
                ..Default::default()
            },
        )
        .await;
        (fixture, control)
    }

    async fn serving(client: &dynamo_runtime::component::Client, id: u64, label: &str) {
        bounded(label, async {
            while !client.instance_ids().contains(&id) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
    }

    async fn withdrawn(client: &dynamo_runtime::component::Client, id: u64, label: &str) {
        bounded(label, async {
            while client.instance_ids().contains(&id) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
    }

    #[tokio::test]
    async fn ready_pod_serves_as_a_worker_keyed_by_its_name_hash() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let env = Environment::new().await;
            let endpoint = env.endpoint("backend");
            let direct = Arc::new(DirectEngines::new(
                endpoint.clone(),
                DisaggregationMode::Aggregated,
            ));
            let (fixture, control) = start_fixture(&env).await;
            let pods = PodEngines::new(direct.clone(), None, connect());
            let client = endpoint.client().await.unwrap();

            pods.apply(&pod("vllm-0", address(&fixture), true)).await;
            let id = hash_pod_name("vllm-0");
            serving(&client, id, "pod engine registration").await;
            assert_eq!(client.instance_ids(), vec![id]);
            let cards = env.cards().await;
            assert_eq!(cards.len(), 1, "the pod's engine publishes one model card");

            let router = PushRouter::from_client_with_dispatch(
                client,
                RouterMode::RoundRobin,
                direct.dispatch(),
            )
            .await
            .unwrap();
            let handle = control.request("pod-1", RequestPlan::default());
            let (context, expected) = request(&env, "pod-1");
            let stream = bounded("pod dispatch", router.generate(context))
                .await
                .unwrap();
            let result = outputs(stream).await;
            vllm::Fixture::assert_stream(&handle, &expected, &result);
            terminal(result, &handle.tokens(), 4, FinishReason::Length);
            released(&handle).await;

            drop(fixture);
        })
        .await
        .expect("pod serving scenario exceeded its overall deadline");
    }

    #[tokio::test]
    async fn pod_that_stops_being_ready_or_is_deleted_is_withdrawn() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let env = Environment::new().await;
            let endpoint = env.endpoint("backend");
            let direct = Arc::new(DirectEngines::new(
                endpoint.clone(),
                DisaggregationMode::Aggregated,
            ));
            let (fixture, _control) = start_fixture(&env).await;
            let pods = PodEngines::new(direct.clone(), None, connect());
            let client = endpoint.client().await.unwrap();
            let id = hash_pod_name("vllm-0");

            pods.apply(&pod("vllm-0", address(&fixture), true)).await;
            serving(&client, id, "pod engine registration").await;

            pods.apply(&pod("vllm-0", address(&fixture), false)).await;
            withdrawn(&client, id, "not-ready pod withdrawn").await;
            assert!(env.cards().await.is_empty(), "its model card leaves too");

            pods.apply(&pod("vllm-0", address(&fixture), true)).await;
            serving(&client, id, "ready again").await;

            pods.delete(&pod("vllm-0", address(&fixture), true)).await;
            withdrawn(&client, id, "deleted pod withdrawn").await;

            drop(fixture);
        })
        .await
        .expect("pod withdrawal scenario exceeded its overall deadline");
    }

    #[tokio::test]
    async fn relist_withdraws_pods_it_no_longer_contains() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let env = Environment::new().await;
            let endpoint = env.endpoint("backend");
            let direct = Arc::new(DirectEngines::new(
                endpoint.clone(),
                DisaggregationMode::Aggregated,
            ));
            let (a, _) = start_fixture(&env).await;
            let (b, _) = start_fixture(&env).await;
            let pods = PodEngines::new(direct.clone(), None, connect());
            let client = endpoint.client().await.unwrap();

            let pod_a = pod("vllm-a", address(&a), true);
            let pod_b = pod("vllm-b", address(&b), true);
            pods.resync(&[pod_a.clone(), pod_b]).await;
            serving(&client, hash_pod_name("vllm-a"), "a registered").await;
            serving(&client, hash_pod_name("vllm-b"), "b registered").await;

            pods.resync(&[pod_a]).await;
            withdrawn(&client, hash_pod_name("vllm-b"), "b dropped by relist").await;
            assert!(client.instance_ids().contains(&hash_pod_name("vllm-a")));

            drop((a, b));
        })
        .await
        .expect("relist scenario exceeded its overall deadline");
    }

    #[tokio::test]
    async fn unreachable_pod_does_not_hold_back_a_reachable_one() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let env = Environment::new().await;
            let endpoint = env.endpoint("backend");
            let direct = Arc::new(DirectEngines::new(
                endpoint.clone(),
                DisaggregationMode::Aggregated,
            ));
            let (fixture, _) = start_fixture(&env).await;
            let pods = PodEngines::new(direct.clone(), None, connect())
                .with_retry(Duration::from_millis(100));
            let client = endpoint.client().await.unwrap();

            // A port nothing listens on.
            let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let closed_port = closed.local_addr().unwrap().port();
            drop(closed);

            pods.apply(&pod("vllm-dead", ("127.0.0.1".into(), closed_port), true))
                .await;
            pods.apply(&pod("vllm-live", address(&fixture), true)).await;
            serving(
                &client,
                hash_pod_name("vllm-live"),
                "reachable pod registered",
            )
            .await;
            assert!(
                !client.instance_ids().contains(&hash_pod_name("vllm-dead")),
                "an engine that cannot be reached is never a worker"
            );

            // Withdrawing the pod stops its retries; nothing is left behind.
            pods.delete(&pod("vllm-dead", ("127.0.0.1".into(), closed_port), true))
                .await;
            assert_eq!(client.instance_ids(), vec![hash_pod_name("vllm-live")]);

            drop(fixture);
        })
        .await
        .expect("unreachable pod scenario exceeded its overall deadline");
    }
}
