// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-direct serving of stock vLLM engines.
//!
//! The Dynamo HTTP frontend whose router dials each vLLM engine's own gRPC
//! server: no sidecar process, no NATS, no request plane. Each engine is a
//! worker to the router. Engines come from two sources:
//!
//! - `--grpc-endpoint host:port`, repeated: fixed engines, for local runs.
//! - `--pod-selector app=vllm`: every `Ready` pod matching the label selector
//!   in `--pod-namespace`, at `<pod IP>:<--pod-grpc-port>` (or the pod's
//!   `nvidia.com/dynamo-engine-grpc-port` annotation), keyed by
//!   `hash_pod_name`. Pods that stop serving are withdrawn. Needs `list` and
//!   `watch` on pods.
//!
//! ```bash
//! dynamo-vllm-direct --grpc-endpoint 127.0.0.1:50051 --http-port 8000
//! dynamo-vllm-direct --pod-selector app=vllm --pod-grpc-port 50051
//! ```

use std::sync::Arc;

use clap::Parser;
use dynamo_backend_common::DisaggregationMode;
use dynamo_backend_common::direct::DirectEngines;
use dynamo_backend_common::direct::pods::{Connect, PodEngines, PodSelector};
use dynamo_backend_common::{BackendError, DynamoError, ErrorType, LLMEngine};
use dynamo_llm::discovery::ModelWatcher;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::namespace::NamespaceFilter;
use dynamo_runtime::discovery::DiscoveryQuery;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use futures::FutureExt;

#[derive(Parser, Debug)]
#[command(
    name = "dynamo-vllm-direct",
    about = "Dynamo frontend serving vLLM gRPC engines directly"
)]
struct Args {
    /// vLLM gRPC server, `host:port`. Repeat for several engines.
    #[arg(long, required_unless_present = "pod_selector")]
    grpc_endpoint: Vec<String>,

    /// Label selector for vLLM engine pods, e.g. `app=vllm`.
    #[arg(long)]
    pod_selector: Option<String>,

    /// Namespace of the engine pods.
    #[arg(long, env = "POD_NAMESPACE", default_value = "default")]
    pod_namespace: String,

    /// gRPC port of every selected pod, unless the pod's
    /// `nvidia.com/dynamo-engine-grpc-port` annotation says otherwise.
    #[arg(long)]
    pod_grpc_port: Option<u16>,

    /// Frontend HTTP port.
    #[arg(long, default_value_t = 8000)]
    http_port: u16,

    /// Frontend HTTP host.
    #[arg(long, default_value = "0.0.0.0")]
    http_host: String,

    /// Routing policy across the engines: `round_robin`, `random`, `kv`, ...
    #[arg(long, default_value = "round_robin", value_parser = parse_router_mode)]
    router_mode: RouterMode,

    /// Discovery namespace the engines register under.
    #[arg(long, default_value = "dynamo")]
    namespace: String,

    /// Seconds to wait for each engine's gRPC server at startup.
    #[arg(long, default_value_t = 300)]
    grpc_startup_deadline_secs: u64,
}

/// `RouterMode` serializes as snake_case; parse the CLI value the same way.
fn parse_router_mode(value: &str) -> Result<RouterMode, String> {
    serde_json::from_value(serde_json::Value::String(value.to_string()))
        .map_err(|error| format!("unknown router mode {value:?}: {error}"))
}

fn main() -> anyhow::Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let runtime = Runtime::from_settings()?;
    let primary = runtime.primary();
    primary.block_on(run(runtime.clone(), args))?;
    runtime.shutdown();
    Ok(())
}

async fn run(runtime: Runtime, args: Args) -> anyhow::Result<()> {
    // Discovery is local to this frontend: engines are listed (flags) or
    // observed (pods) here, never registered by the engines themselves.
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local()).await?;
    let endpoint = drt
        .namespace(&args.namespace)?
        .component("backend")?
        .endpoint("generate");

    let direct = Arc::new(DirectEngines::new(
        endpoint.clone(),
        DisaggregationMode::Aggregated,
    ));
    let connect = vllm_connector(args.grpc_startup_deadline_secs);
    for grpc_endpoint in &args.grpc_endpoint {
        let engine = connect(grpc_endpoint.clone()).await?;
        let instance_id = direct.add(engine).await?;
        tracing::info!(
            grpc_endpoint,
            instance_id = format_args!("{instance_id:x}"),
            "direct vLLM engine serving"
        );
    }
    if let Some(labels) = args.pod_selector.clone() {
        let client = kube::Client::try_default().await?;
        let selector = PodSelector {
            namespace: args.pod_namespace.clone(),
            labels,
            port: args.pod_grpc_port,
        };
        let pods = PodEngines::new(direct.clone(), selector.port, connect.clone());
        let cancel = drt.primary_token();
        tokio::spawn(async move {
            if let Err(error) = pods.watch(client, &selector, cancel).await {
                tracing::error!(%error, "engine pod watch ended");
            }
        });
    }

    let service = HttpService::builder()
        .host(args.http_host.clone())
        .port(args.http_port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .build()?;
    let watcher = Arc::new(ModelWatcher::new(
        drt.clone(),
        service.state().manager_clone(),
        RouterConfig {
            router_mode: args.router_mode,
            ..Default::default()
        },
        0,
        None,
        None,
        None,
        service.state().metrics_clone(),
    ));
    let discovery = drt
        .discovery()
        .list_and_watch(DiscoveryQuery::AllModels, Some(drt.primary_token()))
        .await?;
    let watched = watcher.clone();
    let namespace = NamespaceFilter::Exact(args.namespace.clone());
    tokio::spawn(async move { watched.watch(discovery, namespace).await });
    let model = watcher.wait_for_chat_model().await;
    tracing::info!(
        model,
        engines = direct.instance_ids().len(),
        url = format!("http://{}:{}/v1", args.http_host, args.http_port),
        "frontend serving directly"
    );

    service.run(drt.primary_token()).await
}

/// A vLLM gRPC client for a `host:port` endpoint, not yet started.
fn vllm_connector(startup_deadline_secs: u64) -> Connect {
    Arc::new(move |grpc_endpoint: String| {
        async move {
            let argv = vec![
                "dynamo-vllm-sidecar".to_string(),
                "--grpc-endpoint".to_string(),
                grpc_endpoint,
                "--grpc-startup-deadline-secs".to_string(),
                startup_deadline_secs.to_string(),
            ];
            let (engine, _worker_config) =
                tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv)))
                    .await
                    .map_err(|error| {
                        DynamoError::builder()
                            .error_type(ErrorType::Backend(BackendError::Unknown))
                            .message(format!("vLLM engine argv task: {error}"))
                            .build()
                    })??;
            Ok(Arc::new(engine) as Arc<dyn LLMEngine>)
        }
        .boxed()
    })
}
