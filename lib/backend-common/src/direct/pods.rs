// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct engines discovered from Kubernetes pods by label selector.
//!
//! A stock engine pod runs no Dynamo code and publishes nothing, so the
//! frontend observes it instead: it watches the pods matching a label
//! selector and hosts one direct engine per `Ready`, non-terminating pod. The
//! rules match the gateway EPP's pod discovery, so both agree on worker ids:
//!
//! - a pod serves once it is `Ready` and has no deletion timestamp;
//! - its gRPC address is `<pod IP>:<port>`, the port from the pod's
//!   [`PORT_ANNOTATION`] or else the selector's pool-wide port;
//! - its worker instance id is `hash_pod_name(pod name)`.
//!
//! A pod that stops serving, is deleted, or is missing from a relist is
//! withdrawn: its engine leaves discovery and the dispatch.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::discovery::hash_pod_name;
use futures::StreamExt;
use futures::future::BoxFuture;
use k8s_openapi::api::core::v1::Pod;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::DirectEngines;
use crate::engine::LLMEngine;
use crate::error::DynamoError;

/// Per-pod override of the selector's pool-wide gRPC port.
pub const PORT_ANNOTATION: &str = "nvidia.com/dynamo-engine-grpc-port";

/// Which pods are engines, and where their gRPC servers listen.
#[derive(Clone, Debug)]
pub struct PodSelector {
    /// Namespace to watch.
    pub namespace: String,
    /// Kubernetes label selector, for example `app=vllm,tier=decode`.
    pub labels: String,
    /// Pool-wide gRPC port; a pod's [`PORT_ANNOTATION`] overrides it.
    pub port: Option<u16>,
}

/// The engine a serving pod offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PodEngine {
    pub pod: String,
    pub instance_id: u64,
    pub grpc_endpoint: String,
}

/// The engine `pod` serves right now, or `None` if it serves none.
pub fn pod_engine(pod: &Pod, default_port: Option<u16>) -> Option<PodEngine> {
    let name = pod.metadata.name.as_deref()?;
    if pod.metadata.deletion_timestamp.is_some() {
        return None;
    }
    let status = pod.status.as_ref()?;
    let ready = status
        .conditions
        .as_deref()?
        .iter()
        .any(|condition| condition.type_ == "Ready" && condition.status == "True");
    if !ready {
        return None;
    }
    let ip = status.pod_ip.as_deref().filter(|ip| !ip.is_empty())?;
    let port = match pod
        .metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(PORT_ANNOTATION))
    {
        Some(port) => port.parse().ok()?,
        None => default_port?,
    };
    let host = if ip.contains(':') {
        format!("[{ip}]")
    } else {
        ip.to_string()
    };
    Some(PodEngine {
        pod: name.to_string(),
        instance_id: hash_pod_name(name),
        grpc_endpoint: format!("{host}:{port}"),
    })
}

/// Builds an unstarted engine client for a `host:port` gRPC endpoint.
pub type Connect = Arc<
    dyn Fn(String) -> BoxFuture<'static, Result<Arc<dyn LLMEngine>, DynamoError>> + Send + Sync,
>;

/// Keeps [`DirectEngines`] in step with the serving pods.
///
/// Each serving pod gets a task that connects to its engine and hosts it,
/// retrying with backoff until it succeeds or the pod stops serving, so an
/// unreachable engine never holds back the others.
pub struct PodEngines {
    direct: Arc<DirectEngines>,
    connect: Connect,
    default_port: Option<u16>,
    retry: Duration,
    pods: tokio::sync::Mutex<HashMap<String, Tracked>>,
}

struct Tracked {
    engine: PodEngine,
    task: JoinHandle<()>,
}

const MAX_RETRY: Duration = Duration::from_secs(30);

impl PodEngines {
    pub fn new(direct: Arc<DirectEngines>, default_port: Option<u16>, connect: Connect) -> Self {
        Self {
            direct,
            connect,
            default_port,
            retry: Duration::from_secs(1),
            pods: Default::default(),
        }
    }

    /// First delay between attempts to reach a pod's engine; doubles to 30 s.
    pub fn with_retry(mut self, retry: Duration) -> Self {
        self.retry = retry;
        self
    }

    /// A pod was added or changed.
    pub async fn apply(&self, pod: &Pod) {
        let Some(name) = pod.metadata.name.clone() else {
            return;
        };
        let target = pod_engine(pod, self.default_port);
        let mut pods = self.pods.lock().await;
        if pods.get(&name).map(|tracked| &tracked.engine) == target.as_ref() {
            return;
        }
        if let Some(tracked) = pods.remove(&name) {
            self.retire(tracked).await;
        }
        if let Some(engine) = target {
            let task = self.host(engine.clone());
            pods.insert(name, Tracked { engine, task });
        }
    }

    /// A pod was deleted.
    pub async fn delete(&self, pod: &Pod) {
        let Some(name) = pod.metadata.name.as_deref() else {
            return;
        };
        let tracked = self.pods.lock().await.remove(name);
        if let Some(tracked) = tracked {
            self.retire(tracked).await;
        }
    }

    /// A full relist: these are all the selected pods.
    pub async fn resync(&self, pods: &[Pod]) {
        let listed: HashSet<&str> = pods
            .iter()
            .filter_map(|pod| pod.metadata.name.as_deref())
            .collect();
        let gone: Vec<Tracked> = {
            let mut tracked = self.pods.lock().await;
            let names: Vec<String> = tracked
                .keys()
                .filter(|name| !listed.contains(name.as_str()))
                .cloned()
                .collect();
            names
                .into_iter()
                .filter_map(|name| tracked.remove(&name))
                .collect()
        };
        for tracked in gone {
            self.retire(tracked).await;
        }
        for pod in pods {
            self.apply(pod).await;
        }
    }

    /// Watches the selected pods until `cancel` fires, then withdraws every
    /// engine it hosted.
    pub async fn watch(
        &self,
        client: kube::Client,
        selector: &PodSelector,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        use kube::runtime::{WatchStreamExt, watcher};

        let api: kube::Api<Pod> = kube::Api::namespaced(client, &selector.namespace);
        let config = watcher::Config::default().labels(&selector.labels);
        let stream = watcher(api, config).default_backoff();
        tokio::pin!(stream);
        tracing::info!(
            namespace = %selector.namespace,
            labels = %selector.labels,
            "watching engine pods"
        );

        // A relist arrives as Init, InitApply per pod, InitDone.
        let mut relist: Option<Vec<Pod>> = None;
        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => break,
                event = stream.next() => event,
            };
            match event {
                None => break,
                Some(Err(error)) => tracing::warn!(%error, "engine pod watch"),
                Some(Ok(watcher::Event::Init)) => relist = Some(Vec::new()),
                Some(Ok(watcher::Event::InitApply(pod))) => {
                    relist.get_or_insert_with(Vec::new).push(pod)
                }
                Some(Ok(watcher::Event::InitDone)) => {
                    let pods = relist.take().unwrap_or_default();
                    tracing::info!(pods = pods.len(), "engine pod relist");
                    self.resync(&pods).await;
                }
                Some(Ok(watcher::Event::Apply(pod))) => self.apply(&pod).await,
                Some(Ok(watcher::Event::Delete(pod))) => self.delete(&pod).await,
            }
        }
        self.resync(&[]).await;
        Ok(())
    }

    /// Connects to and hosts `engine`, retrying until it serves.
    fn host(&self, engine: PodEngine) -> JoinHandle<()> {
        let direct = self.direct.clone();
        let connect = self.connect.clone();
        let mut delay = self.retry;
        tokio::spawn(async move {
            loop {
                let attempt = async {
                    let client = connect(engine.grpc_endpoint.clone()).await?;
                    direct.add_as(client, engine.instance_id).await
                };
                match attempt.await {
                    Ok(_) => {
                        tracing::info!(
                            pod = %engine.pod,
                            grpc_endpoint = %engine.grpc_endpoint,
                            instance_id = format_args!("{:x}", engine.instance_id),
                            "engine pod serving"
                        );
                        return;
                    }
                    Err(error) => {
                        tracing::warn!(
                            pod = %engine.pod,
                            grpc_endpoint = %engine.grpc_endpoint,
                            %error,
                            retry_in = ?delay,
                            "engine pod not reachable"
                        );
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(MAX_RETRY);
                    }
                }
            }
        })
    }

    /// Stops trying to host a pod's engine and withdraws it if hosted.
    async fn retire(&self, tracked: Tracked) {
        tracked.task.abort();
        // An aborted attempt may have hosted part of the engine; `remove`
        // undoes whatever landed.
        let _ = tracked.task.await;
        if self.direct.remove(tracked.engine.instance_id).await {
            tracing::info!(pod = %tracked.engine.pod, "engine pod withdrawn");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pod(value: serde_json::Value) -> Pod {
        serde_json::from_value(value).unwrap()
    }

    fn serving_pod(name: &str, ip: &str, annotations: serde_json::Value) -> Pod {
        pod(json!({
            "metadata": { "name": name, "annotations": annotations },
            "status": {
                "podIP": ip,
                "conditions": [{ "type": "Ready", "status": "True" }],
            },
        }))
    }

    #[test]
    fn ready_pod_serves_at_its_ip_and_pool_port_under_its_name_hash() {
        let engine = pod_engine(&serving_pod("vllm-0", "10.1.2.3", json!({})), Some(50051));
        assert_eq!(
            engine,
            Some(PodEngine {
                pod: "vllm-0".into(),
                instance_id: hash_pod_name("vllm-0"),
                grpc_endpoint: "10.1.2.3:50051".into(),
            })
        );
    }

    #[test]
    fn port_annotation_overrides_the_pool_port() {
        let pod = serving_pod("vllm-0", "10.1.2.3", json!({ PORT_ANNOTATION: "50061" }));
        assert_eq!(
            pod_engine(&pod, Some(50051)).unwrap().grpc_endpoint,
            "10.1.2.3:50061"
        );
        assert_eq!(
            pod_engine(&pod, None).unwrap().grpc_endpoint,
            "10.1.2.3:50061"
        );
    }

    #[test]
    fn ipv6_pod_address_is_bracketed() {
        let pod = serving_pod("vllm-0", "fd00::7", json!({}));
        assert_eq!(
            pod_engine(&pod, Some(50051)).unwrap().grpc_endpoint,
            "[fd00::7]:50051"
        );
    }

    #[test]
    fn pod_without_a_port_serves_nothing() {
        assert_eq!(
            pod_engine(&serving_pod("vllm-0", "10.1.2.3", json!({})), None),
            None
        );
        let bad = serving_pod("vllm-0", "10.1.2.3", json!({ PORT_ANNOTATION: "grpc" }));
        assert_eq!(pod_engine(&bad, None), None);
    }

    #[test]
    fn pod_that_is_not_ready_serves_nothing() {
        let not_ready = pod(json!({
            "metadata": { "name": "vllm-0" },
            "status": {
                "podIP": "10.1.2.3",
                "conditions": [{ "type": "Ready", "status": "False" }],
            },
        }));
        assert_eq!(pod_engine(&not_ready, Some(50051)), None);
        let no_conditions = pod(json!({
            "metadata": { "name": "vllm-0" },
            "status": { "podIP": "10.1.2.3" },
        }));
        assert_eq!(pod_engine(&no_conditions, Some(50051)), None);
    }

    #[test]
    fn pod_without_an_ip_serves_nothing() {
        let no_ip = pod(json!({
            "metadata": { "name": "vllm-0" },
            "status": { "conditions": [{ "type": "Ready", "status": "True" }] },
        }));
        assert_eq!(pod_engine(&no_ip, Some(50051)), None);
    }

    #[test]
    fn terminating_pod_serves_nothing() {
        let terminating = pod(json!({
            "metadata": { "name": "vllm-0", "deletionTimestamp": "2026-10-30T00:00:00Z" },
            "status": {
                "podIP": "10.1.2.3",
                "conditions": [{ "type": "Ready", "status": "True" }],
            },
        }));
        assert_eq!(pod_engine(&terminating, Some(50051)), None);
    }
}
