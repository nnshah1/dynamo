// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-direct serving of stock vLLM engines, in one process.
//!
//! The Dynamo HTTP frontend with each `--grpc-endpoint` hosted in-process as a
//! worker: no sidecar process, no etcd, no NATS, no request plane. The router's
//! final hop is vLLM's own gRPC. Prototype vehicle for the direct-engine plan.
//!
//! ```bash
//! python -m vllm.entrypoints.grpc_server --model Qwen/Qwen3-0.6B --port 50051 &
//! dynamo-vllm-direct --grpc-endpoint 127.0.0.1:50051 --http-port 8000
//! ```

use std::sync::Arc;

use clap::Parser;
use dynamo_backend_common::DisaggregationMode;
use dynamo_backend_common::direct::DirectEngines;
use dynamo_llm::discovery::ModelWatcher;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::namespace::NamespaceFilter;
use dynamo_runtime::discovery::DiscoveryQuery;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_vllm_sidecar::VllmSidecarEngine;

#[derive(Parser, Debug)]
#[command(name = "dynamo-vllm-direct", about = "Dynamo frontend serving vLLM gRPC engines directly")]
struct Args {
    /// vLLM gRPC server, `host:port`. Repeat for several engines.
    #[arg(long, required = true)]
    grpc_endpoint: Vec<String>,

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
    // Everything in this process: in-memory discovery, no brokers.
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local()).await?;
    let endpoint = drt
        .namespace(&args.namespace)?
        .component("backend")?
        .endpoint("generate");

    let direct = DirectEngines::new(endpoint.clone(), DisaggregationMode::Aggregated);
    for grpc_endpoint in &args.grpc_endpoint {
        let argv = vec![
            "dynamo-vllm-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            grpc_endpoint.clone(),
            "--grpc-startup-deadline-secs".to_string(),
            args.grpc_startup_deadline_secs.to_string(),
        ];
        let argv_for_parse = argv.clone();
        let (engine, _worker_config) =
            tokio::task::spawn_blocking(move || VllmSidecarEngine::from_args(Some(argv_for_parse)))
                .await??;
        let instance_id = direct.add(Arc::new(engine)).await?;
        tracing::info!(
            grpc_endpoint,
            instance_id = format_args!("{instance_id:x}"),
            "direct vLLM engine serving"
        );
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
        engines = args.grpc_endpoint.len(),
        url = format!("http://{}:{}/v1", args.http_host, args.http_port),
        "frontend serving directly"
    );

    service.run(drt.primary_token()).await
}
