<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend-direct serving (prototype)

`dynamo-vllm-direct` runs the stock Dynamo HTTP frontend with every vLLM
engine hosted **in the same process** as a worker. The router's final hop is
vLLM's own gRPC. There is no sidecar process, no etcd, no NATS and no request
plane in the path.

```bash
dynamo-vllm-direct --grpc-endpoint 127.0.0.1:50051 --grpc-endpoint 127.0.0.1:50052 --http-port 8000
```

## Before and after

```mermaid
flowchart LR
    subgraph today["Today: sidecar per engine"]
        direction LR
        C1[Client] --> F1[Frontend<br/>preprocessor · PushRouter]
        F1 -- request plane<br/>TCP / NATS --> S1[dynamo-vllm-sidecar]
        F1 -- request plane --> S2[dynamo-vllm-sidecar]
        S1 -- gRPC --> V1[vLLM]
        S2 -- gRPC --> V2[vLLM]
        F1 -. watches .- K[(Kubernetes discovery<br/>pod metadata · EndpointSlices)]
        S1 -. registers .- K
        S2 -. registers .- K
    end
```

```mermaid
flowchart LR
    subgraph direct["This branch: dynamo-vllm-direct, one process"]
        direction LR
        C2[Client] --> F2[Frontend<br/>preprocessor · PushRouter]
        F2 -- StreamingDispatch --> D[DirectDispatch<br/>instance id → EngineAdapter]
        D -- gRPC --> V3[vLLM]
        D -- gRPC --> V4[vLLM]
        F2 -. watches .- M[(Discovery<br/>process-local today,<br/>Kubernetes next)]
        D -. register_as(id) .- M
    end
```

Discovery is the one piece the direct process still owns. Today the runner uses
the process-local backend, since the frontend and its engines are the only
parties. In a cluster the same process should register its engines through the
Kubernetes discovery backend, so the operator and other frontends see them as
ordinary workers. `register_as` is implemented for the KV-store and mock
backends; the Kubernetes backend still refuses caller-chosen instance ids.

## The seam

`PushRouter` already ends in a `StreamingDispatch` trait object; the request
plane is just its default implementation. Everything above that seam is
untouched: the model watcher, the preprocessor, selection (`round_robin`,
`random`, `kv`, ...), occupancy, fault detection and migration.

```mermaid
flowchart TB
    subgraph process["dynamo-vllm-direct process"]
        HTTP[HttpService<br/>/v1/completions · /v1/chat/completions]
        MW[ModelWatcher]
        PRE[Preprocessor]
        PR[PushRouter<br/>select · fault detect · migrate]
        HOOK{{"direct_dispatch::lookup(endpoint)<br/>dynamo-llm registry"}}
        RP[Request plane<br/>default final hop]
        DD[DirectDispatch]
        DE[DirectEngines]
        A1[EngineAdapter → VllmSidecarEngine]
        A2[EngineAdapter → VllmSidecarEngine]
        DISC[(Discovery<br/>process_local)]

        HTTP --> PRE --> PR
        MW -. builds router for each model .-> PR
        PR --> HOOK
        HOOK -- none installed --> RP
        HOOK -- installed --> DD
        DD -- "engines[instance_id]" --> A1
        DD -- "engines[instance_id]" --> A2
        DE -- "add(engine): start, build card,<br/>register_as(Model), register_as(Endpoint)" --> DISC
        DE -- install(endpoint, dispatch) --> HOOK
        DE --> DD
        DISC -. watch .-> MW
    end
    A1 -- "vllm.Inference gRPC" --> V1[(vLLM 0.28<br/>vllm-rs server)]
    A2 -- "vllm.Inference gRPC" --> V2[(vLLM 0.28<br/>vllm-rs server)]
```

## Error lifting

The engine client reports transport loss as `Backend(...)` errors, like every
engine. `PushRouter` quarantines a worker only on top-level transport types, so
`DirectDispatch` lifts them at the seam. A cancellation the caller never asked
for is the gRPC stream dying under the request, so it is lifted too.

```mermaid
flowchart LR
    subgraph engine["From the engine client"]
        B1["Backend(CannotConnect)"]
        B2["Backend(Disconnected)"]
        B3["Backend(ConnectionTimeout)"]
        B4["Backend(ResponseTimeout)"]
        B5["Backend(Cancelled)<br/>caller did not cancel"]
        B6["Backend(Cancelled)<br/>caller cancelled"]
        B7["any other Backend(...)"]
    end
    subgraph router["Seen by PushRouter"]
        T1[CannotConnect]
        T2[Disconnected]
        T3[ConnectionTimeout]
        T4[ResponseTimeout]
        T5["unchanged"]
    end
    B1 --> T1
    B2 --> T2
    B3 --> T3
    B4 --> T4
    B5 --> T2
    B6 --> T5
    B7 --> T5
    T1 & T2 & T3 & T4 --> Q[worker leaves avail set<br/>healthy engines keep serving]
```

## What changed, by crate

| Crate | Change |
|---|---|
| `dynamo-runtime` | `Discovery::register_as(spec, instance_id)` so N engines in one process register as distinct workers. KV-store and mock backends implement it. |
| `dynamo-backend-common` | `direct::DirectEngines` hosts `LLMEngine`s, registers each as a worker with the Worker's own model card, serves them through `DirectDispatch` (a `StreamingDispatch`), lifts transport faults. |
| `dynamo-llm` | `direct_dispatch` registry keyed by endpoint id. The frontend router builder takes an installed dispatch as the final hop instead of the request plane. |
| `dynamo-vllm-sidecar` | `dynamo-vllm-direct` binary: frontend plus every `--grpc-endpoint` in-process. |
| `dynamo-sidecar-testkit` | `tests/direct_engine_integration.rs`: four CPU-only scenarios over mock vLLM gRPC servers. |

## Verified

| Scenario | Result |
|---|---|
| Two engines register as distinct workers with model cards | pass |
| Request reaches the selected engine over gRPC, wire fields intact | pass |
| Lost engine surfaces as a transport fault and leaves the avail set | pass |
| Completion through the real HTTP frontend | pass |
| Real vLLM 0.28 (`vllm-rs` gRPC), Qwen3-0.6B: completions, chat, streaming, 16 concurrent | pass |

The dev image's vLLM 0.18 `[grpc]` extra speaks a different protocol (SMG
servicer) and cannot serve this path; vLLM ≥ 0.28 with its bundled `vllm-rs`
server is required, matching `deploy/`.

## Not yet

- `register_as` on the Kubernetes discovery backend, so a direct process
  publishes its engines through k8s and needs no in-process store.
- KV events into the indexer, so `--router-mode kv` is prefix-aware.
- `remove()` and health-driven unregister.
- `--direct-engine` flag in the Python frontend behind a cargo feature.
- SGLang and TRT-LLM engines on the same seam.
- AIPerf comparison against the sidecar path.
