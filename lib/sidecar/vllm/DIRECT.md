<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend-direct serving (prototype)

`dynamo-vllm-direct` is the stock Dynamo HTTP frontend whose router dials
each vLLM engine's own gRPC server. The frontend and every vLLM server are
still separate processes (separate pods in a cluster). What is removed is the
hop between them: no sidecar process per engine, no request plane, no NATS.
Each engine is still an ordinary worker to the router, so selection, fault
detection and migration are unchanged.

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
    subgraph direct["This branch: frontend dials the engines directly"]
        direction LR
        C2[Client] --> F2[Frontend pod<br/>preprocessor · PushRouter<br/>DirectDispatch]
        F2 -- vLLM gRPC --> V3[vLLM pod]
        F2 -- vLLM gRPC --> V4[vLLM pod]
        F2 -. "label selector<br/>nvidia.com/dynamo-engine=vllm-grpc" .- K2[(Kubernetes API<br/>Pods · EndpointSlices)]
        V3 -. labels + annotations .- K2
        V4 -. labels + annotations .- K2
    end
```

Same processes as today minus the sidecars: one per frontend, one per vLLM
server. The frontend holds a gRPC client per engine.

### Discovery by label selector

Today's Kubernetes backend expects each worker pod to publish its own
`DynamoWorkerMetadata` CR; the daemon joins those CRs with pod or
EndpointSlice readiness under the labels
`nvidia.com/dynamo-discovery-backend=kubernetes` and
`nvidia.com/dynamo-discovery-enabled=true`. A stock vLLM pod runs no Dynamo
code and writes no CR, so for direct engines the frontend does the publishing
side itself:

| Step | Who | What |
|---|---|---|
| select | frontend | watch Pods (or EndpointSlices) matching an engine label, e.g. `nvidia.com/dynamo-engine=vllm-grpc`, same namespace, Ready only |
| describe | pod spec | annotations name the gRPC port, the Dynamo namespace/component/endpoint and the disaggregation role; the model name and tokenizer come from the engine's own model and server info calls when the frontend connects |
| identify | frontend | instance id = `hash_pod_name(pod)`, the function the EPP already uses, so every frontend derives the same id without coordination |
| publish | frontend | synthesize the `Model` and `Endpoint` discovery instances in its own `DiscoveryInstance` stream, so `ModelWatcher`, `ModelManager` and the KV router see an ordinary worker |
| retire | frontend | pod NotReady or deleted removes the instance; `PushRouter` drains it like any withdrawn worker |

This makes `register_as` unnecessary on the Kubernetes backend: nothing is
registered, engines are observed. The prototype runner still takes engines as
`--grpc-endpoint` flags and publishes them into a process-local store
(`DirectEngines::add`), which is the right shape for a laptop or a test and
becomes one of two engine sources, the other being the label watch.

## The seam

`PushRouter` already ends in a `StreamingDispatch` trait object; the request
plane is just its default implementation. Everything above that seam is
untouched: the model watcher, the preprocessor, selection (`round_robin`,
`random`, `kv`, ...), occupancy, fault detection and migration.

```mermaid
flowchart TB
    subgraph process["Frontend process"]
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
        DISC[(Discovery stream)]
        LW["Pod label watch<br/>planned"]

        HTTP --> PRE --> PR
        MW -. builds router for each model .-> PR
        PR --> HOOK
        HOOK -- none installed --> RP
        HOOK -- installed --> DD
        DD -- "engines[instance_id]" --> A1
        DD -- "engines[instance_id]" --> A2
        DE -- "--grpc-endpoint flags: start, build card,<br/>register_as(Model), register_as(Endpoint)" --> DISC
        LW -. "Ready pods by label: id = hash_pod_name,<br/>Model + Endpoint from annotations" .-> DISC
        LW -. "connect gRPC client per pod" .-> DD
        DE -- install(endpoint, dispatch) --> HOOK
        DE --> DD
        DISC -. watch .-> MW
    end
    A1 -- "vllm.Inference gRPC" --> V1[(vLLM 0.28<br/>vllm-rs server)]
    A2 -- "vllm.Inference gRPC" --> V2[(vLLM 0.28<br/>vllm-rs server)]
    K8S[(Kubernetes API)] -. "watch Pods" .-> LW
```

## Request and response path

Both directions cross the same seam. The request goes down as a
`PreprocessedRequest`; the response comes back as a stream of
`LLMEngineOutput` deltas that each layer maps on its way up. There is no
response plane: the gRPC server stream *is* the response stream.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as HttpService
    participant P as Preprocessor
    participant R as PushRouter
    participant D as DirectDispatch
    participant A as EngineAdapter<br/>VllmSidecarEngine
    participant V as vLLM pod<br/>vllm-rs gRPC

    C->>H: POST /v1/chat/completions
    H->>P: OpenAI request
    P->>P: chat template, tokenize
    P->>R: PreprocessedRequest (token ids)
    R->>R: select instance id, add occupancy
    R->>D: AddressedRequest(instance id)
    D->>A: engines[instance id].generate
    A->>V: Inference.GenerateStream (proto request)

    loop each step until finish
        V-->>A: GenerateResponse (token ids, logprobs)
        A-->>D: LLMEngineOutput delta
        D-->>R: Annotated LLMEngineOutput<br/>transport faults lifted here
        R-->>P: same stream, occupancy tracked
        P-->>H: detokenize, reasoning and tool parsing
        H-->>C: SSE chunk
    end
    V-->>A: final response (finish reason, usage)
    A-->>C: finish chunk, then [DONE], via the same layers

    opt client disconnects or stop
        C--xH: connection closed
        H->>R: context stopped
        R->>A: stop propagates through the request context
        A->>V: Control.Abort(request id), stream dropped
    end

    opt engine pod lost mid-stream
        V--xA: stream error (Unavailable or Cancelled)
        A-->>D: Backend(Disconnected / Cancelled)
        D-->>R: lifted to Disconnected
        R->>R: quarantine instance, migrate if enabled
    end
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

- Label-selector engine source: watch Pods by label, read annotations,
  derive ids from `hash_pod_name`, publish into the frontend's discovery stream.
- KV events into the indexer, so `--router-mode kv` is prefix-aware.
- `remove()` and health-driven unregister.
- `--direct-engine` flag in the Python frontend behind a cargo feature.
- SGLang and TRT-LLM engines on the same seam.
- AIPerf comparison against the sidecar path.
