// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request routing cost of the Plan-returning `Router` against a
//! hand-composed sequence of core operations, over real selection cores with
//! identical workers, prompt, cache state and policies.
//!
//! Each iteration routes one request to completion and releases every
//! booking with an acknowledged release, so both sides time the same
//! lifecycle: admission through the last booking, then cleanup. `direct/*`
//! runs one `Lease` operation per stage with the accounting a planned stage
//! of the same work gets; `plan/*` is `Router::plan` + `schedule` over
//! `SelectionCore` and `MultiStageRouter`. `direct/aggregated/book` is the
//! wire `select_and_reserve` + `free_reservation` pair, a different
//! ownership path (the core's reservation index), kept for reference only.
//!
//! Before timing, each baseline's booked loads are checked against the
//! planned scenarios it stands in for. This compares Plan with the direct
//! core path; it says nothing about other coordination designs or about
//! serving latency.
//!
//! `ROUTER_BENCH_PERCENTILES=<samples>` replaces criterion with a fixed-sample
//! run that prints p50/p95/p99 per scenario; `ROUTER_BENCH_WORKERS` sets the
//! workers per set (default 8).

use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use dynamo_kv_router::conditional_disagg::IslBoundingPolicy;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::indexer::KvIndexerInterface;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, RoutingConstraints, StorageTier,
    compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::router::{
    ClassTable, MultiStageRouter, Outcome, Plan, Router, SkipRule, StageList, StageState, StageWork,
};
use dynamo_kv_router::scheduling::queue::BookingHandle;
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionAdmission, SelectionCacheConfig,
    SelectionCore, SelectionOperation, SelectionOutcome, SessionBinding, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, RouterConfigOverride, WorkerType};
use tokio_util::sync::CancellationToken;

const BLOCK_SIZE: u32 = 16;
const MODEL: &str = "default";

fn prompt_tokens() -> Vec<u32> {
    (1..=256).collect()
}

fn workers_per_set() -> u64 {
    std::env::var("ROUTER_BENCH_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8)
}

/// A core for one worker set with `workers` single-rank workers, ids
/// `base + 1 ..= base + workers`.
fn core_for(
    runtime: &tokio::runtime::Runtime,
    worker_type: WorkerType,
    base: u64,
    workers: u64,
) -> Arc<SelectionCore> {
    let config = KvRouterConfig {
        use_kv_events: true,
        router_queue_threshold: None,
        ..Default::default()
    };
    let core = SelectionCore::try_new_local_for(
        worker_type,
        config,
        1,
        CancellationToken::new(),
        SelectionCacheConfig::default(),
        Arc::new(|config, role, _| {
            dynamo_kv_router::WorkerSelectionPolicy::reference(
                config.clone(),
                role.default_selector_label(),
            )
        }),
    )
    .expect("core");
    runtime.block_on(async {
        for worker_id in (base + 1)..=(base + workers) {
            core.upsert_worker(WorkerRequest {
                worker_id,
                endpoint: Some(format!("http://worker-{worker_id}:8000")),
                kv_events_endpoint: Some(format!("tcp://127.0.0.1:{}", 40_000 + worker_id)),
                block_size: Some(BLOCK_SIZE),
                max_num_batched_tokens: Some(8192),
                ..WorkerRequest::default()
            })
            .await
            .expect("upsert");
        }
    });
    Arc::new(core)
}

/// Record the whole prompt as cached on `worker_id`, so a preview of this
/// core sees a full prefix hit there.
async fn seed_prefix(core: &SelectionCore, worker_id: u64) {
    let key = RoutingPartitionId::new(MODEL, "default");
    let local_hashes =
        compute_block_hash_for_seq(&prompt_tokens(), BLOCK_SIZE, BlockHashOptions::default());
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
    let blocks = local_hashes
        .iter()
        .zip(sequence_hashes.iter())
        .map(|(&tokens_hash, &sequence_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(sequence_hash),
            tokens_hash,
            mm_extra_info: None,
        })
        .collect();
    let indexer = core.partition(&key).expect("partition").indexer().clone();
    indexer
        .apply_event_routed(RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        ))
        .await
        .expect("seed index");
    if let Indexer::Single { primary, .. } = &indexer {
        primary.flush().await;
    }
}

fn request(id: &str) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: MODEL.to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(id.to_string()),
        prompt: PromptRequest {
            token_ids: Some(prompt_tokens()),
            ..PromptRequest::default()
        },
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        policy_class: None,
        all_now: false,
        export_bookings: false,
    }
}

/// Book what is bookable, forward what is ready, repeat until every stage
/// has run. The bookings stay held.
async fn drive(router: &dyn Router, req: &SelectAndReserveRequest) -> Plan {
    let mut plan = router.plan(req).expect("plan");
    loop {
        router.schedule(req, &mut plan).await.expect("schedule");
        let ready: Vec<usize> = plan.ready().collect();
        if ready.is_empty() {
            break;
        }
        for k in ready {
            let attempt = plan.dispatch(k).expect("dispatch");
            let worker = plan.worker(k).expect("worker");
            plan.complete(
                k,
                attempt,
                Outcome {
                    worker,
                    kv_hint: None,
                },
            )
            .expect("complete");
        }
    }
    assert!(!plan.has_pending(), "a stage was never booked");
    plan
}

/// Route one request to completion over a `Router`, then release every
/// booking and wait for the schedulers to acknowledge.
async fn route(router: &dyn Router, req: &SelectAndReserveRequest) {
    drive(router, req).await.release().await.expect("release");
}

/// One hand-composed stage on the `Lease` path, charged as `book_stage`
/// charges a planned stage of the same work: prefill tracking follows the
/// work, a prefill-only stage projects one output token, a decode-only stage
/// assumes no KV reuse.
async fn lease(core: &SelectionCore, id: String, work: StageWork) -> BookingHandle {
    let req = request(&id);
    let mut config = RouterConfigOverride {
        track_prefill_tokens: Some(matches!(
            work,
            StageWork::PrefillAndDecode | StageWork::PrefillOnly
        )),
        ..Default::default()
    };
    let mut expected_output_tokens = None;
    match work {
        StageWork::PrefillAndDecode | StageWork::None => {}
        StageWork::PrefillOnly => expected_output_tokens = Some(1),
        StageWork::DecodeOnly => config.assume_kv_reuse = Some(false),
    }
    let run = core
        .run_selection(SelectionOperation {
            key: RoutingPartitionId::new(MODEL, "default"),
            prompt: req.prompt.view(),
            router_config_override: Some(config),
            expected_output_tokens,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            session: SessionBinding::None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            admission: SelectionAdmission::Lease { request_id: id },
            track_active_blocks: work != StageWork::None,
            return_routing_hashes: false,
            replay_id: None,
            hold_budget: None,
        })
        .await;
    match run.result.expect("lease") {
        SelectionOutcome::Selected(selected) => selected.booking.expect("booking handle"),
        SelectionOutcome::QueueRejected { .. } => panic!("unexpected rejection"),
    }
}

/// The stages a hand-composed baseline books, in order.
type Composition = Vec<(Arc<SelectionCore>, StageWork)>;

/// A planned scenario: name, router, and whether the request asks for
/// every stage now.
type Planned = (&'static str, Arc<dyn Router>, bool);

/// Book every stage of a composition; the bookings stay held.
async fn compose(stages: &Composition, id: &str) -> Vec<BookingHandle> {
    let mut handles = Vec::with_capacity(stages.len());
    for (k, (core, work)) in stages.iter().enumerate() {
        handles.push(lease(core, format!("{id}/{k}"), *work).await);
    }
    handles
}

async fn release_all(handles: Vec<BookingHandle>) {
    for handle in handles {
        handle.release().await.expect("release");
    }
}

/// What the held bookings charge, per worker set, without worker identity:
/// equal-cost ties may land on different workers of a homogeneous set.
fn held_loads(cores: &[(&str, &Arc<SelectionCore>)]) -> Vec<(String, usize, usize, usize)> {
    let mut loads: Vec<_> = cores
        .iter()
        .flat_map(|(set, core)| {
            core.loads(None, None).into_iter().flat_map(move |model| {
                model
                    .loads
                    .into_iter()
                    .filter(|load| load.active_requests > 0)
                    .map(move |load| {
                        (
                            set.to_string(),
                            load.potential_prefill_tokens,
                            load.potential_decode_blocks,
                            load.active_requests,
                        )
                    })
            })
        })
        .collect();
    loads.sort();
    loads
}

/// Routes request number `sequence` once, on the given runtime.
type Route = Box<dyn FnMut(&tokio::runtime::Runtime, u64)>;

/// One scenario: a name and the work of routing one request.
struct Scenario {
    name: String,
    run: Route,
}

fn scenarios(runtime: &tokio::runtime::Runtime, workers: u64) -> Vec<Scenario> {
    let aggregated = core_for(runtime, WorkerType::Aggregated, 0, workers);
    let prefill = core_for(runtime, WorkerType::Prefill, 100, workers);
    let decode = core_for(runtime, WorkerType::Decode, 200, workers);
    // A second decode set with the prompt cached on one worker: the bypass case.
    let cached_decode = core_for(runtime, WorkerType::Decode, 300, workers);
    runtime.block_on(seed_prefix(&cached_decode, 301));
    let sets = [
        ("aggregated", &aggregated),
        ("prefill", &prefill),
        ("decode", &decode),
        ("cached_decode", &cached_decode),
    ];

    let multistage = |list: StageList, decode: &Arc<SelectionCore>, conditional: bool| {
        let mut builder = MultiStageRouter::builder()
            .set(WorkerType::Aggregated, aggregated.clone())
            .set(WorkerType::Prefill, prefill.clone())
            .set(WorkerType::Decode, decode.clone())
            .classes(ClassTable::new(list));
        if conditional {
            builder = builder
                .conditional_disagg(Arc::new(IslBoundingPolicy::new(true, 2048, 0.7)), false);
        }
        Arc::new(builder.build().expect("router")) as Arc<dyn Router>
    };
    let mut conditional_list = StageList::prefill_decode();
    conditional_list.stages[0].skip = Some(SkipRule::ConditionalDisagg);

    // Each baseline and the planned scenarios it stands in for.
    let baselines: Vec<(&str, Composition, Vec<Planned>)> = vec![
        (
            "direct/aggregated",
            vec![(aggregated.clone(), StageWork::PrefillAndDecode)],
            vec![("plan/aggregated", aggregated.clone(), false)],
        ),
        (
            "direct/prefill_decode",
            vec![
                (prefill.clone(), StageWork::PrefillOnly),
                (decode.clone(), StageWork::DecodeOnly),
            ],
            vec![
                (
                    "plan/prefill_decode",
                    multistage(StageList::prefill_decode(), &decode, false),
                    false,
                ),
                (
                    "plan/prefill_decode/all_now",
                    multistage(StageList::prefill_decode(), &decode, false),
                    true,
                ),
                (
                    "plan/prefill_decode_deferred",
                    multistage(
                        StageList::prefill_decode_deferred(Duration::from_secs(5)),
                        &decode,
                        false,
                    ),
                    false,
                ),
                (
                    "plan/conditional/remote",
                    multistage(conditional_list.clone(), &decode, true),
                    false,
                ),
            ],
        ),
        (
            "direct/decode_first",
            vec![
                (decode.clone(), StageWork::DecodeOnly),
                (prefill.clone(), StageWork::PrefillOnly),
            ],
            vec![(
                "plan/decode_first",
                multistage(StageList::decode_first(), &decode, false),
                false,
            )],
        ),
        (
            "direct/decode_only",
            vec![(cached_decode.clone(), StageWork::PrefillAndDecode)],
            vec![(
                "plan/conditional/bypass",
                multistage(conditional_list, &cached_decode, true),
                false,
            )],
        ),
    ];

    // The two conditional scenarios must take different paths, and every
    // planned scenario must charge what its baseline charges, or the
    // comparison measures nothing.
    for (baseline, composition, planned) in &baselines {
        let expected = runtime.block_on(async {
            let handles = compose(composition, "probe").await;
            let loads = held_loads(&sets);
            release_all(handles).await;
            loads
        });
        assert!(!expected.is_empty(), "{baseline}: the probe booked nothing");
        for (name, router, all_now) in planned {
            let mut req = request("probe");
            req.all_now = *all_now;
            let actual = runtime.block_on(async {
                let plan = drive(router.as_ref(), &req).await;
                let skipped = plan.state_of(0) == Some(&StageState::Skipped);
                assert_eq!(
                    skipped,
                    name.ends_with("/bypass"),
                    "{name}: the conditional probe took the wrong path"
                );
                let loads = held_loads(&sets);
                plan.release().await.expect("release");
                loads
            });
            assert_eq!(
                actual, expected,
                "{name} charges differently from {baseline}"
            );
        }
        assert!(
            held_loads(&sets).is_empty(),
            "{baseline}: a probe booking outlived its release"
        );
    }

    let mut scenarios = vec![Scenario {
        name: "direct/aggregated/book".to_string(),
        run: Box::new(move |runtime, sequence| {
            let id = format!("book-{sequence}");
            runtime.block_on(async {
                aggregated
                    .select_and_reserve(request(&id))
                    .await
                    .expect("reserve");
                aggregated.free_reservation(&id).await.expect("free");
            });
        }),
    }];
    for (baseline, composition, planned) in baselines {
        let name = baseline.to_string();
        scenarios.push(Scenario {
            name: name.clone(),
            run: Box::new(move |runtime, sequence| {
                runtime.block_on(async {
                    let handles = compose(&composition, &format!("{name}-{sequence}")).await;
                    release_all(handles).await;
                });
            }),
        });
        for (name, router, all_now) in planned {
            let name = name.to_string();
            scenarios.push(Scenario {
                name: name.clone(),
                run: Box::new(move |runtime, sequence| {
                    let mut req = request(&format!("{name}-{sequence}"));
                    req.all_now = all_now;
                    runtime.block_on(route(router.as_ref(), &req));
                }),
            });
        }
    }
    scenarios
}

fn bench_routes(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let workers = workers_per_set();
    let mut group = c.benchmark_group("router_plan/route_then_release");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    for mut scenario in scenarios(&runtime, workers) {
        let mut sequence = 0u64;
        group.bench_function(BenchmarkId::new(&scenario.name, workers), |b| {
            b.iter(|| {
                sequence += 1;
                (scenario.run)(&runtime, sequence);
            });
        });
    }
    group.finish();
}

fn percentiles(samples: usize) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let workers = workers_per_set();
    let warmup = samples / 10;
    println!("scenario\tworkers_per_set\tsamples\tp50_us\tp95_us\tp99_us\tmean_us\tmax_us");
    for mut scenario in scenarios(&runtime, workers) {
        let mut durations = Vec::with_capacity(samples);
        for sequence in 0..(warmup + samples) as u64 {
            let started = Instant::now();
            (scenario.run)(&runtime, sequence);
            if sequence >= warmup as u64 {
                durations.push(started.elapsed());
            }
        }
        durations.sort_unstable();
        let at =
            |q: f64| durations[((durations.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
        let mean =
            durations.iter().map(Duration::as_secs_f64).sum::<f64>() / durations.len() as f64 * 1e6;
        println!(
            "{}\t{workers}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}",
            scenario.name,
            durations.len(),
            at(0.50),
            at(0.95),
            at(0.99),
            mean,
            durations.last().expect("samples").as_secs_f64() * 1e6
        );
    }
}

fn main() {
    if let Some(samples) = std::env::var("ROUTER_BENCH_PERCENTILES")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        percentiles(samples);
        return;
    }
    let mut criterion = Criterion::default().configure_from_args();
    bench_routes(&mut criterion);
    criterion.final_summary();
}
