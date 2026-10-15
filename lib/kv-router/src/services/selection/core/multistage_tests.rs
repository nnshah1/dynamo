// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `MultiStageRouter` over real prefill and decode cores: what each stage's
//! booking is charged for, on the remote-prefill and the bypass paths.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use super::super::input::MmRoutingInfoRequest;
use super::tests::{core_with, reserve_request, test_config, wait_until, worker};
use super::*;
use crate::conditional_disagg::{ConditionalDisaggDecisionInput, ConditionalDisaggPolicy};
use crate::protocols::PotentialLoad;
use crate::router::{
    ClassTable, Constraint, DomainMode, Failure, MultiStageRouter, Plan, Router, SkipRule, Stage,
    StageList, StageState, When, topology_taint,
};
use crate::scheduling::KvSchedulerError;
use crate::services::selection::{SelectRequest, SelectResponse};

fn typed_core(worker_type: WorkerType) -> Arc<SelectionCore> {
    typed_core_with(worker_type, test_config(false))
}

fn typed_core_with(
    worker_type: WorkerType,
    config: crate::config::KvRouterConfig,
) -> Arc<SelectionCore> {
    Arc::new(core_with(
        config,
        SelectionHost::default(),
        None,
        worker_type,
        None,
    ))
}

fn loads(core: &SelectionCore) -> Vec<PotentialLoad> {
    core.loads(Some("model"), Some("default"))
        .first()
        .map(|model| model.loads.clone())
        .unwrap_or_default()
}

fn prefill_tokens(core: &SelectionCore) -> Option<usize> {
    loads(core)
        .first()
        .map(|load| load.potential_prefill_tokens)
}

fn prefill_tokens_of(core: &SelectionCore, worker_id: WorkerId) -> usize {
    loads(core)
        .iter()
        .find(|load| load.worker_id == worker_id)
        .map(|load| load.potential_prefill_tokens)
        .unwrap_or_else(|| panic!("worker {worker_id} has no load"))
}

async fn ready(core: &SelectionCore) {
    wait_until("slot tracker sees the worker", || {
        prefill_tokens(core).is_some()
    })
    .await;
}

async fn ready_for(core: &SelectionCore, workers: usize) {
    wait_until("slot tracker sees every worker", || {
        loads(core).len() == workers
    })
    .await;
}

fn zoned(worker_id: WorkerId, zone: &str) -> WorkerRequest {
    WorkerRequest {
        taints: HashSet::from([topology_taint("zone", zone)]),
        topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
        ..worker(worker_id)
    }
}

fn multimodal(mut req: SelectAndReserveRequest) -> SelectAndReserveRequest {
    req.prompt.mm_routing_info = Some(MmRoutingInfoRequest {
        routing_token_ids: vec![1],
        block_mm_infos: Vec::new(),
    });
    req
}

/// The conditional policy at its most permissive, so a test can watch where
/// the bypass lands without staging a cache hit on a real core.
struct AlwaysBypass;

#[async_trait::async_trait]
impl ConditionalDisaggPolicy for AlwaysBypass {
    fn is_enabled(&self) -> bool {
        true
    }

    async fn should_bypass_remote_prefill(&self, _input: ConditionalDisaggDecisionInput) -> bool {
        true
    }
}

fn same_zone_as(stage: usize) -> Constraint {
    Constraint::SameDomain {
        stage,
        key: "zone".to_string(),
        mode: DomainMode::Required,
    }
}

/// Forward stage `k` and report its outcome.
fn run(plan: &mut crate::router::Plan, k: usize) {
    let attempt = plan.mark_dispatched(k).expect("ready to dispatch");
    let worker = plan.worker(k).expect("booked");
    plan.mark_completed(k, attempt, worker).expect("complete");
}

#[tokio::test]
async fn remote_decode_is_not_charged_for_the_prompt_but_a_bypass_decode_is() {
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    prefill.upsert_worker(worker(11)).await.unwrap();
    decode.upsert_worker(worker(21)).await.unwrap();
    ready(&prefill).await;
    ready(&decode).await;
    let router = MultiStageRouter::builder()
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::new(StageList::prefill_decode()))
        .build()
        .unwrap();

    // Remote prefill: the prompt is prefill's load, not decode's.
    let req = reserve_request("pd");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(state(&plan, 1), &StageState::Booked);
    assert_eq!(prefill_tokens(&prefill), Some(4));
    assert_eq!(
        prefill_tokens(&decode),
        Some(0),
        "a remote-prefill decode holds no prompt load"
    );
    plan.abort();
    wait_until("release", || prefill_tokens(&prefill) == Some(0)).await;

    // Prefill skipped (a bypass): decode does the prefill and is charged.
    let req = reserve_request("bypass");
    let mut plan = router.plan(&req).unwrap();
    plan.skip(0).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(
        prefill_tokens(&decode),
        Some(4),
        "a local-prefill decode is charged for the prompt"
    );
    assert_eq!(prefill_tokens(&prefill), Some(0));
    plan.abort();
}

fn state(plan: &crate::router::Plan, k: usize) -> &StageState {
    plan.state_of(k).expect("stage")
}

#[tokio::test]
async fn a_decode_core_that_does_not_track_prefill_still_pays_for_a_bypass() {
    let prefill = typed_core(WorkerType::Prefill);
    let mut decode_config = test_config(false);
    decode_config.router_track_prefill_tokens = false;
    let decode = typed_core_with(WorkerType::Decode, decode_config);
    prefill.upsert_worker(worker(11)).await.unwrap();
    decode.upsert_worker(worker(21)).await.unwrap();
    decode.upsert_worker(worker(22)).await.unwrap();
    ready(&prefill).await;
    ready_for(&decode, 2).await;
    let router = MultiStageRouter::builder()
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::new(StageList::prefill_decode()))
        .build()
        .unwrap();

    let req = reserve_request("remote");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    let remote = plan.worker(1).unwrap().worker_id;
    assert_eq!(prefill_tokens_of(&decode, remote), 0);
    plan.abort();
    wait_until("release", || prefill_tokens(&prefill) == Some(0)).await;

    // The bypass: the decode core's role default says "no prefill here",
    // the stage's work says otherwise, and the stage's work wins.
    let req = reserve_request("bypass");
    let mut plan = router.plan(&req).unwrap();
    plan.skip(0).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    let first = plan.worker(1).unwrap().worker_id;
    assert_eq!(
        prefill_tokens_of(&decode, first),
        4,
        "a local-prefill decode is charged for the prompt"
    );

    // Retried onto the other, uncached worker: the charge moves with it.
    let attempt = plan.mark_dispatched(1).unwrap();
    plan.mark_failed(
        1,
        attempt,
        Failure {
            is_retryable: true,
            reason: "scripted".to_string(),
        },
    )
    .unwrap();
    plan.retry(1).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    let second = plan.worker(1).unwrap().worker_id;
    assert_ne!(second, first);
    assert_eq!(prefill_tokens_of(&decode, second), 4);
    wait_until("the failed booking is released", || {
        prefill_tokens_of(&decode, first) == 0
    })
    .await;
    assert_eq!(prefill_tokens(&prefill), Some(0));
    plan.abort();
}

#[tokio::test]
async fn an_encoder_booking_holds_no_prompt_tokens_or_kv_blocks() {
    let encode = typed_core(WorkerType::Encode);
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    encode.upsert_worker(worker(31)).await.unwrap();
    prefill.upsert_worker(worker(11)).await.unwrap();
    decode.upsert_worker(worker(21)).await.unwrap();
    ready(&encode).await;
    ready(&prefill).await;
    ready(&decode).await;
    let router = MultiStageRouter::builder()
        .set(WorkerType::Encode, encode.clone())
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::new(StageList::encode_prefill_decode()))
        .build()
        .unwrap();
    let req = multimodal(reserve_request("encode"));
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(
        state(&plan, 1),
        &StageState::Pending,
        "prefill waits for encode"
    );
    let load = &loads(&encode)[0];
    assert_eq!(load.potential_prefill_tokens, 0);
    assert_eq!(
        load.potential_decode_blocks, 0,
        "an encoder holds no prompt KV blocks"
    );
    assert_eq!(load.active_requests, 1, "but its booking is live");
    plan.abort();
    wait_until("release", || loads(&encode)[0].active_requests == 0).await;
}

#[tokio::test]
async fn target_stage_pins_exclusions_and_topology_hold_through_real_admission() {
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    prefill.upsert_worker(zoned(11, "a")).await.unwrap();
    decode.upsert_worker(zoned(21, "a")).await.unwrap();
    decode.upsert_worker(zoned(22, "b")).await.unwrap();
    ready(&prefill).await;
    ready_for(&decode, 2).await;
    let router = |decode_constraints: Vec<Constraint>| {
        let mut list = StageList::prefill_decode();
        list.stages[1].constraints = decode_constraints;
        MultiStageRouter::builder()
            .set(WorkerType::Prefill, prefill.clone())
            .set(WorkerType::Decode, decode.clone())
            .classes(ClassTable::new(list))
            .build()
            .unwrap()
    };

    // Topology: decode in prefill's zone.
    let req = reserve_request("zone");
    let mut plan = router(vec![same_zone_as(0)]).plan(&req).unwrap();
    router(vec![same_zone_as(0)])
        .schedule(&req, &mut plan)
        .await
        .unwrap();
    assert_eq!(plan.worker(1).unwrap().worker_id, 21);
    plan.abort();

    // A hard pin to the other zone's worker.
    let req = reserve_request("pin");
    let pinned = router(vec![Constraint::Pin(WorkerWithDpRank::new(22, 0))]);
    let mut plan = pinned.plan(&req).unwrap();
    pinned.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.worker(1).unwrap().worker_id, 22);
    plan.abort();

    // Excluding the only in-zone worker leaves nothing eligible: a deliberate
    // error, not a silent placement elsewhere.
    let req = reserve_request("excluded");
    let excluded = router(vec![same_zone_as(0), Constraint::Exclude(21)]);
    let mut plan = excluded.plan(&req).unwrap();
    let error = excluded.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::NoEndpoints)
        ),
        "the selector reports a required taint nobody carries as no endpoints: {error}"
    );
    assert_eq!(
        state(&plan, 0),
        &StageState::Booked,
        "prefill is held for a retry"
    );
    plan.abort();
    wait_until("release", || prefill_tokens(&prefill) == Some(0)).await;
}

#[tokio::test]
async fn the_conditional_preview_places_decode_where_topology_allows() {
    let encode = typed_core(WorkerType::Encode);
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    encode.upsert_worker(zoned(31, "b")).await.unwrap();
    prefill.upsert_worker(zoned(11, "a")).await.unwrap();
    decode.upsert_worker(zoned(21, "a")).await.unwrap();
    decode.upsert_worker(zoned(22, "b")).await.unwrap();
    ready(&encode).await;
    ready(&prefill).await;
    ready_for(&decode, 2).await;
    let list = StageList::new(vec![
        Stage::new(WorkerType::Encode),
        Stage {
            when: When::After(0),
            skip: Some(SkipRule::ConditionalDisagg),
            ..Stage::new(WorkerType::Prefill)
        },
        Stage {
            inputs: vec![1],
            constraints: vec![Constraint::TransferCompatible(1), same_zone_as(0)],
            ..Stage::new(WorkerType::Decode)
        },
    ]);
    let router = MultiStageRouter::builder()
        .set(WorkerType::Encode, encode.clone())
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::new(list))
        .conditional_disagg(Arc::new(AlwaysBypass), false)
        .build()
        .unwrap();
    let req = reserve_request("zoned-bypass");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(
        state(&plan, 1),
        &StageState::Skipped,
        "the short prompt bypasses prefill"
    );
    assert_eq!(
        plan.worker(2).unwrap().worker_id,
        22,
        "the preview chose the decode worker in the encoder's zone"
    );
    assert_eq!(
        prefill_tokens_of(&decode, 22),
        4,
        "and that worker is charged for the prompt it prefills"
    );
    assert_eq!(prefill_tokens(&prefill), Some(0));
    plan.abort();
}

#[tokio::test]
async fn requests_resolve_their_stage_list_through_the_policy_profile() {
    let yaml = |regular: &str| {
        format!(
            r#"
default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: regular-all
    policy_family: regular
    cache_bucket: all
    quantum: 128
    {regular}
  - name: direct
    quantum: 128
    stages: aggregated
"#
        )
    };
    let cores_for = |yaml: String| {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        let mut config = test_config(false);
        config.router_policy_config = Some(file.path().display().to_string());
        let profile = config.policy_profile(Some("model")).unwrap();
        let cores = [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ]
        .map(|set| typed_core_with(set, config.clone()));
        (file, profile, cores)
    };

    // The default family declares a list: a request naming no class, or an
    // unknown name, routes by it; an explicit class routes by its own.
    let (_file, profile, [aggregated, prefill, decode]) = cores_for(yaml("stages: prefill_decode"));
    aggregated.upsert_worker(worker(1)).await.unwrap();
    prefill.upsert_worker(worker(11)).await.unwrap();
    decode.upsert_worker(worker(21)).await.unwrap();
    for core in [&aggregated, &prefill, &decode] {
        ready(core).await;
    }
    let router = MultiStageRouter::builder()
        .set(WorkerType::Aggregated, aggregated.clone())
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::from_profile(&profile, StageList::aggregated()).unwrap())
        .build()
        .unwrap();
    for (id, class) in [("none", None), ("unknown", Some("no-such-class"))] {
        let mut req = reserve_request(id);
        req.policy_class = class.map(str::to_string);
        let mut plan = router.plan(&req).unwrap();
        router.schedule(&req, &mut plan).await.unwrap();
        assert_eq!(plan.stage_count(), 2, "{id}: the default family's list");
        assert_eq!(prefill_tokens(&prefill), Some(4));
        assert_eq!(prefill_tokens(&aggregated), Some(0));
        plan.abort();
        wait_until("release", || prefill_tokens(&prefill) == Some(0)).await;
    }
    let mut req = reserve_request("direct");
    req.policy_class = Some("direct".to_string());
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.stage_count(), 1, "the explicit class's own list");
    assert_eq!(prefill_tokens(&aggregated), Some(4));
    plan.abort();

    // The default family declares no list: the table's fallback applies.
    let (_file, profile, _) = cores_for(yaml(""));
    let classes = ClassTable::from_profile(&profile, StageList::decode_first()).unwrap();
    assert_eq!(classes.stages(None), &StageList::decode_first());
    assert_eq!(classes.stages(Some("direct")), &StageList::aggregated());
}

/// Record the whole prompt as cached on `worker_id` in the core's index.
async fn seed_prompt(core: &SelectionCore, worker_id: WorkerId, tokens: &[u32]) {
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
        compute_block_hash_for_seq, compute_seq_hash_for_block,
    };
    use crate::services::indexer::backend::Indexer;
    let key = super::tests::default_key();
    let block_size = core.entry(&key).expect("entry").block_size;
    let local_hashes = compute_block_hash_for_seq(tokens, block_size, BlockHashOptions::default());
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

/// The second stage selects with the block hashes the first stage left on
/// the plan: with the real hashes it finds worker 2's cached prompt, with
/// garbage on the plan it does not.
#[tokio::test]
async fn a_later_stage_sees_the_cached_prompt() {
    let core = typed_core_with(WorkerType::Aggregated, test_config(false));
    core.upsert_worker(worker(1)).await.unwrap();
    core.upsert_worker(worker(2)).await.unwrap();
    ready_for(&core, 2).await;
    let key = super::tests::default_key();
    let block_size = core.entry(&key).expect("entry").block_size;
    let tokens: Vec<u32> = (1..=4 * block_size).collect();
    seed_prompt(&core, 2, &tokens).await;

    let router = MultiStageRouter::builder()
        .set(WorkerType::Aggregated, core.clone() as Arc<dyn Router>)
        .classes(ClassTable::new(StageList::new(vec![
            Stage {
                constraints: vec![Constraint::Pin(WorkerWithDpRank::new(1, 0))],
                ..Stage::new(WorkerType::Aggregated)
            },
            Stage {
                constraints: vec![Constraint::Exclude(1)],
                ..Stage::new(WorkerType::Aggregated)
            },
        ])))
        .build()
        .unwrap();
    let mut req = reserve_request("real");
    req.prompt.token_ids = Some(tokens.clone());
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.worker(0).unwrap().worker_id, 1);
    assert_eq!(plan.worker(1).unwrap().worker_id, 2);
    assert_eq!(
        prefill_tokens_of(&core, 1),
        tokens.len(),
        "worker 1 has nothing cached"
    );
    let charged_on_cached_worker = prefill_tokens_of(&core, 2);
    assert!(
        charged_on_cached_worker < tokens.len() / 2,
        "the second stage finds worker 2's cached prompt: charged {charged_on_cached_worker}"
    );
    plan.release().await.unwrap();
}

/// The gateway's list: a prefill stage its own reservation router records
/// untracked (no plan booking), then decode placed against it. Decode must
/// read prefill's worker and facts through the untracked record.
struct UntrackedPrefill;

#[async_trait::async_trait]
impl Router for UntrackedPrefill {
    async fn select(
        &self,
        _req: crate::services::selection::SelectRequest,
    ) -> Result<crate::services::selection::SelectResponse, SelectionError> {
        Err(SelectionError::NotReady("no preview".to_string()))
    }

    fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        let mut stage = Stage::new(WorkerType::Prefill);
        stage.work = Some(crate::router::StageWork::None);
        Ok(Plan::new(
            crate::router::PlanId::from(req.selection_id.clone().unwrap_or_default()),
            crate::identity::RoutingPartitionId::new(&req.model_name, &req.routing_group),
            vec![stage],
        )?)
    }

    async fn schedule(
        &self,
        _req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        loop {
            let next = plan
                .schedulable()
                .find(|&k| plan.stage(k).is_some_and(|s| s.set == WorkerType::Prefill));
            let Some(k) = next else { break };
            plan.book_untracked(
                k,
                WorkerWithDpRank::new(99, 0),
                Arc::new(crate::test_utils::SimpleWorkerConfig::default()),
            )?;
        }
        Ok(())
    }
}

#[tokio::test]
async fn decode_is_booked_after_an_untracked_prefill_selection() {
    let decode = typed_core(WorkerType::Decode);
    decode.upsert_worker(worker(1)).await.unwrap();
    ready(&decode).await;
    let mut list = StageList::prefill_decode();
    list.stages[0].work = Some(crate::router::StageWork::None);
    let router = MultiStageRouter::builder()
        .set(
            WorkerType::Prefill,
            Arc::new(UntrackedPrefill) as Arc<dyn Router>,
        )
        .set(WorkerType::Decode, decode.clone() as Arc<dyn Router>)
        .classes(ClassTable::new(
            list.with_fallback(crate::router::Fallback::Aggregated),
        ))
        .build()
        .unwrap();
    let req = reserve_request("gateway-untracked-prefill");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.worker(0).map(|w| w.worker_id), Some(99));
    assert!(plan.booking(0).is_none(), "prefill is untracked");
    assert_eq!(plan.state_of(1), Some(&StageState::Booked));
    assert_eq!(plan.worker(1).map(|w| w.worker_id), Some(1));
    plan.release().await.unwrap();
}

/// Forget the whole prompt on `worker_id` in the core's index: the mirror of
/// `seed_prompt`.
async fn evict_prompt(core: &SelectionCore, worker_id: WorkerId, tokens: &[u32]) {
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheRemoveData, RouterEvent, StorageTier, compute_block_hash_for_seq,
        compute_seq_hash_for_block,
    };
    use crate::services::indexer::backend::Indexer;
    let key = super::tests::default_key();
    let block_size = core.entry(&key).expect("entry").block_size;
    let local_hashes = compute_block_hash_for_seq(tokens, block_size, BlockHashOptions::default());
    let block_hashes = compute_seq_hash_for_block(&local_hashes)
        .into_iter()
        .map(ExternalSequenceBlockHash)
        .collect();
    let indexer = core.partition(&key).expect("partition").indexer().clone();
    indexer
        .apply_event_routed(RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                dp_rank: 0,
            },
            StorageTier::Device,
        ))
        .await
        .expect("evict");
    if let Indexer::Single { primary, .. } = &indexer {
        primary.flush().await;
    }
}

/// A previewed pin is admitted against the worker's state at admission, not
/// at preview. A decode worker whose placement changed under the same id
/// since the preview is refused by the live check and nothing is kept for
/// it; one whose cache was evicted since the preview is booked and charged
/// for the whole prompt it now prefills itself.
#[tokio::test]
async fn a_previewed_pin_is_admitted_against_the_workers_current_state() {
    // Placement: the preview chose 22 while it shared prefill's zone.
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    prefill.upsert_worker(zoned(11, "a")).await.unwrap();
    decode.upsert_worker(zoned(21, "a")).await.unwrap();
    decode.upsert_worker(zoned(22, "a")).await.unwrap();
    ready(&prefill).await;
    ready_for(&decode, 2).await;
    let router = MultiStageRouter::builder()
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::new(StageList::new(vec![
            Stage::new(WorkerType::Prefill),
            Stage {
                inputs: vec![0],
                constraints: vec![same_zone_as(0)],
                ..Stage::new(WorkerType::Decode)
            },
        ])))
        .build()
        .unwrap();
    let req = reserve_request("stale-preview");
    let mut plan = router.plan(&req).unwrap();
    plan.constrain(1, Constraint::Previewed(WorkerWithDpRank::new(22, 0)))
        .unwrap();
    // The set is rebuilt under the same id: 22 is now in zone b.
    decode.upsert_worker(zoned(22, "b")).await.unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert_eq!(
        state(&plan, 0),
        &StageState::Booked,
        "prefill is booked: {error}"
    );
    assert!(
        state(&plan, 1).is_pending(),
        "the stale preview is not admitted"
    );
    assert_eq!(
        prefill_tokens_of(&decode, 22),
        0,
        "nothing was kept on the refused worker"
    );
    plan.release().await.unwrap();

    // Cache: the preview saw the prompt on 22; it is gone by admission.
    let decode = typed_core(WorkerType::Decode);
    decode.upsert_worker(worker(22)).await.unwrap();
    ready(&decode).await;
    let key = super::tests::default_key();
    let block_size = decode.entry(&key).expect("entry").block_size;
    let tokens: Vec<u32> = (1..=4 * block_size).collect();
    seed_prompt(&decode, 22, &tokens).await;
    let previewed = |id: &str| {
        let mut plan = Plan::new(
            crate::router::PlanId::from(id),
            key.clone(),
            vec![Stage::new(WorkerType::Decode)],
        )
        .unwrap();
        plan.constrain(0, Constraint::Previewed(WorkerWithDpRank::new(22, 0)))
            .unwrap();
        plan
    };
    let mut req = reserve_request("cached");
    req.prompt.token_ids = Some(tokens.clone());
    let mut hit = previewed("cached");
    Router::schedule(&*decode, &req, &mut hit).await.unwrap();
    let charged_with_cache = prefill_tokens_of(&decode, 22);
    assert!(
        charged_with_cache < tokens.len(),
        "with the prompt cached the booking is charged less than the prompt: {charged_with_cache}"
    );
    hit.release().await.unwrap();
    wait_until("release", || prefill_tokens_of(&decode, 22) == 0).await;

    evict_prompt(&decode, 22, &tokens).await;
    let mut req = reserve_request("evicted");
    req.prompt.token_ids = Some(tokens.clone());
    let mut miss = previewed("evicted");
    Router::schedule(&*decode, &req, &mut miss).await.unwrap();
    assert_eq!(
        plan_worker(&miss),
        22,
        "the previewed worker is still the one booked"
    );
    assert_eq!(
        prefill_tokens_of(&decode, 22),
        tokens.len(),
        "and it is charged for the whole prompt it now prefills"
    );
    miss.release().await.unwrap();
}

fn plan_worker(plan: &Plan) -> WorkerId {
    plan.worker(0).unwrap().worker_id
}

/// A custom `Router` over real cores reads the scheduler's recorded answer
/// for an earlier stage and steers a later one from it: here, decode is
/// pinned to worker 22 once prefill's answer names worker 11.
struct SteerFromPrefillAnswer(MultiStageRouter);

#[async_trait::async_trait]
impl Router for SteerFromPrefillAnswer {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        self.0.select(req).await
    }

    fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        self.0.plan(req)
    }

    async fn schedule(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        self.0.schedule(req, plan).await?;
        let prefill_on_11 = plan
            .scheduling(0)
            .is_some_and(|answer| answer.best_worker.worker_id == 11);
        let decode_unpinned = plan.state_of(1).is_some_and(StageState::is_pending)
            && plan.stage(1).is_some_and(|stage| {
                !stage
                    .constraints
                    .iter()
                    .any(|constraint| matches!(constraint, Constraint::Pin(_)))
            });
        if prefill_on_11 && decode_unpinned {
            plan.constrain(1, Constraint::Pin(WorkerWithDpRank::new(22, 0)))?;
        }
        Ok(())
    }
}

#[tokio::test]
async fn a_custom_router_steers_from_a_real_stages_scheduling_answer() {
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    prefill.upsert_worker(worker(11)).await.unwrap();
    decode.upsert_worker(worker(21)).await.unwrap();
    decode.upsert_worker(worker(22)).await.unwrap();
    ready(&prefill).await;
    ready_for(&decode, 2).await;
    let router = SteerFromPrefillAnswer(
        MultiStageRouter::builder()
            .set(WorkerType::Prefill, prefill.clone())
            .set(WorkerType::Decode, decode.clone())
            .classes(ClassTable::new(StageList::prefill_decode_deferred(
                Duration::from_secs(2),
            )))
            .build()
            .unwrap(),
    );
    let req = reserve_request("steer-real");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    let answer = plan.scheduling(0).expect("prefill's answer is on the plan");
    assert_eq!(answer.best_worker.worker_id, 11);
    assert_eq!(answer.cached_tokens, 0, "a cold prompt");
    assert!(state(&plan, 1).is_pending(), "decode is deferred");
    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(
        plan.worker(1).unwrap().worker_id,
        22,
        "decode followed the custom router's reading of prefill's real answer"
    );
    plan.release().await.unwrap();
}
