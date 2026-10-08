// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DEP #15457 request flows, placement rules and failure matrix over
//! the fake per-set router: one test per row of tmonty12/dynamo's
//! `coordinator_tests.rs`, same names and order, then the rows the Plan adds.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use crate::WorkerType;
use crate::conditional_disagg::{
    ConditionalDisaggPolicy, IslBoundingPolicy, PrefillLoadPolicy, make_conditional_disagg_policy,
};
use crate::protocols::{KvTransferEnforcement, RoutingConstraints, WorkerWithDpRank};
use crate::scheduling::KvSchedulerError;
use crate::scheduling::policy_config::RouterPolicyConfig;
use crate::services::selection::{
    MmRoutingInfoRequest, PromptRequest, SelectAndReserveRequest, SelectionError,
};

use super::class_table::{ClassTable, Fallback, StageList};
use super::fake::{FakeEvent, FakeRouter, FakeSignals, FakeThresholds, FakeWorker};
use super::multistage::{MultiStageRouter, RouterLimits};
use super::placement::topology_taint;
use super::plan::{
    Budget, Constraint, DomainMode, Failure, Outcome, Plan, PlanError, SkipRule, Stage,
    StageAttempt, StageState, StageWork, When, WorkerFacts,
};
use super::router_trait::Router;

const PROMPT: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

fn request(id: &str) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(id.to_string()),
        prompt: PromptRequest {
            token_ids: Some(PROMPT.to_vec()),
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

fn multimodal(mut req: SelectAndReserveRequest) -> SelectAndReserveRequest {
    req.prompt.mm_routing_info = Some(MmRoutingInfoRequest {
        routing_token_ids: vec![1],
        block_mm_infos: Vec::new(),
    });
    req
}

fn all_now(mut req: SelectAndReserveRequest) -> SelectAndReserveRequest {
    req.all_now = true;
    req
}

fn outcome(w: WorkerWithDpRank) -> Outcome {
    Outcome {
        worker: w,
        kv_hint: None,
    }
}

fn failure(is_retryable: bool) -> Failure {
    Failure {
        is_retryable,
        reason: "scripted".to_string(),
    }
}

fn zoned(worker_id: u64, zone: &str) -> FakeWorker {
    FakeWorker::new(worker_id).with_facts(WorkerFacts {
        taints: HashSet::from([topology_taint("zone", zone)]),
        topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
        ..WorkerFacts::default()
    })
}

fn transfer_worker(worker_id: u64, zone: &str, enforcement: KvTransferEnforcement) -> FakeWorker {
    let mut worker = zoned(worker_id, zone);
    worker.facts.kv_transfer_domain = Some("zone".to_string());
    worker.facts.kv_transfer_enforcement = Some(enforcement);
    worker.facts.kv_transfer_preferred_weight = Some(0.5);
    worker
}

fn cached(cached_tokens: usize, potential_decode_blocks: u64) -> FakeSignals {
    FakeSignals {
        cached_tokens,
        potential_decode_blocks,
        total_kv_blocks: Some(100),
        prefill_token_capacity: 1000,
        ..FakeSignals::default()
    }
}

/// `prefill_decode_deferred` with the conditional skip on prefill: the
/// progressive conditional flow.
fn conditional_deferred() -> StageList {
    let mut list = StageList::prefill_decode_deferred(Duration::from_secs(2));
    list.stages[0].skip = Some(SkipRule::ConditionalDisagg);
    list
}

fn isl_policy(enabled: bool) -> Arc<dyn ConditionalDisaggPolicy> {
    Arc::new(IslBoundingPolicy::new(enabled, 2048, 0.7))
}

struct Fixture {
    prefill: Arc<FakeRouter>,
    decode: Arc<FakeRouter>,
    encode: Arc<FakeRouter>,
    aggregated: Arc<FakeRouter>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            prefill: FakeRouter::with_workers(WorkerType::Prefill, [11, 12]),
            decode: FakeRouter::with_workers(WorkerType::Decode, [21, 22]),
            encode: FakeRouter::without_select(WorkerType::Encode, [31]),
            aggregated: FakeRouter::with_workers(WorkerType::Aggregated, [1, 2]),
        }
    }

    fn with_decode_gate(mut self, decode_busy: f64) -> Self {
        self.decode = FakeRouter::with_workers(WorkerType::Decode, [21, 22]).with_thresholds(
            FakeThresholds {
                prefill_busy: None,
                decode_busy: Some(decode_busy),
            },
        );
        self
    }

    /// Encoder 31 in zone `b`; decode 21 in zone `a`, 22 in zone `b`.
    fn zoned(self) -> Self {
        for (router, worker_id, zone) in [
            (&self.encode, 31, "b"),
            (&self.decode, 21, "a"),
            (&self.decode, 22, "b"),
        ] {
            router.remove_worker(worker_id);
            router.add_worker(zoned(worker_id, zone));
        }
        self
    }

    fn builder(&self) -> super::multistage::MultiStageRouterBuilder {
        MultiStageRouter::builder()
            .set(WorkerType::Prefill, self.prefill.clone())
            .set(WorkerType::Decode, self.decode.clone())
            .set(WorkerType::Encode, self.encode.clone())
            .set(WorkerType::Aggregated, self.aggregated.clone())
    }

    fn router(&self, list: StageList) -> MultiStageRouter {
        self.builder()
            .classes(ClassTable::new(list))
            .build()
            .expect("valid router")
    }

    fn conditional(
        &self,
        list: StageList,
        policy: Arc<dyn ConditionalDisaggPolicy>,
    ) -> MultiStageRouter {
        self.conditional_gated(list, policy, false)
    }

    fn conditional_gated(
        &self,
        list: StageList,
        policy: Arc<dyn ConditionalDisaggPolicy>,
        has_decode_gate: bool,
    ) -> MultiStageRouter {
        self.builder()
            .classes(ClassTable::new(list))
            .conditional_disagg(policy, has_decode_gate)
            .build()
            .expect("valid router")
    }

    fn outstanding(&self) -> usize {
        self.prefill.outstanding().len()
            + self.decode.outstanding().len()
            + self.encode.outstanding().len()
            + self.aggregated.outstanding().len()
    }
}

async fn booked(router: &MultiStageRouter, req: &SelectAndReserveRequest) -> Plan {
    let mut plan = router.plan(req).expect("plan");
    router.schedule(req, &mut plan).await.expect("schedule");
    plan
}

fn ready(plan: &Plan) -> Vec<usize> {
    plan.ready().collect()
}

fn state(plan: &Plan, k: usize) -> &StageState {
    plan.state_of(k).expect("stage")
}

fn books(router: &FakeRouter) -> usize {
    router
        .events()
        .iter()
        .filter(|event| matches!(event, FakeEvent::Book { .. }))
        .count()
}

fn previews(router: &FakeRouter) -> usize {
    router
        .events()
        .iter()
        .filter(|event| matches!(event, FakeEvent::Preview { .. }))
        .count()
}

/// Forward stage `k` and report its outcome.
fn run(plan: &mut Plan, k: usize) -> StageAttempt {
    let attempt = plan.dispatch(k).expect("ready to dispatch");
    let worker = plan.worker(k).expect("booked");
    plan.complete(k, attempt, outcome(worker))
        .expect("complete");
    attempt
}

// ---------------------------------------------------------------------------
// Request flows (DEP §5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn aggregated_serving_admits_executes_and_completes() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::aggregated());
    let req = request("agg-1");
    let mut plan = booked(&router, &req).await;
    assert_eq!(plan.stage_count(), 1);
    assert_eq!(ready(&plan), vec![0]);
    assert_eq!(fixture.aggregated.outstanding().len(), 1);
    run(&mut plan, 0);
    assert_eq!(
        fixture.aggregated.outstanding().len(),
        1,
        "held until release"
    );
    assert!(!plan.has_pending());
    plan.release().await.unwrap();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn progressive_prefill_decode_waits_for_the_handoff() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("pd-1");
    let mut plan = booked(&router, &req).await;
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert!(state(&plan, 1).is_pending());
    assert_eq!(fixture.decode.outstanding().len(), 0);
    assert_eq!(ready(&plan), vec![0]);

    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.booking(1).unwrap().id(), "pd-1/1/0");
    assert_eq!(ready(&plan), vec![1]);
    run(&mut plan, 1);
    assert_eq!(fixture.outstanding(), 2);
    plan.release().await.unwrap();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn plan_all_selects_every_stage_before_execution() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let plan = booked(&router, &all_now(request("plan-1"))).await;
    assert_eq!(
        plan.stages().map(|stage| stage.set).collect::<Vec<_>>(),
        vec![WorkerType::Prefill, WorkerType::Decode]
    );
    assert_eq!(plan.stage(1).unwrap().inputs, vec![0]);
    assert_eq!(plan.stage(1).unwrap().wait, Budget::Immediate);
    assert_eq!(fixture.outstanding(), 2);
    assert_eq!(ready(&plan), vec![0], "decode still runs after prefill");
    plan.release().await.unwrap();
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("plan-1/0/0"), 1);
    assert_eq!(fixture.decode.release_count("plan-1/1/0"), 1);
}

#[tokio::test]
async fn plan_all_orders_a_decode_first_selection_by_execution_dependency() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::decode_first());
    let plan = booked(&router, &all_now(request("plan-2"))).await;
    assert_eq!(
        plan.stage(0).unwrap().set,
        WorkerType::Decode,
        "decode was booked first"
    );
    assert_eq!(ready(&plan), vec![1], "but prefill runs first");
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn progressive_decode_first_executes_prefill_first() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::decode_first());
    let mut plan = booked(&router, &request("df-1")).await;
    assert_eq!(
        state(&plan, 0),
        &StageState::Booked,
        "decode is booked and held"
    );
    assert_eq!(ready(&plan), vec![1]);
    run(&mut plan, 1);
    assert_eq!(ready(&plan), vec![0]);
    run(&mut plan, 0);
    assert!(!plan.has_pending());
}

#[tokio::test]
async fn conditional_disagg_bypasses_to_the_previewed_decode_worker() {
    let fixture = Fixture::new();
    fixture.decode.set_signals(22, cached(15, 1));
    fixture.decode.set_signals(21, cached(0, 5));
    let router = fixture.conditional(StageList::conditional_prefill_decode(), isl_policy(true));
    let plan = booked(&router, &request("cond-1")).await;
    assert_eq!(state(&plan, 0), &StageState::Skipped);
    assert_eq!(plan.worker(1).unwrap().worker_id, 22);
    assert!(matches!(
        fixture.decode.events().as_slice(),
        [FakeEvent::Preview { worker }, FakeEvent::Book { worker: booked, .. }]
            if worker.worker_id == 22 && booked.worker_id == 22
    ));
    assert!(fixture.prefill.events().is_empty());
    assert_eq!(
        ready(&plan),
        vec![1],
        "nothing to wait for: prefill was skipped"
    );
}

#[tokio::test]
async fn conditional_bypass_survives_placement_rules() {
    let fixture = Fixture::new();
    fixture.decode.remove_worker(21);
    fixture.decode.set_signals(22, cached(16, 0));
    let mut list = StageList::conditional_prefill_decode();
    list.stages[1].constraints.push(Constraint::SameDomain {
        stage: 0,
        key: "zone".to_string(),
        mode: DomainMode::Required,
    });
    let router = fixture.conditional(list, isl_policy(true));
    let plan = booked(&router, &request("cond-7")).await;
    assert_eq!(state(&plan, 0), &StageState::Skipped);
    assert_eq!(
        plan.worker(1).unwrap().worker_id,
        22,
        "rules reading a skipped stage derive nothing"
    );
    assert!(matches!(
        fixture.decode.events().as_slice(),
        [FakeEvent::Preview { .. }, FakeEvent::Book { .. }]
    ));
}

#[tokio::test]
async fn conditional_disagg_takes_remote_prefill_when_the_policy_declines() {
    let fixture = Fixture::new();
    let router = fixture.conditional(conditional_deferred(), isl_policy(true));
    let req = request("cond-2");
    let mut plan = booked(&router, &req).await;
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert!(matches!(
        fixture.decode.events().as_slice(),
        [FakeEvent::Preview { .. }]
    ));
    assert!(state(&plan, 1).is_pending());
    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(ready(&plan), vec![1]);
}

#[tokio::test]
async fn conditional_disagg_decode_gate_vetoes_a_busy_decode_worker() {
    let fixture = Fixture::new().with_decode_gate(0.9);
    fixture.decode.remove_worker(21);
    fixture.decode.set_signals(22, cached(15, 95));
    let router = fixture.conditional_gated(
        StageList::conditional_prefill_decode(),
        isl_policy(true),
        true,
    );
    let plan = booked(&router, &request("cond-3")).await;
    assert_eq!(
        state(&plan, 0),
        &StageState::Booked,
        "the busy decode worker vetoed the bypass"
    );
    assert_eq!(previews(&fixture.decode), 1);
    assert!(
        !plan
            .stage(1)
            .unwrap()
            .constraints
            .iter()
            .any(|c| matches!(c, Constraint::Pin(_))),
        "remote prefill: decode is not pinned to the preview"
    );
}

#[tokio::test]
async fn conditional_disagg_skips_the_decision_for_a_pinned_prefill_worker() {
    let fixture = Fixture::new();
    fixture.decode.set_signals(22, cached(16, 0));
    let mut list = StageList::conditional_prefill_decode();
    list.stages[0]
        .constraints
        .push(Constraint::Pin(WorkerWithDpRank::new(12, 0)));
    let router = fixture.conditional(list, isl_policy(true));
    let plan = booked(&router, &request("cond-4")).await;
    assert_eq!(plan.worker(0).unwrap().worker_id, 12);
    assert_eq!(
        previews(&fixture.decode),
        0,
        "no decode preview for a pinned prefill"
    );
}

#[tokio::test]
async fn conditional_disagg_disabled_goes_straight_to_remote_prefill() {
    let fixture = Fixture::new();
    let router = fixture.conditional(StageList::conditional_prefill_decode(), isl_policy(false));
    let plan = booked(&router, &all_now(request("cond-5"))).await;
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(state(&plan, 1), &StageState::Booked);
    assert_eq!(previews(&fixture.decode), 0);
}

#[tokio::test]
async fn conditional_disagg_consults_prefill_load_when_the_policy_needs_it() {
    let mut fixture = Fixture::new();
    fixture.prefill =
        FakeRouter::with_workers(WorkerType::Prefill, [11]).with_thresholds(FakeThresholds {
            prefill_busy: Some(0.5),
            decode_busy: None,
        });
    fixture.prefill.set_signals(
        11,
        FakeSignals {
            active_prefill_tokens: 900,
            prefill_token_capacity: 1000,
            ..FakeSignals::default()
        },
    );
    let router = fixture.conditional(
        StageList::conditional_prefill_decode(),
        Arc::new(PrefillLoadPolicy::new(true)),
    );
    let plan = booked(&router, &request("cond-6")).await;
    assert_eq!(
        state(&plan, 0),
        &StageState::Skipped,
        "busy prefill: decode does the prefill"
    );
    assert_eq!(previews(&fixture.prefill), 1);
    assert_eq!(fixture.prefill.outstanding().len(), 0);
}

#[tokio::test]
async fn encode_prefill_decode_passes_each_handoff_in_order() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::encode_prefill_decode());
    let req = multimodal(request("epd-1"));
    let mut plan = booked(&router, &req).await;
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert!(state(&plan, 1).is_pending() && state(&plan, 2).is_pending());
    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(state(&plan, 1), &StageState::Booked);
    assert!(state(&plan, 2).is_pending());
    assert_eq!(plan.stage(1).unwrap().inputs, vec![0]);
    run(&mut plan, 1);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.stage(2).unwrap().inputs, vec![1]);
    run(&mut plan, 2);
    assert!(!plan.has_pending());
}

#[tokio::test]
async fn encode_prefill_decode_skips_encode_without_multimodal_input() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::encode_prefill_decode());
    let plan = booked(&router, &all_now(request("epd-2"))).await;
    assert_eq!(state(&plan, 0), &StageState::Skipped);
    assert_eq!(state(&plan, 1), &StageState::Booked);
    assert_eq!(state(&plan, 2), &StageState::Booked);
    assert!(fixture.encode.events().is_empty());
}

// ---------------------------------------------------------------------------
// Cross-stage constraints (DEP §4.5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn same_domain_rule_constrains_decode_to_the_prefill_zone() {
    let fixture = Fixture::new();
    fixture.prefill.remove_worker(11);
    fixture.prefill.remove_worker(12);
    fixture.prefill.add_worker(zoned(11, "b"));
    fixture.decode.remove_worker(21);
    fixture.decode.remove_worker(22);
    fixture.decode.add_worker(zoned(21, "a"));
    fixture
        .decode
        .add_worker(zoned(22, "b").with_signals(cached(0, 50)));
    let mut list = StageList::prefill_decode();
    list.stages[1].constraints.push(Constraint::SameDomain {
        stage: 0,
        key: "zone".to_string(),
        mode: DomainMode::Required,
    });
    let router = fixture.router(list);
    let plan = booked(&router, &request("zone-1")).await;
    assert_eq!(plan.worker(0).unwrap().worker_id, 11);
    assert_eq!(
        plan.worker(1).unwrap().worker_id,
        22,
        "lighter 21 is in the wrong zone"
    );
}

#[tokio::test]
async fn unsatisfiable_placement_releases_the_held_reservation() {
    let fixture = Fixture::new();
    fixture.prefill.remove_worker(11);
    fixture.prefill.remove_worker(12);
    fixture
        .prefill
        .add_worker(transfer_worker(11, "a", KvTransferEnforcement::Required));
    fixture.decode.remove_worker(21);
    fixture.decode.remove_worker(22);
    fixture.decode.add_worker(zoned(21, "b"));
    let router = fixture.router(StageList::prefill_decode());
    let req = request("zone-2");
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(matches!(error, SelectionError::Scheduler(_)), "{error}");
    // The host owns the plan: prefill is still held until it decides.
    assert_eq!(state(&plan, 0), &StageState::Booked);
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("zone-2/0/0"), 1);
}

#[tokio::test]
async fn decode_first_selection_still_satisfies_the_prefill_transfer_requirement() {
    let fixture = Fixture::new();
    fixture.prefill.remove_worker(11);
    fixture.prefill.remove_worker(12);
    fixture
        .prefill
        .add_worker(transfer_worker(11, "a", KvTransferEnforcement::Required));
    fixture.decode.remove_worker(21);
    fixture.decode.remove_worker(22);
    fixture.decode.add_worker(zoned(21, "b"));
    let router = fixture.router(StageList::decode_first());
    let req = request("zone-3");
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(matches!(error, SelectionError::Conflict(_)), "{error}");
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn caller_restrictions_apply_only_to_the_named_stage() {
    let fixture = Fixture::new();
    let mut list = StageList::prefill_decode();
    list.stages[1].constraints.push(Constraint::Exclude(21));
    let router = fixture.router(list);
    let plan = booked(&router, &request("subset-1")).await;
    assert_eq!(plan.worker(1).unwrap().worker_id, 22);
    assert_eq!(
        plan.worker(0).unwrap().worker_id,
        11,
        "the prefill set is not narrowed"
    );
}

// ---------------------------------------------------------------------------
// Lifecycle and correctness (DEP §6) and the failure-injection matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_before_admission_completes_leaves_nothing_reserved() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("cancel-1");
    let mut plan = router.plan(&req).unwrap();
    fixture.prefill.set_admission_open(false);
    {
        let schedule = router.schedule(&req, &mut plan);
        tokio::pin!(schedule);
        assert!(futures_util::poll!(schedule.as_mut()).is_pending());
        // Dropping the pinned future cancels the in-flight admission.
    }
    assert!(fixture.prefill.events().is_empty());
    assert!(state(&plan, 0).is_pending());
    // The plan is still usable once admission opens.
    fixture.prefill.set_admission_open(true);
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.attempt(0), Some(StageAttempt::FIRST));
    assert_eq!(state(&plan, 0), &StageState::Booked);
}

#[tokio::test]
async fn cancel_between_admission_and_transfer_releases_the_held_stage() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::decode_first());
    let mut plan = booked(&router, &request("cancel-2")).await;
    assert_eq!(fixture.decode.outstanding().len(), 1);
    // Cancelled before prefill was dispatched: the host declines to dispatch
    // and fails the booked stage; decode read it, so decode is released too.
    let attempt = plan.attempt(1).unwrap();
    plan.fail(1, attempt, failure(false)).unwrap();
    assert!(state(&plan, 0).is_pending());
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn cancel_after_prefill_dispatch_still_routes_decode_for_kv_cleanup() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("cancel-3");
    let mut plan = booked(&router, &req).await;
    let attempt = plan.dispatch(0).unwrap();
    // The client is gone, but prefill is staging KV for one decode worker:
    // the host keeps the plan and routes decode when prefill completes.
    plan.complete(0, attempt, outcome(plan.worker(0).unwrap()))
        .unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.stage(1).unwrap().inputs, vec![0]);
    run(&mut plan, 1);
    plan.release().await.unwrap();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn cancel_with_a_held_dependent_of_a_dispatched_producer_keeps_it() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::decode_first());
    let mut plan = booked(&router, &request("cancel-4")).await;
    let attempt = plan.dispatch(1).unwrap();
    // Cancellation arrives; prefill already reached a worker, so the host
    // keeps the plan: decode stays reserved and nothing on it changes.
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(fixture.decode.outstanding().len(), 1);
    plan.complete(1, attempt, outcome(plan.worker(1).unwrap()))
        .unwrap();
    assert_eq!(ready(&plan), vec![0]);
    run(&mut plan, 0);
}

#[tokio::test]
async fn dropping_the_host_side_after_transfer_releases_the_reservation() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::aggregated());
    let mut plan = booked(&router, &request("drop-1")).await;
    let _ = plan.dispatch(0).unwrap();
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.aggregated.release_count("drop-1"), 1);

    let plan = booked(&router, &all_now(request("drop-2"))).await;
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn retry_fences_the_old_attempts_handoff_and_failure() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("retry-1");
    let mut plan = booked(&router, &req).await;
    let first = plan.dispatch(0).unwrap();
    assert_eq!(first, StageAttempt::FIRST);
    let failed_worker = plan.worker(0).unwrap();
    assert_eq!(failed_worker.worker_id, 11);

    plan.fail(0, first, failure(true)).unwrap();
    plan.retry(0).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    let second = plan.attempt(0).unwrap();
    assert_eq!(second, first.next());
    assert_ne!(plan.worker(0).unwrap(), failed_worker);
    assert_eq!(plan.booking(0).unwrap().id(), "retry-1/0/1");

    // Late events for the old attempt change nothing.
    assert!(matches!(
        plan.complete(0, first, outcome(failed_worker)),
        Err(PlanError::StaleAttempt { .. })
    ));
    assert!(state(&plan, 1).is_pending());
    assert!(matches!(
        plan.fail(0, first, failure(true)),
        Err(PlanError::StaleAttempt { .. })
    ));
    assert_eq!(state(&plan, 0), &StageState::Booked);

    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    run(&mut plan, 1);
    assert!(!plan.has_pending());
}

#[tokio::test]
async fn duplicate_terminal_events_are_idempotent() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("dup-1");
    let mut plan = booked(&router, &req).await;
    let attempt = plan.dispatch(0).unwrap();
    let worker = plan.worker(0).unwrap();
    plan.complete(0, attempt, outcome(worker)).unwrap();
    plan.complete(0, attempt, outcome(worker)).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(books(&fixture.decode), 1, "decode is booked once");
    run(&mut plan, 1);
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn second_stage_admission_failure_releases_the_first_reservation() {
    let fixture = Fixture::new();
    fixture.decode.fail_next(SelectionError::Scheduler(
        KvSchedulerError::AllEligibleWorkersOverloaded,
    ));
    let router = fixture.router(StageList::prefill_decode());
    let req = all_now(request("fail-1"));
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(matches!(error, SelectionError::Scheduler(_)), "{error}");
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("fail-1/0/0"), 1);
}

#[tokio::test]
async fn admission_failure_in_a_progressive_session_aborts_it() {
    let fixture = Fixture::new();
    fixture.decode.fail_next(SelectionError::Scheduler(
        KvSchedulerError::AllEligibleWorkersFiltered,
    ));
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("fail-2");
    let mut plan = booked(&router, &req).await;
    run(&mut plan, 0);
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::AllEligibleWorkersFiltered)
        ),
        "{error}"
    );
    assert!(matches!(state(&plan, 0), StageState::Completed(_)));
    assert!(state(&plan, 1).is_pending());
    // Prefill is host-owned; the host releases it.
    assert!(plan.booking(0).is_some());
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn stale_pool_generation_between_preview_and_admission_is_rejected() {
    let fixture = Fixture::new();
    fixture.decode.remove_worker(21);
    fixture.decode.set_signals(22, cached(16, 0));
    fixture.decode.leaves_after_preview(22);
    // There is no pool generation here: the previewed worker is pinned and
    // has gone by the time it is booked, so the pinned booking fails.
    let router = fixture.conditional(StageList::conditional_prefill_decode(), isl_policy(true));
    let req = request("stale-1");
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(matches!(error, SelectionError::Scheduler(_)), "{error}");
    assert_eq!(state(&plan, 0), &StageState::Skipped);
    assert!(state(&plan, 1).is_pending());
    assert_eq!(fixture.decode.outstanding().len(), 0);
}

#[tokio::test]
async fn admission_timeout_releases_everything_the_session_holds() {
    let fixture = Fixture::new();
    fixture.decode.set_admission_open(false);
    let router = fixture.router(StageList::prefill_decode());
    let req = all_now(request("timeout-1"));
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
        ),
        "{error}"
    );
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("timeout-1/0/0"), 1);
}

#[tokio::test(start_paused = true)]
async fn a_reservation_held_too_long_aborts_the_session() {
    let fixture = Fixture::new();
    let router = fixture
        .builder()
        .classes(ClassTable::new(StageList::decode_first()))
        .limits(RouterLimits {
            max_hold: Duration::from_secs(1),
            ..RouterLimits::default()
        })
        .build()
        .unwrap();
    let req = request("hold-1");
    let mut plan = booked(&router, &req).await;
    run(&mut plan, 1);
    tokio::time::advance(Duration::from_secs(2)).await;
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
        ),
        "{error}"
    );
    // The router refuses to continue; the booking is the host's to release.
    assert_eq!(fixture.decode.outstanding().len(), 1);
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn dropping_a_session_releases_what_it_owns() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::decode_first());
    let plan = booked(&router, &request("drop-session")).await;
    assert_eq!(fixture.decode.outstanding().len(), 1);
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn retry_exhaustion_fails_the_stage_and_releases_dependents() {
    let fixture = Fixture::new();
    let router = fixture
        .builder()
        .classes(ClassTable::new(StageList::decode_first()))
        .limits(RouterLimits {
            max_attempts: 2,
            ..RouterLimits::default()
        })
        .build()
        .unwrap();
    let req = request("exhaust-1");
    let mut plan = booked(&router, &req).await;
    let first_decode = fixture.decode.outstanding()[0].clone();
    let first = plan.dispatch(1).unwrap();
    plan.fail(1, first, failure(true)).unwrap();
    assert!(
        state(&plan, 0).is_pending(),
        "the held decode read the failed stage"
    );
    assert_eq!(fixture.decode.release_count(&first_decode), 1);
    plan.retry(1).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(plan.booking(1).unwrap().id(), "exhaust-1/1/1");
    assert_eq!(
        fixture.decode.outstanding(),
        vec!["exhaust-1/0/1".to_string()]
    );

    let second = plan.dispatch(1).unwrap();
    plan.fail(1, second, failure(true)).unwrap();
    plan.retry(1).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(matches!(error, SelectionError::Conflict(_)), "{error}");
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn non_retryable_failure_completes_routing_after_releasing_held_stages() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::decode_first());
    let mut plan = booked(&router, &request("fatal-1")).await;
    let attempt = plan.dispatch(1).unwrap();
    plan.fail(1, attempt, failure(false)).unwrap();
    assert!(matches!(
        plan.retry(1),
        Err(PlanError::InvalidTransition { .. })
    ));
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn upfront_planning_rejects_a_policy_that_waits() {
    // There is no policy that "waits": every stage's constraints read
    // bookings, never outcomes, so any list books upfront. What `all_now`
    // rejects is a stage that cannot be booked on this pass.
    let fixture = Fixture::new();
    fixture.decode.set_admission_open(false);
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = all_now(request("wait-1"));
    let mut plan = router.plan(&req).unwrap();
    assert!(plan.stages().all(|stage| stage.when == When::Now));
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
        ),
        "{error}"
    );
    plan.abort();
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("wait-1/0/0"), 1);
}

#[tokio::test]
async fn finishing_with_an_unadmitted_stage_is_a_policy_error() {
    // The Plan's form of the rule: when `schedule` returns Ok, no stage
    // that could be booked is still pending. Only a deferred stage waits.
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode());
    let plan = booked(&router, &request("finish-1")).await;
    assert_eq!(plan.schedulable().count(), 0);
    assert!(!plan.has_pending());
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let plan = booked(&router, &request("finish-2")).await;
    assert_eq!(plan.schedulable().count(), 0);
    assert!(
        plan.has_pending(),
        "the deferred decode waits for prefill's outcome"
    );
}

#[tokio::test]
async fn unbound_topology_stages_are_rejected_at_construction() {
    let fixture = Fixture::new();
    let error = MultiStageRouter::builder()
        .set(WorkerType::Prefill, fixture.prefill.clone())
        .classes(ClassTable::new(StageList::prefill_decode()))
        .build()
        .err()
        .expect("an unbound stage must be rejected");
    assert!(matches!(error, SelectionError::BadRequest(message) if message.contains("decode")));
}

#[tokio::test]
async fn preview_is_refused_for_a_stage_without_preview_support() {
    let mut fixture = Fixture::new();
    fixture.decode = FakeRouter::without_select(WorkerType::Decode, [21, 22]);
    let router = fixture.conditional(StageList::conditional_prefill_decode(), isl_policy(true));
    let req = request("enc-1");
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(matches!(error, SelectionError::BadRequest(_)), "{error}");
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn registry_built_policies_drive_the_coordinator() {
    let fixture = Fixture::new();
    let config = RouterPolicyConfig::from_yaml(
        r#"
default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: cond
    policy_family: regular
    cache_bucket: all
    quantum: 128
    stages: conditional_prefill_decode
  - name: agg
    quantum: 128
    stages: aggregated
"#,
    )
    .unwrap();
    let profile = config.resolve_profile(None, None, Default::default());
    let classes = ClassTable::from_profile(&profile, StageList::aggregated()).unwrap();
    let policy = make_conditional_disagg_policy(Some(&crate::config::KvRouterConfig {
        conditional_disagg_enabled: true,
        ..Default::default()
    }));
    let router = fixture
        .builder()
        .classes(classes)
        .conditional_disagg(Arc::from(policy), false)
        .build()
        .unwrap();

    // The request names the routing family; the scheduler resolves the
    // physical class (`cond`) by cached-token bucket for queue settings.
    let mut req = request("reg-1");
    req.policy_class = Some("regular".to_string());
    let plan = booked(&router, &req).await;
    assert_eq!(
        state(&plan, 0),
        &StageState::Booked,
        "nothing cached on decode: remote prefill"
    );
    assert_eq!(plan.stage_count(), 2);

    let mut req = all_now(request("reg-2"));
    req.policy_class = Some("agg".to_string());
    let plan = booked(&router, &req).await;
    assert_eq!(plan.stage_count(), 1);
}

// ---------------------------------------------------------------------------
// Rows the Plan adds
// ---------------------------------------------------------------------------

#[tokio::test]
async fn skip_rules_are_decided_before_a_set_books_a_stage_they_cover() {
    // Two encode stages; the second becomes bookable only once the first is
    // booked, inside the encode router's call. Its skip rule must already
    // have been applied by then.
    let fixture = Fixture::new();
    let router = fixture.router(StageList::new(vec![
        Stage::new(WorkerType::Encode),
        Stage {
            constraints: vec![Constraint::TransferCompatible(0)],
            skip: Some(SkipRule::NoMultimodal),
            ..Stage::new(WorkerType::Encode)
        },
    ]));
    let plan = booked(&router, &request("two-encodes")).await;
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(state(&plan, 1), &StageState::Skipped);
    assert_eq!(books(&fixture.encode), 1);
}

#[tokio::test]
async fn the_conditional_preview_respects_the_decode_stages_own_constraints() {
    let fixture = Fixture::new();
    fixture.decode.set_signals(21, cached(16, 0));
    fixture.decode.set_signals(22, cached(16, 5));
    let mut list = StageList::conditional_prefill_decode();
    list.stages[1]
        .constraints
        .push(Constraint::Pin(WorkerWithDpRank::new(22, 0)));
    let router = fixture.conditional(list, isl_policy(true));
    let plan = booked(&router, &request("pinned-decode")).await;
    assert_eq!(state(&plan, 0), &StageState::Skipped);
    assert_eq!(
        plan.worker(1).unwrap().worker_id,
        22,
        "the preview honoured the pin"
    );

    // An excluded worker can be previewed but never booked: keep remote prefill.
    let fixture = Fixture::new();
    fixture.decode.set_signals(21, cached(16, 0));
    let mut list = StageList::conditional_prefill_decode();
    list.stages[1].constraints.push(Constraint::Exclude(21));
    let router = fixture.conditional(list, isl_policy(true));
    let plan = booked(&router, &request("excluded-decode")).await;
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(plan.worker(1).unwrap().worker_id, 22);
}

#[tokio::test]
async fn fallback_does_not_hide_a_non_capacity_error() {
    let fixture = Fixture::new();
    fixture.prefill.fail_next(SelectionError::Scheduler(
        KvSchedulerError::InvalidClassificationMetadata("bad class".to_string()),
    ));
    let router = fixture.router(StageList::prefill_decode().with_fallback(Fallback::Aggregated));
    let req = request("no-fallback");
    let mut plan = router.plan(&req).unwrap();
    let error = router.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::InvalidClassificationMetadata(_))
        ),
        "{error}"
    );
    assert_eq!(plan.stage_count(), 2, "no fallback plan was substituted");
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn stage_lists_attach_to_the_family_a_request_names() {
    let yaml = |second: &str| {
        format!(
            r#"
default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: short
  - min_tokens: 1024
    bucket: long
policy_classes:
  - name: regular-short
    policy_family: regular
    cache_bucket: short
    quantum: 128
    stages: prefill_decode
  - name: regular-long
    policy_family: regular
    cache_bucket: long
    quantum: 128
    stages: {second}
"#
        )
    };
    let config = RouterPolicyConfig::from_yaml(&yaml("prefill_decode")).unwrap();
    let profile = config.resolve_profile(None, None, Default::default());
    let classes = ClassTable::from_profile(&profile, StageList::aggregated()).unwrap();
    assert_eq!(
        classes.stages(Some("regular")),
        &StageList::prefill_decode()
    );
    assert_eq!(
        classes.stages(None),
        &StageList::prefill_decode(),
        "no class names the default family, which declares a list"
    );
    assert_eq!(
        classes.stages(Some("regular-short")),
        &StageList::prefill_decode(),
        "an unknown name is queued under the default family and routes by it"
    );

    let config = RouterPolicyConfig::from_yaml(&yaml("decode_first")).unwrap();
    let profile = config.resolve_profile(None, None, Default::default());
    assert!(ClassTable::from_profile(&profile, StageList::aggregated()).is_err());
}

/// A bucket that omits `stages` inherits the family's list, and the default
/// family's list comes from that same mapping: omitting the family, naming an
/// unknown one, and naming the default family all route the same way.
#[test]
fn a_bucket_without_a_stage_list_inherits_the_familys_list() {
    let config = RouterPolicyConfig::from_yaml(
        r#"
default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: short
  - min_tokens: 1024
    bucket: long
policy_classes:
  - name: regular-short
    policy_family: regular
    cache_bucket: short
    quantum: 128
  - name: regular-long
    policy_family: regular
    cache_bucket: long
    quantum: 128
    stages: prefill_decode
"#,
    )
    .unwrap();
    let profile = config.resolve_profile(None, None, Default::default());
    let classes = ClassTable::from_profile(&profile, StageList::aggregated()).unwrap();
    for name in [None, Some("regular"), Some("unknown")] {
        assert_eq!(
            classes.stages(name),
            &StageList::prefill_decode(),
            "{name:?} routes by the declaration the long bucket carries"
        );
    }

    let config = RouterPolicyConfig::from_yaml(
        r#"
default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: regular
    policy_family: regular
    cache_bucket: all
    quantum: 128
  - name: direct
    quantum: 128
    stages: decode_first
"#,
    )
    .unwrap();
    let profile = config.resolve_profile(None, None, Default::default());
    let classes = ClassTable::from_profile(&profile, StageList::aggregated()).unwrap();
    assert_eq!(
        classes.stages(None),
        &StageList::aggregated(),
        "a default family declaring nothing routes by the fallback, not by another class"
    );
    assert_eq!(classes.stages(Some("direct")), &StageList::decode_first());
}

/// Encode on a zone, then prefill or a bypass, then decode in the encoder's
/// zone: the preview must look where the decode stage may actually book.
fn zoned_encode_conditional_decode() -> StageList {
    StageList::new(vec![
        Stage::new(WorkerType::Encode),
        Stage {
            when: When::After(0),
            skip: Some(SkipRule::ConditionalDisagg),
            ..Stage::new(WorkerType::Prefill)
        },
        Stage {
            inputs: vec![1],
            constraints: vec![
                Constraint::TransferCompatible(1),
                Constraint::SameDomain {
                    stage: 0,
                    key: "zone".to_string(),
                    mode: DomainMode::Required,
                },
            ],
            ..Stage::new(WorkerType::Decode)
        },
    ])
}

#[tokio::test]
async fn the_conditional_preview_honours_the_decode_stages_topology_rules() {
    let fixture = Fixture::new().zoned();
    fixture.decode.set_signals(21, cached(16, 0));
    fixture.decode.set_signals(22, cached(16, 0));
    let router = fixture.conditional(zoned_encode_conditional_decode(), isl_policy(true));
    let req = request("zoned-bypass");
    let mut plan = booked(&router, &req).await;
    assert_eq!(plan.worker(0).unwrap().worker_id, 31);
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
        "the preview chose a decode worker in the encoder's zone, not the lower-numbered one"
    );
    assert_eq!(fixture.outstanding(), 2);

    // The same rule read before the encoder is booked cannot be previewed:
    // prefill stays remote rather than guessing a decode placement.
    let fixture = Fixture::new().zoned();
    fixture.decode.set_signals(21, cached(16, 0));
    fixture.decode.set_signals(22, cached(16, 0));
    let mut list = zoned_encode_conditional_decode();
    list.stages[1].when = When::Now;
    let router = fixture.conditional(list, isl_policy(true));
    let plan = booked(&router, &request("unresolved")).await;
    assert_eq!(
        state(&plan, 1),
        &StageState::Booked,
        "eligibility of the decode stage was not yet established"
    );
    assert_eq!(previews(&fixture.decode), 0);
}

#[tokio::test]
async fn a_bypassed_decode_can_be_retried_elsewhere() {
    let fixture = Fixture::new();
    fixture.decode.set_signals(22, cached(16, 0));
    fixture.decode.set_signals(21, cached(0, 5));
    let router = fixture.conditional(StageList::conditional_prefill_decode(), isl_policy(true));
    let req = request("bypass-retry");
    let mut plan = booked(&router, &req).await;
    assert_eq!(
        plan.worker(1).unwrap().worker_id,
        22,
        "pinned to the preview"
    );
    let attempt = plan.dispatch(1).unwrap();
    plan.fail(1, attempt, failure(true)).unwrap();
    plan.retry(1).unwrap();
    router
        .schedule(&req, &mut plan)
        .await
        .expect("the pin went with the failed worker");
    assert_eq!(plan.worker(1).unwrap().worker_id, 21);
    assert_eq!(
        state(&plan, 0),
        &StageState::Skipped,
        "prefill stays skipped"
    );
    assert_eq!(
        plan.work_of(1),
        Some(StageWork::PrefillAndDecode),
        "the uncached worker does the prefill itself and is charged for it"
    );
}

#[tokio::test]
async fn an_unknown_decode_signal_denies_the_bypass_only_when_a_gate_is_configured() {
    let fixture = Fixture::new();
    fixture.decode.set_signals(
        22,
        FakeSignals {
            total_kv_blocks: None,
            ..cached(16, 0)
        },
    );
    fixture.decode.set_signals(21, cached(0, 5));
    let gated = fixture.conditional_gated(
        StageList::conditional_prefill_decode(),
        isl_policy(true),
        true,
    );
    let plan = booked(&gated, &request("gate-1")).await;
    assert_eq!(
        state(&plan, 0),
        &StageState::Booked,
        "unknown capacity: no bypass"
    );
    let ungated = fixture.conditional(StageList::conditional_prefill_decode(), isl_policy(true));
    let plan = booked(&ungated, &request("gate-2")).await;
    assert_eq!(state(&plan, 0), &StageState::Skipped);
}

#[tokio::test(start_paused = true)]
async fn a_schedule_deadline_caps_a_full_wait_only_when_configured() {
    let fixture = Fixture::new();
    fixture.aggregated.set_admission_open(false);
    let capped = fixture
        .builder()
        .classes(ClassTable::new(StageList::aggregated()))
        .limits(RouterLimits {
            schedule_deadline: Some(Duration::from_millis(100)),
            ..RouterLimits::default()
        })
        .build()
        .unwrap();
    let req = request("deadline-1");
    let mut plan = capped.plan(&req).unwrap();
    let error = capped.schedule(&req, &mut plan).await.unwrap_err();
    assert!(
        matches!(
            error,
            SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
        ),
        "{error}"
    );
    assert!(state(&plan, 0).is_pending());

    let uncapped = fixture.router(StageList::aggregated());
    let mut plan = uncapped.plan(&req).unwrap();
    let schedule = uncapped.schedule(&req, &mut plan);
    tokio::pin!(schedule);
    tokio::time::advance(Duration::from_secs(3600)).await;
    assert!(
        futures_util::poll!(schedule.as_mut()).is_pending(),
        "a Full wait is the class's to end"
    );
    fixture.aggregated.set_admission_open(true);
    schedule.await.unwrap();
}

#[tokio::test]
async fn immediate_wait_books_both_or_nothing() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode());
    let plan = booked(&router, &request("both-1")).await;
    assert_eq!(fixture.outstanding(), 2);
    drop(plan);

    fixture.decode.set_admission_open(false);
    let req = request("both-2");
    let mut plan = router.plan(&req).unwrap();
    assert!(router.schedule(&req, &mut plan).await.is_err());
    plan.abort();
    assert_eq!(fixture.outstanding(), 0, "nothing is held across the gap");
}

#[tokio::test]
async fn deferred_stage_returns_pending_then_books_on_second_schedule() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::prefill_decode_deferred(Duration::from_secs(2)));
    let req = request("deferred-1");
    let mut plan = booked(&router, &req).await;
    assert!(plan.has_pending());
    router.schedule(&req, &mut plan).await.unwrap();
    assert!(
        plan.has_pending(),
        "prefill has not completed: still deferred"
    );
    run(&mut plan, 0);
    router.schedule(&req, &mut plan).await.unwrap();
    assert!(!plan.has_pending());
}

#[tokio::test]
async fn fallback_to_agg_when_prefill_set_saturated() {
    let fixture = Fixture::new();
    fixture.prefill.set_admission_open(false);
    let mut list = StageList::prefill_decode();
    list.stages[0].wait = Budget::Immediate;
    let router = fixture.router(list.with_fallback(Fallback::Aggregated));
    let req = request("fallback-1");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.expect("fell back");
    assert_eq!(plan.stage_count(), 1);
    assert_eq!(plan.stage(0).unwrap().set, WorkerType::Decode);
    assert_eq!(fixture.decode.outstanding(), vec!["fallback-1".to_string()]);
    assert!(fixture.prefill.outstanding().is_empty());
}

#[tokio::test]
async fn all_now_books_every_stage_in_one_call() {
    let fixture = Fixture::new();
    let router = fixture.router(StageList::encode_prefill_decode());
    let plan = booked(&router, &all_now(multimodal(request("now-1")))).await;
    assert!(plan.stages().all(|stage| stage.wait == Budget::Immediate));
    assert_eq!(fixture.outstanding(), 3);
    assert_eq!(
        ready(&plan),
        vec![0],
        "execution order is still encode → prefill → decode"
    );
}
