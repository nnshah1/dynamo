// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend as the host of a Plan-returning [`Router`].
//!
//! One [`HostSetRouter`] per worker set wraps the [`RoutingHost`]'s advisory
//! preview and admitted selection. Each admission produces the host's own
//! [`RoutePlan`], which is parked beside the plan (the *sidecar*) until the
//! host dispatches it through `dispatch_kv_plan` exactly as the legacy path
//! does. The plan records the booking as `Committed`: the route plan's
//! cleanup owns the lease, and dropping the plan frees nothing.
//!
//! The router library sees bookings, placement and accounting. The host keeps
//! dispatch, response streams, cleanup, affinity holds and cache tracking.

/// A [`Router`] over a bare [`KvRouter`](crate::kv_router::KvRouter) for
/// id-based hosts (the gateway EPP).
pub mod wire;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Error;
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::{WorkerId, WorkerWithDpRank};
use dynamo_kv_router::router::{
    Booking, Budget, Constraint, Plan, PlanId, Router, Stage, StageWork, WorkerFacts,
};
use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_kv_router::services::overlap::MooncakeOverlapSummary;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionError,
    SelectionWorkerLoad,
};
use dynamo_runtime::pipeline::{Context, SingleIn, async_trait};

use super::routing_host::{CleanupBudget, RoutePlan, RoutePlanSignals, RoutePreview, RoutingHost};
use crate::protocols::common::{llm_backend::PreprocessedRequest, timing::RequestPhase};

/// The conditional-disagg thresholds the legacy host consults; the set router
/// turns them into the `worker_load` / `decode_busy` signals a preview reports.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BusyThresholds {
    pub prefill: Option<f64>,
    pub decode: Option<f64>,
}

/// What the host keeps beside a booked stage until it dispatches it.
pub(crate) struct StageSide {
    pub plan: RoutePlan,
    pub signals: RoutePlanSignals,
}

/// A [`Router`] over one worker set's [`RoutingHost`].
pub(crate) struct HostSetRouter {
    host: Arc<RoutingHost>,
    set: WorkerType,
    phase: RequestPhase,
    /// The request to select with: shares the dispatch request's controller
    /// (id, cancellation) and metadata; the dispatched request is untouched.
    base: Context<PreprocessedRequest>,
    /// Normal disaggregation keeps decode routing load-only (zero overlap
    /// credit); conditional disaggregation lets the base credit apply.
    allow_decode_overlap_affinity: bool,
    thresholds: BusyThresholds,
    staged_kv_cleanup: AtomicBool,
    /// The host preview behind the last advisory `select`, so an admission
    /// pinned to that worker continues its cleanup budget.
    pending_preview: Mutex<Option<(WorkerWithDpRank, RoutePreview)>>,
    sidecar: Mutex<HashMap<usize, StageSide>>,
}

impl HostSetRouter {
    pub(crate) fn new(
        host: Arc<RoutingHost>,
        set: WorkerType,
        phase: RequestPhase,
        request: &SingleIn<PreprocessedRequest>,
        allow_decode_overlap_affinity: bool,
        thresholds: BusyThresholds,
    ) -> Self {
        Self {
            host,
            set,
            phase,
            base: request.fork(request.content().clone()),
            allow_decode_overlap_affinity,
            thresholds,
            staged_kv_cleanup: AtomicBool::new(false),
            pending_preview: Mutex::new(None),
            sidecar: Mutex::new(HashMap::new()),
        }
    }

    /// A remote prefill has staged KV for this request's decode leg: decode
    /// selection must go on through a client disconnect.
    pub(crate) fn set_staged_kv_cleanup(&self, staged: bool) {
        self.staged_kv_cleanup.store(staged, Ordering::Release);
    }

    /// The admitted route for stage `k`, for the host to dispatch.
    pub(crate) fn take_side(&self, k: usize) -> Option<StageSide> {
        self.sidecar.lock().expect("sidecar lock").remove(&k)
    }

    pub(crate) fn partition(&self) -> RoutingPartitionId {
        self.host.kv_router().selection.partition_key().clone()
    }

    fn facts(&self, worker_id: WorkerId) -> WorkerFacts {
        self.host
            .kv_router()
            .workers_with_configs
            .borrow()
            .get(&worker_id)
            .map(WorkerFacts::from_config)
            .unwrap_or_default()
    }

    fn pin(&self, body: &mut PreprocessedRequest, worker: WorkerWithDpRank) {
        let routing = body.routing_mut();
        match self.phase {
            RequestPhase::Prefill => {
                routing.prefill_worker_id = Some(worker.worker_id);
                routing.prefill_dp_rank = Some(worker.dp_rank);
            }
            RequestPhase::Decode | RequestPhase::Aggregated => {
                routing.decode_worker_id = Some(worker.worker_id);
                routing.dp_rank = Some(worker.dp_rank);
            }
        }
    }

    /// The worker the rendered request pins for this phase, by the host's
    /// own hint precedence; an overload on it is that worker's, not the set's.
    fn pinned_worker_id(&self, body: &PreprocessedRequest) -> Option<WorkerId> {
        let routing = body.routing.as_ref()?;
        match self.phase {
            RequestPhase::Prefill => routing.prefill_worker_id.or(routing.backend_instance_id),
            RequestPhase::Decode | RequestPhase::Aggregated => {
                routing.decode_worker_id.or(routing.backend_instance_id)
            }
        }
    }

    fn exclude(&self, body: &mut PreprocessedRequest, excluded: &HashSet<WorkerId>) {
        if excluded.is_empty() {
            return;
        }
        let all: HashSet<WorkerId> = self
            .host
            .kv_router()
            .workers_with_configs
            .borrow()
            .keys()
            .copied()
            .collect();
        let routing = body.routing_mut();
        let allowed = routing.allowed_worker_ids.take().unwrap_or(all);
        routing.allowed_worker_ids = Some(
            allowed
                .into_iter()
                .filter(|id| !excluded.contains(id))
                .collect(),
        );
    }

    /// The selection request for stage `k`: the base request with the stage's
    /// accounting, derived placement constraints, pins and exclusions.
    fn render(
        &self,
        req: &SelectAndReserveRequest,
        plan: &Plan,
        k: usize,
    ) -> Result<SingleIn<PreprocessedRequest>, SelectionError> {
        let mut body = self.base.content().clone();
        body.staged_kv_cleanup = self.staged_kv_cleanup.load(Ordering::Acquire);
        let work = plan
            .work_of(k)
            .ok_or_else(|| SelectionError::Internal(format!("plan has no stage {k}")))?;
        // Accounting by the stage's work, from the library's one definition.
        let mut config = body.router_config_override.take().unwrap_or_default();
        let mut expected_output_tokens = body
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        work.apply_to(&mut config, &mut expected_output_tokens);
        if let Some(expected) = expected_output_tokens {
            body.routing_mut().expected_output_tokens = Some(expected);
        }
        // Normal disaggregation keeps decode routing load-only; conditional
        // disaggregation lets the base overlap credit apply.
        if work == StageWork::DecodeOnly && !self.allow_decode_overlap_affinity {
            config.overlap_score_credit = Some(0.0);
        }
        body.router_config_override = Some(config);

        let constraints = plan
            .placement_constraints(k, &req.routing_constraints)
            .map_err(|error| SelectionError::Conflict(error.to_string()))?;
        if !constraints.is_empty() {
            body.routing_mut().routing_constraints = Some(constraints);
        }
        let stage = plan.stage(k).expect("stage index checked by work_of");
        let mut excluded = HashSet::new();
        for constraint in &stage.constraints {
            match constraint {
                Constraint::Pin(worker) | Constraint::Previewed(worker) => {
                    self.pin(&mut body, *worker);
                }
                Constraint::Exclude(worker_id) => {
                    excluded.insert(*worker_id);
                }
                Constraint::TransferCompatible(_) | Constraint::SameDomain { .. } => {}
            }
        }
        self.exclude(&mut body, &excluded);
        Ok(self.base.fork(body))
    }

    fn previewed_worker(plan: &Plan, k: usize) -> Option<WorkerWithDpRank> {
        plan.stage(k)?
            .constraints
            .iter()
            .find_map(|constraint| match constraint {
                Constraint::Previewed(worker) => Some(*worker),
                _ => None,
            })
    }
}

#[async_trait]
impl Router for HostSetRouter {
    /// An advisory pick: the host's KV route preview, with the busy signals
    /// the conditional policy reads.
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        let mut body = self.base.content().clone();
        if let Some(worker) = req.pinned_worker {
            self.pin(&mut body, worker);
        }
        if let Some(allowed) = req.allowed_worker_ids {
            body.routing_mut().allowed_worker_ids = Some(allowed);
        }
        if !req.routing_constraints.is_empty() {
            body.routing_mut().routing_constraints = Some(req.routing_constraints);
        }
        let rendered = self.base.fork(body);
        let preview = self
            .host
            .preview_kv_route(&rendered, self.phase)
            .await
            .map_err(|error| host_error(error, self.pinned_worker_id(rendered.content())))?;
        let signals = preview.signals;
        let prompt_tokens = req.prompt.token_ids.as_ref().map_or(0, Vec::len);
        let decode_busy = self
            .thresholds
            .decode
            .and_then(|threshold| signals.decode_load_exceeds(threshold));
        let worker_load = match (self.phase, self.thresholds.prefill) {
            (RequestPhase::Prefill, Some(threshold)) => Some(SelectionWorkerLoad {
                active_prefill_tokens: 0,
                prefill_token_capacity: 0,
                total_kv_blocks: signals.total_kv_blocks,
                prefill_busy: self
                    .host
                    .prefill_worker_busy(&rendered, threshold)
                    .await
                    .ok(),
            }),
            _ => None,
        };
        *self.pending_preview.lock().expect("preview lock") = Some((signals.worker, preview));
        let key = self.partition();
        let cached = signals.cached_tokens as u32;
        Ok(SelectResponse {
            selection_id: None,
            sequence_hashes: None,
            isl_tokens: Some(prompt_tokens),
            track_prefill_tokens: None,
            model_name: key.model_name.clone(),
            routing_group: key.routing_group.clone(),
            worker_id: signals.worker.worker_id,
            dp_rank: signals.worker.dp_rank,
            endpoint: String::new(),
            block_size: self.host.kv_router().block_size(),
            overlap: MooncakeOverlapSummary {
                longest_matched: cached,
                gpu: cached,
                ..MooncakeOverlapSummary::default()
            },
            effective_prefill_tokens: prompt_tokens.saturating_sub(signals.cached_tokens),
            potential_decode_blocks: signals.potential_decode_blocks,
            decode_busy,
            worker_load,
            kv_hint: None,
        })
    }

    fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        Plan::new(
            PlanId::from(req.selection_id.clone().unwrap_or_default()),
            self.partition(),
            vec![Stage::new(self.set)],
        )
        .map_err(|error| SelectionError::BadRequest(error.to_string()))
    }

    async fn schedule(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        loop {
            let next = plan
                .schedulable()
                .find(|&k| plan.stage(k).is_some_and(|stage| stage.set == self.set));
            let Some(k) = next else { break };
            let rendered = self.render(req, plan, k)?;
            // The stage's wait is the queue's hold budget, as in the core.
            let hold_budget = match plan.stage(k).expect("stage exists").wait {
                Budget::Full => None,
                Budget::Immediate => Some(std::time::Duration::ZERO),
                Budget::Bounded(budget) => Some(budget),
            };
            let pending = self.pending_preview.lock().expect("preview lock").take();
            // A previewed pin continues the preview's admission and budget. A
            // caller's pin is on the rendered request's routing hints, where
            // the host validates it as it does today.
            let pinned = self.pinned_worker_id(rendered.content());
            let route_plan = match (Self::previewed_worker(plan, k), pending) {
                (Some(previewed), Some((worker, preview))) if worker == previewed => self
                    .host
                    .plan_kv_route_from_preview(&rendered, preview, hold_budget)
                    .await
                    .map_err(|error| host_error(error, pinned))?,
                (previewed, _) => self
                    .host
                    .admit_kv_route(
                        &rendered,
                        self.phase,
                        previewed,
                        CleanupBudget::default(),
                        hold_budget,
                    )
                    .await
                    .map_err(|error| host_error(error, pinned))?,
            };
            let descriptor = route_plan.booking_descriptor().ok_or_else(|| {
                SelectionError::Internal("admitted route has no booking".to_string())
            })?;
            // The chosen worker's own requirement on its earlier peers is the
            // one check the forward constraints it was selected under cannot
            // make; a refused pair is aborted before the host keeps it.
            let facts = self.facts(route_plan.worker().worker_id);
            if let Err(error) = plan
                .check_placement(k, &facts)
                .and_then(|()| plan.book(k, Booking::Committed(descriptor), facts, None))
            {
                route_plan.abort().await;
                return Err(error.into());
            }
            let side = StageSide {
                signals: route_plan.signals,
                plan: route_plan,
            };
            self.sidecar.lock().expect("sidecar lock").insert(k, side);
        }
        Ok(())
    }
}

/// The plan-level request for a frontend request: the prompt the selector
/// hashes, the caller's constraints, and the class the scheduler queues under.
pub fn routing_request(
    request: &SingleIn<PreprocessedRequest>,
    partition: &RoutingPartitionId,
    policy_class: Option<String>,
) -> SelectAndReserveRequest {
    let body = request.content();
    let (token_ids, _) = body.block_mm_routing_info();
    let routing = body.routing.as_ref();
    SelectAndReserveRequest {
        model_name: partition.model_name.clone(),
        routing_group: partition.routing_group.clone(),
        selection_id: Some(request.id().to_string()),
        prompt: PromptRequest {
            token_ids: Some(token_ids.to_vec()),
            ..PromptRequest::default()
        },
        router_config_override: body.router_config_override.clone(),
        expected_output_tokens: routing.and_then(|hints| hints.expected_output_tokens),
        priority_jump: routing.and_then(|hints| hints.priority_jump),
        strict_priority: routing.and_then(|hints| hints.strict_priority),
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: routing.and_then(|hints| hints.allowed_worker_ids.clone()),
        routing_constraints: routing
            .and_then(|hints| hints.routing_constraints.clone())
            .unwrap_or_default(),
        policy_class,
        all_now: false,
        export_bookings: false,
    }
}

/// A host error as the router library reports it. The frontend types the
/// scheduler's answers as `BackendError`s, so the error type is what carries
/// them through the chain; the plan sees the scheduler variant again, and
/// `Fallback`, retry logic and budget semantics keep working. Anything else
/// is internal.
/// The host's canonical error as the Router contract's typed answer, so the
/// multi-stage router's fallback and queue logic read it as they read the
/// core's. [`frontend_error`] is the inverse: the two are the one conversion
/// pair between the host's classification and the library's.
pub(crate) fn host_error(error: Error, pinned: Option<WorkerId>) -> SelectionError {
    use dynamo_runtime::error::{ErrorType, match_error_chain};
    // A scheduler answer the host passed through untouched (a queue rejection
    // with its payload) keeps its type.
    let error = match error.downcast::<KvSchedulerError>() {
        Ok(scheduler) => return SelectionError::Scheduler(scheduler),
        Err(error) => error,
    };
    let scheduler = if match_error_chain(error.as_ref(), &[ErrorType::DeadlineExceeded], &[]) {
        Some(KvSchedulerError::DeadlineExceeded)
    } else if match_error_chain(error.as_ref(), &[ErrorType::Unavailable], &[]) {
        Some(KvSchedulerError::AllEligibleWorkersFiltered)
    } else if match_error_chain(error.as_ref(), &[ErrorType::WorkerOverloaded], &[]) {
        Some(match pinned {
            Some(worker_id) => KvSchedulerError::PinnedWorkerOverloaded { worker_id },
            None => KvSchedulerError::AllEligibleWorkersOverloaded,
        })
    } else if match_error_chain(error.as_ref(), &[ErrorType::ResourceExhausted], &[]) {
        Some(KvSchedulerError::AllEligibleWorkersOverloaded)
    } else {
        None
    };
    match scheduler {
        Some(scheduler) => SelectionError::Scheduler(scheduler),
        None => SelectionError::Internal(error.to_string()),
    }
}

/// A Router error at the frontend's exit: a scheduler answer goes back through
/// the host's one classification (type, queue-deadline reason, overload
/// cause), so the HTTP status and metrics are those of a direct admission.
pub(crate) fn frontend_error(error: SelectionError) -> Error {
    match error {
        SelectionError::Scheduler(scheduler) => crate::kv_router::map_scheduler_error(scheduler),
        other => other.into(),
    }
}
