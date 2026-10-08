// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The multistage [`Router`]: one router per worker set, composed under a
//! class table. `schedule` walks the plan's schedulable stages in order,
//! applies each stage's skip rule, and hands every stage to its set's router.
//! Every stage runs the same four steps in its own set's queue; the only
//! per-stage difference is the wait budget the class gave it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::WorkerType;
use crate::conditional_disagg::{ConditionalDisaggDecisionInput, ConditionalDisaggPolicy};
use crate::identity::RoutingPartitionId;
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::KvSchedulerError;
use crate::services::selection::{
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionError,
};

use super::class_table::{ClassTable, Fallback, StageList};
use super::plan::{Budget, Constraint, Plan, PlanId, PlanState, SkipRule, Stage};
use super::router_trait::Router;

/// Bounds the multistage router enforces around a plan: the equivalent of
/// the DEP's `CoordinationLimits`, kept on the router, not on a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterLimits {
    /// Total time one `schedule` call may spend. `None` leaves a stage's
    /// `Budget::Full` wait to its class's queue policy, as today; a deadline
    /// caps every wait in the call, `Full` included.
    pub schedule_deadline: Option<Duration>,
    /// A stage's own attempts, counting the first: failures the host
    /// retried, not re-placements caused by another stage.
    pub max_attempts: u32,
    /// How long a booked stage may wait for the host to dispatch it before
    /// the next `schedule` call refuses to continue the plan.
    pub max_hold: Duration,
}

impl Default for RouterLimits {
    fn default() -> Self {
        Self {
            schedule_deadline: None,
            max_attempts: 3,
            max_hold: Duration::from_secs(120),
        }
    }
}

pub struct MultiStageRouter {
    sets: HashMap<WorkerType, Arc<dyn Router>>,
    classes: ClassTable,
    limits: RouterLimits,
    conditional_disagg: Option<Arc<dyn ConditionalDisaggPolicy>>,
    has_decode_gate: bool,
}

#[derive(Default)]
pub struct MultiStageRouterBuilder {
    sets: HashMap<WorkerType, Arc<dyn Router>>,
    classes: Option<ClassTable>,
    limits: RouterLimits,
    conditional_disagg: Option<Arc<dyn ConditionalDisaggPolicy>>,
    has_decode_gate: bool,
}

impl MultiStageRouterBuilder {
    pub fn set(mut self, set: WorkerType, router: Arc<dyn Router>) -> Self {
        self.sets.insert(set, router);
        self
    }

    pub fn classes(mut self, classes: ClassTable) -> Self {
        self.classes = Some(classes);
        self
    }

    pub fn limits(mut self, limits: RouterLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The policy behind `SkipRule::ConditionalDisagg`; without one the rule
    /// never skips. `has_decode_gate` says the decode set evaluates a
    /// decode-busy threshold: a bypass then needs `decode_busy == Some(false)`,
    /// and an unknown signal denies it, as the frontend does today.
    pub fn conditional_disagg(
        mut self,
        policy: Arc<dyn ConditionalDisaggPolicy>,
        has_decode_gate: bool,
    ) -> Self {
        self.conditional_disagg = Some(policy);
        self.has_decode_gate = has_decode_gate;
        self
    }

    /// Rejects a class whose stage list names a set with no router, is not
    /// a valid plan, puts a decode-reading skip rule after its decode stage,
    /// or falls back to a decode set it does not have.
    pub fn build(self) -> Result<MultiStageRouter, SelectionError> {
        let classes = self.classes.unwrap_or_default();
        let invalid = |class: &str, reason: String| {
            SelectionError::BadRequest(format!("class {class:?} stage list: {reason}"))
        };
        for (class, list) in classes.lists() {
            let class = class.unwrap_or("default");
            if let Some(set) = list.sets().find(|set| !self.sets.contains_key(set)) {
                return Err(invalid(
                    class,
                    format!("no router was given for the {set} set"),
                ));
            }
            if list.fallback.is_some() && !self.sets.contains_key(&WorkerType::Decode) {
                return Err(invalid(
                    class,
                    "falls back to a decode set it does not have".to_string(),
                ));
            }
            for (k, stage) in list.stages.iter().enumerate() {
                let reads_decode = matches!(
                    stage.skip,
                    Some(SkipRule::ConditionalDisagg | SkipRule::DecodeHoldsPrefix)
                );
                if reads_decode
                    && !list.stages[k + 1..]
                        .iter()
                        .any(|later| later.set == WorkerType::Decode)
                {
                    return Err(invalid(
                        class,
                        format!(
                            "stage {k}'s skip rule previews decode but no decode stage follows it"
                        ),
                    ));
                }
            }
            Plan::new(
                PlanId::from("validate"),
                RoutingPartitionId::new("validate", "validate"),
                list.stages.clone(),
            )
            .map_err(|error| invalid(class, error.to_string()))?;
        }
        Ok(MultiStageRouter {
            sets: self.sets,
            classes,
            limits: self.limits,
            conditional_disagg: self.conditional_disagg,
            has_decode_gate: self.has_decode_gate,
        })
    }
}

/// What a skip rule decided for a stage.
enum Decision {
    Keep,
    Skip,
    /// Skip, and pin `stage` to the worker whose signals decided it.
    SkipAndPin {
        stage: usize,
        worker: WorkerWithDpRank,
    },
}

impl MultiStageRouter {
    pub fn builder() -> MultiStageRouterBuilder {
        MultiStageRouterBuilder::default()
    }

    fn stage_list(&self, req: &SelectAndReserveRequest) -> &StageList {
        self.classes.stages(req.policy_class.as_deref())
    }

    fn set_router(&self, set: WorkerType) -> Result<&Arc<dyn Router>, SelectionError> {
        self.sets
            .get(&set)
            .ok_or_else(|| SelectionError::Internal(format!("no router for the {set} set")))
    }

    fn check_limits(&self, plan: &Plan) -> Result<(), SelectionError> {
        for k in 0..plan.stage_count() {
            if plan.state_of(k).is_some_and(|state| state.is_pending())
                && plan
                    .failures(k)
                    .is_some_and(|failures| failures + 1 > self.limits.max_attempts)
            {
                return Err(SelectionError::Conflict(format!(
                    "stage {k} used its {} attempts",
                    self.limits.max_attempts
                )));
            }
            if plan
                .held_for(k)
                .is_some_and(|held| held > self.limits.max_hold)
            {
                tracing::debug!(
                    plan = %plan.id(),
                    stage = k,
                    max_hold_ms = self.limits.max_hold.as_millis() as u64,
                    "booked stage held too long; the plan cannot continue"
                );
                return Err(SelectionError::Scheduler(
                    KvSchedulerError::DeadlineExceeded,
                ));
            }
        }
        Ok(())
    }

    async fn schedule_inner(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        self.check_limits(plan)?;
        let mut kept = HashSet::new();
        loop {
            // Skip rules first, on every stage whose turn has come, whether or
            // not its constraints can be read yet: a set's router books every
            // stage of its set that becomes bookable, so each must be decided
            // before any is handed over.
            let candidates: Vec<usize> = plan.due().filter(|k| !kept.contains(k)).collect();
            for k in candidates {
                let Some(rule) = plan.stage(k).and_then(|stage| stage.skip) else {
                    continue;
                };
                match self.evaluate_skip(req, plan, k, rule).await? {
                    Decision::Keep => {
                        kept.insert(k);
                    }
                    Decision::Skip => plan.skip(k).map_err(internal)?,
                    Decision::SkipAndPin { stage, worker } => {
                        plan.skip(k).map_err(internal)?;
                        plan.constrain(stage, Constraint::Previewed(worker))
                            .map_err(internal)?;
                    }
                }
            }
            let Some(k) = plan.schedulable().next() else {
                break;
            };
            let set = plan
                .stage(k)
                .map(|stage| stage.set)
                .ok_or_else(|| internal(super::plan::PlanError::NoSuchStage { stage: k }))?;
            self.set_router(set)?.schedule(req, plan).await?;
            if plan.state_of(k).is_some_and(|state| state.is_pending()) {
                return Err(SelectionError::Internal(format!(
                    "the {set} set's router returned without booking stage {k}"
                )));
            }
        }
        Ok(())
    }

    /// The first stage could not be booked for capacity and the class
    /// allows it: book the decode set as one stage instead.
    async fn fall_back(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        let mut decode = Stage::new(WorkerType::Decode);
        if req.all_now {
            decode.wait = Budget::Immediate;
        }
        *plan = Plan::new(plan.id().clone(), plan.partition().clone(), vec![decode])
            .map_err(internal)?;
        self.set_router(WorkerType::Decode)?
            .schedule(req, plan)
            .await
    }

    async fn schedule_with_fallback(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        match self.schedule_inner(req, plan).await {
            Err(error)
                if self.stage_list(req).fallback == Some(Fallback::Aggregated)
                    && plan.state() == PlanState::Planned
                    && is_capacity_error(&error) =>
            {
                self.fall_back(req, plan).await
            }
            result => result,
        }
    }

    async fn evaluate_skip(
        &self,
        req: &SelectAndReserveRequest,
        plan: &Plan,
        k: usize,
        rule: SkipRule,
    ) -> Result<Decision, SelectionError> {
        let stage = plan
            .stage(k)
            .ok_or_else(|| internal(super::plan::PlanError::NoSuchStage { stage: k }))?;
        match rule {
            SkipRule::NoMultimodal => Ok(if req.prompt.mm_routing_info.is_none() {
                Decision::Skip
            } else {
                Decision::Keep
            }),
            SkipRule::WorkerBusy(fraction) => {
                let response = self.set_router(stage.set)?.select(advisory(req)).await?;
                let is_busy = response.worker_load.is_some_and(|load| {
                    load.active_prefill_tokens as f64
                        > fraction * load.prefill_token_capacity as f64
                });
                Ok(if is_busy {
                    Decision::Skip
                } else {
                    Decision::Keep
                })
            }
            SkipRule::DecodeHoldsPrefix | SkipRule::ConditionalDisagg => {
                // A caller-pinned stage is not up for discussion.
                let is_pinned = req.pinned_worker.is_some()
                    || stage
                        .constraints
                        .iter()
                        .any(|constraint| matches!(constraint, Constraint::Pin(_)));
                let Some(decode) = ((k + 1)..plan.stage_count())
                    .find(|&j| plan.stage(j).is_some_and(|s| s.set == WorkerType::Decode))
                else {
                    return Ok(Decision::Keep);
                };
                let conditional = match rule {
                    SkipRule::ConditionalDisagg => match &self.conditional_disagg {
                        Some(policy) if policy.is_enabled() => Some(policy),
                        _ => return Ok(Decision::Keep),
                    },
                    _ => None,
                };
                if is_pinned {
                    return Ok(Decision::Keep);
                }
                // Preview the decode worker under the decode stage's own
                // constraints: the same selection it would get, without a
                // booking. Its signals decide whether prefill runs.
                let decode_stage = plan.stage(decode).ok_or_else(|| {
                    internal(super::plan::PlanError::NoSuchStage { stage: decode })
                })?;
                // A placement rule that reads a stage still to be booked
                // (other than the one a skip would remove) cannot be
                // previewed: keep remote prefill rather than guess.
                if plan.unresolved_reads(decode).any(|j| j != k) {
                    return Ok(Decision::Keep);
                }
                // One advisory copy of the request; a second is made only
                // for a policy that also asks the prefill set.
                let mut decode_probe = advisory(req);
                decode_probe.routing_constraints = plan
                    .placement_constraints(decode, &req.routing_constraints)
                    .map_err(internal)?;
                let mut excluded = HashSet::new();
                for constraint in &decode_stage.constraints {
                    match constraint {
                        Constraint::Pin(worker) | Constraint::Previewed(worker) => {
                            decode_probe.pinned_worker = Some(*worker);
                        }
                        Constraint::Exclude(worker_id) => {
                            excluded.insert(*worker_id);
                        }
                        Constraint::TransferCompatible(_) | Constraint::SameDomain { .. } => {}
                    }
                }
                if let Some(allowed) = decode_probe.allowed_worker_ids.as_mut() {
                    allowed.retain(|worker_id| !excluded.contains(worker_id));
                }
                let preview = self
                    .set_router(WorkerType::Decode)?
                    .select(decode_probe)
                    .await?;
                if excluded.contains(&preview.worker_id) {
                    // The decode set would not book this worker: keep remote prefill.
                    return Ok(Decision::Keep);
                }
                let prompt_tokens = prompt_tokens(req, &preview);
                // The weighted cache credit the core already applied.
                let cached_tokens = prompt_tokens.saturating_sub(preview.effective_prefill_tokens);
                let should_bypass = match conditional {
                    None => {
                        let block = (preview.block_size as usize).max(1);
                        cached_tokens >= prompt_tokens - prompt_tokens % block
                    }
                    Some(policy) => {
                        // A failed prefill-load probe is a missing signal,
                        // not a failed request, as in the frontend today.
                        let prefill_busy = if policy.needs_prefill_worker_busy() {
                            self.set_router(stage.set)?
                                .select(advisory(req))
                                .await
                                .ok()
                                .and_then(|response| response.worker_load)
                                .and_then(|load| load.prefill_busy)
                        } else {
                            None
                        };
                        let input =
                            ConditionalDisaggDecisionInput::new(prompt_tokens, cached_tokens)
                                .with_prefill_chosen_worker_busy(prefill_busy)
                                .with_decode_chosen_worker_busy(preview.decode_busy);
                        let says_bypass = policy.should_bypass_remote_prefill(input).await;
                        let gate_allows = if self.has_decode_gate {
                            preview.decode_busy == Some(false)
                        } else {
                            true
                        };
                        says_bypass && gate_allows
                    }
                };
                Ok(if should_bypass {
                    Decision::SkipAndPin {
                        stage: decode,
                        worker: WorkerWithDpRank::new(preview.worker_id, preview.dp_rank),
                    }
                } else {
                    Decision::Keep
                })
            }
        }
    }
}

fn internal(error: super::plan::PlanError) -> SelectionError {
    SelectionError::Internal(error.to_string())
}

/// The prompt length the conditional policy reasons about.
fn prompt_tokens(req: &SelectAndReserveRequest, preview: &SelectResponse) -> usize {
    req.prompt
        .isl_tokens
        .or_else(|| req.prompt.token_ids.as_ref().map(Vec::len))
        .or(preview.isl_tokens)
        .unwrap_or(0)
}

/// The advisory form of a booking request: same prompt and constraints, no
/// booking.
fn advisory(req: &SelectAndReserveRequest) -> SelectRequest {
    SelectRequest {
        model_name: req.model_name.clone(),
        routing_group: req.routing_group.clone(),
        selection_id: None,
        prompt: req.prompt.clone(),
        router_config_override: req.router_config_override.clone(),
        expected_output_tokens: req.expected_output_tokens,
        priority_jump: req.priority_jump,
        strict_priority: req.strict_priority,
        session_id: req.session_id.clone(),
        session_context: req.session_context.clone(),
        affinity_target: req.affinity_target,
        pinned_worker: req.pinned_worker,
        allowed_worker_ids: req.allowed_worker_ids.clone(),
        routing_constraints: req.routing_constraints.clone(),
        advisory: true,
    }
}

#[async_trait]
impl Router for MultiStageRouter {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        // An advisory pick is for the set that serves the response: decode
        // when the default list has one, else its last stage.
        let stages = &self.classes.stages(None).stages;
        let set = stages
            .iter()
            .map(|stage| stage.set)
            .find(|set| *set == WorkerType::Decode)
            .or_else(|| stages.last().map(|stage| stage.set))
            .unwrap_or(WorkerType::Aggregated);
        self.set_router(set)?.select(req).await
    }

    fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        let list = self.stage_list(req);
        let stages = if req.all_now {
            list.all_now().stages
        } else {
            list.stages.clone()
        };
        let id = req
            .selection_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Plan::new(
            PlanId::from(id),
            RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone()),
            stages,
        )
        .map_err(|error| SelectionError::BadRequest(error.to_string()))
    }

    async fn schedule(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        match self.limits.schedule_deadline {
            None => self.schedule_with_fallback(req, plan).await,
            Some(deadline) => {
                tokio::time::timeout(deadline, self.schedule_with_fallback(req, plan))
                    .await
                    .unwrap_or(Err(SelectionError::Scheduler(
                        KvSchedulerError::DeadlineExceeded,
                    )))
            }
        }
    }
}

/// Only a capacity answer from the scheduler may trigger a fallback; an
/// error about the request, the plan or the scheduler's own state is
/// reported as is.
fn is_capacity_error(error: &SelectionError) -> bool {
    matches!(
        error,
        SelectionError::NotReady(_)
            | SelectionError::Scheduler(
                KvSchedulerError::QueueRejected(_)
                    | KvSchedulerError::DeadlineExceeded
                    | KvSchedulerError::NoEndpoints
                    | KvSchedulerError::AllEligibleWorkersOverloaded
                    | KvSchedulerError::AllEligibleWorkersFiltered
                    | KvSchedulerError::PinnedWorkerOverloaded { .. }
            )
    )
}
