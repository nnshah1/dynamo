// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill/decode routing with the frontend as the host of a Plan-returning
//! `Router`.
//!
//! The frontend schedules a plan on the routers of its worker sets (the
//! selection cores its `KvRouter`s already embed, composed by a
//! `MultiStageRouter` or a custom `Router` from the factory) and dispatches
//! from the plan: each booked stage's worker, booking, KV hint and scheduler
//! answer are read off the plan and handed to the host's existing dispatch,
//! tracking and cleanup. The library owns the stage list, the conditional
//! bypass, placement, budgets and accounting; this module owns session
//! affinity, the request's classifier lifecycle, the prefill handoff and the
//! response stream, as the legacy path does. Once decode is dispatched the
//! plan is informational: its bookings already live in the host's cleanups,
//! so dropping it frees nothing.
//!
//! `DYN_ROUTER_PLAN_HOST=0` falls back to the legacy hand-composed path for
//! one release.
// TODO(v1.7): remove the switch and the legacy KV prefill/decode path.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::conditional_disagg::ConditionalDisaggPolicy;
use dynamo_kv_router::protocols::{WorkerAffinityTarget, WorkerWithDpRank};
use dynamo_kv_router::router::{
    Booking, ClassTable, Constraint, Failure, MultiStageRouter, Plan, Router, StageList, StageState,
};
use dynamo_kv_router::scheduling::RequestLifecycle;
use dynamo_kv_router::services::selection::SelectAndReserveRequest;
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    pipeline::{
        AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tokio::sync::Mutex;

use super::{
    BYPASS_REMOTE_PREFILL_ANNOTATION, PrefillBinding, PrefillCompletion, PrefillOutcome,
    PrefillRouter, PreparedPrefill, build_decode_router_override, extract_bootstrap_info,
    handoff::PrefillTask, independent_prefill_context, into_decode_request,
    strip_terminal_disaggregated_params,
};
use crate::kv_router::RoutingHost;
use crate::kv_router::plan_host::{frontend_error, routing_request};
use crate::kv_router::routing_host::{
    CleanupBudget, StagedKv, await_with_cleanup_policy, cancelled_error,
    kv_selection::{WorkerSelection, pinned_worker_hint, resolve_pinned_worker_rank},
};
use crate::protocols::common::{
    extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::{PrefillResult, TraceLink},
    timing::{RequestPhase, RequestTracker},
};
use crate::session_affinity::{AffinityTarget, Hold, SessionAffinityMode};

/// What a [`PlanRouterFactory`] receives: one `Router` per worker set (the
/// set's selection core), and the conditional-disaggregation policy with
/// whether a decode-busy gate is configured. The factory returns the
/// `Router` the frontend drives for the request; the default builds a
/// `MultiStageRouter` over the class's stage list.
pub struct PlanRouterParts {
    pub prefill: Arc<dyn Router>,
    pub decode: Arc<dyn Router>,
    pub conditional: Option<(Arc<dyn ConditionalDisaggPolicy>, bool)>,
}

/// A compiled injection point for a custom `Router`: wrap or replace the
/// default without changing the host's dispatch and lifecycle code. Set once
/// at construction time with [`PrefillRouter::set_plan_router_factory`].
pub type PlanRouterFactory = Arc<dyn Fn(PlanRouterParts) -> Result<Arc<dyn Router>> + Send + Sync>;

/// The default: a `MultiStageRouter` over the plain or conditional
/// prefill/decode list.
pub fn default_plan_router(parts: PlanRouterParts) -> Result<Arc<dyn Router>> {
    let list = if parts.conditional.is_some() {
        StageList::conditional_prefill_decode()
    } else {
        StageList::prefill_decode()
    };
    let mut builder = MultiStageRouter::builder()
        .set(WorkerType::Prefill, parts.prefill)
        .set(WorkerType::Decode, parts.decode)
        .classes(ClassTable::new(list));
    if let Some((policy, has_decode_gate)) = parts.conditional {
        builder = builder.conditional_disagg(policy, has_decode_gate);
    }
    Ok(Arc::new(builder.build()?))
}

pub(crate) fn plan_host_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = PLAN_HOST_OVERRIDE.get() {
        return *forced;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("DYN_ROUTER_PLAN_HOST") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    })
}

#[cfg(test)]
static PLAN_HOST_OVERRIDE: OnceLock<bool> = OnceLock::new();

/// Force the switch for a test binary (first call wins).
#[cfg(test)]
pub(crate) fn force_plan_host(enabled: bool) {
    let _ = PLAN_HOST_OVERRIDE.set(enabled);
}

impl PrefillRouter {
    /// The decode host to plan with, when both hops are KV-routed and the
    /// planned path is enabled.
    pub(super) fn planned_decode_host(&self, binding: &PrefillBinding) -> Option<Arc<RoutingHost>> {
        if !plan_host_enabled() || !binding.prefill_router_mode.is_kv_routing() {
            return None;
        }
        binding.router.kv_router_if_enabled()?;
        let decode_host = self.decode_routing_host.get()?;
        decode_host.kv_router_if_enabled()?;
        Some(Arc::clone(decode_host))
    }

    /// Route one request's prefill and decode through a plan.
    pub(super) async fn generate_planned(
        &self,
        mut req: PreprocessedRequest,
        context: Context<()>,
        binding: Arc<PrefillBinding>,
        decode_host: Arc<RoutingHost>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let request_id = context.id().to_string();
        let original_max_tokens = req.stop_conditions.max_tokens;
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.clone().expect("tracker set above");
        let session_affinity = context
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(|message| anyhow!("invalid session affinity context: {message}"))?;
        let policy_class = context.metadata().get("policy-class").cloned();

        let decode_request: SingleIn<PreprocessedRequest> = context.map(|_| req);
        let mut prefill_body = decode_request.content().clone();
        prefill_body.stop_conditions.max_tokens = Some(1);
        let mut prefill_request = independent_prefill_context(prefill_body, &decode_request)?;
        if let Some(session_affinity) = session_affinity {
            prefill_request.insert(
                SESSION_AFFINITY_CONTEXT_KEY,
                session_affinity.as_ref().clone(),
            );
        }

        let conditional = self.conditional_disagg_policy.is_enabled();
        let decode_router = decode_host.kv_router();
        let routing = routing_request(
            &decode_request,
            &decode_router.partition(),
            policy_class,
            decode_router.is_eagle(),
        );

        let mut flow = PlannedFlow {
            router: self,
            binding: &binding,
            decode_host: &decode_host,
            tracker,
            request_id,
            original_max_tokens,
            prefill_request: Some(prefill_request),
            decode_request: Some(decode_request),
            // One cleanup budget for the request's whole route, as the
            // legacy host uses.
            budget: CleanupBudget::default(),
            affinity: None,
            prefill_affinity: None,
            migration_exhausted: Vec::new(),
            lifecycle: None,
            lifecycle_selected: false,
            staged_kv_cleanup: false,
            outcome: None,
            prefill_completion: None,
            dispatched: false,
        };
        match flow.run(routing, conditional).await {
            Ok(response) => Ok(response),
            // A failed conditional decision falls back to remote prefill, as
            // before, unless the request itself was cancelled or invalid.
            Err((routing, error))
                if conditional
                    && !flow.dispatched
                    && !match_error_chain(
                        error.as_ref(),
                        &[ErrorType::Cancelled, ErrorType::InvalidArgument],
                        &[],
                    ) =>
            {
                tracing::warn!(
                    request_id = %flow.request_id,
                    error = %error,
                    "Conditional disagg decision failed; falling back to remote prefill"
                );
                flow.run(routing, false).await.map_err(|(_, error)| error)
            }
            Err((_, error)) => Err(error),
        }
    }
}

/// One request's planned route: the plan's host.
struct PlannedFlow<'a> {
    router: &'a PrefillRouter,
    binding: &'a PrefillBinding,
    decode_host: &'a Arc<RoutingHost>,
    tracker: Arc<RequestTracker>,
    request_id: String,
    original_max_tokens: Option<u32>,
    prefill_request: Option<SingleIn<PreprocessedRequest>>,
    decode_request: Option<SingleIn<PreprocessedRequest>>,
    budget: CleanupBudget,
    /// The decode phase's session-affinity hold, bound to the decode stream.
    affinity: Option<Hold>,
    /// The prefill phase's hold, bound to the prefill stream at dispatch.
    prefill_affinity: Option<Hold>,
    /// The request's classifier lifecycle, owned by the decode leg.
    lifecycle: Option<Box<RequestLifecycle>>,
    lifecycle_selected: bool,
    /// A remote prefill staged KV: decode must go on through a disconnect.
    staged_kv_cleanup: bool,
    /// Sets a migration retry has no eligible worker left in. A stage of
    /// such a set that still needs admission fails with the migration's
    /// exhaustion error; a skipped or absent stage does not.
    migration_exhausted: Vec<WorkerType>,
    outcome: Option<PrefillOutcome>,
    prefill_completion: Option<PrefillTask>,
    /// Something reached a worker; a failure now is not a routing decision
    /// to fall back from.
    dispatched: bool,
}

enum Step {
    Continue,
    Done(ManyOut<Annotated<LLMEngineOutput>>),
}

/// The routing request and the plan it produced, shared with the affinity
/// wrapper's retry closure during the first `schedule`.
type Shared = Mutex<(SelectAndReserveRequest, Option<Plan>)>;

impl PlannedFlow<'_> {
    /// The `Router` for this request: the configured factory's, or the
    /// default `MultiStageRouter`, over the worker sets' selection cores.
    fn plan_router(&self, conditional: bool) -> Result<Arc<dyn Router>> {
        let parts = PlanRouterParts {
            prefill: self.binding.router.kv_router().plan_router(),
            decode: self.decode_host.kv_router().plan_router(),
            conditional: conditional.then(|| {
                (
                    Arc::clone(&self.router.conditional_disagg_policy),
                    self.router
                        .conditional_disagg_decode_busy_threshold
                        .is_some(),
                )
            }),
        };
        match self.router.plan_router_factory.get() {
            Some(factory) => factory(parts),
            None => default_plan_router(parts),
        }
    }

    /// Plan, schedule and dispatch. The routing request comes back with an
    /// error so a conditional fallback can rerun with it.
    async fn run(
        &mut self,
        routing: SelectAndReserveRequest,
        conditional: bool,
    ) -> std::result::Result<
        ManyOut<Annotated<LLMEngineOutput>>,
        (SelectAndReserveRequest, anyhow::Error),
    > {
        let shared: Shared = Mutex::new((routing, None));
        let result = self.run_inner(&shared, conditional).await;
        let (routing, _) = shared.into_inner();
        result.map_err(|error| (routing, error))
    }

    async fn run_inner(
        &mut self,
        shared: &Shared,
        conditional: bool,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let multistage = self.plan_router(conditional)?;
        let engine_ctx = self
            .decode_request
            .as_ref()
            .ok_or_else(|| anyhow!("decode request already consumed"))?
            .context();
        // A caller-named worker is checked once, before planning, as the
        // host's own path checks it; a withdrawn request is not planned.
        {
            let body = self
                .decode_request
                .as_ref()
                .ok_or_else(|| anyhow!("decode request already consumed"))?
                .content();
            self.binding
                .router
                .validate_explicit_worker(body, RequestPhase::Prefill)
                .map_err(|error| self.routing_error(error))?;
            self.decode_host
                .validate_explicit_worker(body, RequestPhase::Decode)
                .map_err(|error| self.routing_error(error))?;
        }
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            return Err(self.routing_error(cancelled_error(engine_ctx.id())));
        }
        {
            let mut guard = shared.lock().await;
            let (routing, slot) = &mut *guard;
            let mut plan = multistage
                .plan(routing)
                .map_err(|error| self.routing_error(frontend_error(error)))?;
            self.pin_caller_targets(&mut plan)?;
            self.exclude_migration_failures(routing, &mut plan)?;
            *slot = Some(plan);
        }
        self.schedule_first(&multistage, shared).await?;
        let mut guard = shared.lock().await;
        let (routing, plan) = &mut *guard;
        let plan = plan.as_mut().expect("plan set above");
        // The loop: book what is due, dispatch what is ready, record what
        // happened. A booking is the plan's until its stage is dispatched;
        // `run_prefill` / `run_decode` take it into the host's request guard,
        // which owns its cleanup from then on. Dropping the plan afterwards
        // frees nothing.
        loop {
            if plan.ready().next().is_none() {
                // A deferred stage waits in its queue under the host's
                // cancellation policy: a stopped client withdraws it, unless
                // a remote prefill staged KV that only decode can release.
                let staged_kv = if self.staged_kv_cleanup {
                    StagedKv::Present
                } else {
                    StagedKv::Absent
                };
                let body = self
                    .decode_request
                    .as_ref()
                    .ok_or_else(|| anyhow!("decode request already consumed"))?
                    .content();
                await_with_cleanup_policy(
                    engine_ctx.as_ref(),
                    RequestPhase::Decode,
                    staged_kv,
                    "plan schedule",
                    &self.budget,
                    multistage.schedule(routing, plan),
                )
                .await
                .map_err(|error| self.routing_error(error))?
                .map_err(|error| {
                    let error = migration_exhaustion(plan, &self.migration_exhausted, body)
                        .unwrap_or_else(|| frontend_error(error));
                    self.routing_error(error)
                })?;
            }
            let Some(k) = plan.ready().next() else {
                return Err(anyhow!(
                    "plan has nothing ready to dispatch: {:?}",
                    (0..plan.stage_count())
                        .map(|k| plan.state_of(k).cloned())
                        .collect::<Vec<_>>()
                ));
            };
            let set = plan.stage(k).expect("ready stage exists").set;
            let step = match set {
                WorkerType::Prefill => self.run_prefill(plan, k, &engine_ctx).await?,
                WorkerType::Decode => self.run_decode(plan, k, conditional, &engine_ctx).await?,
                other => return Err(anyhow!("plan dispatched unexpected set {other:?}")),
            };
            if let Step::Done(response) = step {
                return Ok(response);
            }
        }
    }

    /// The first `schedule`, under the host's session-affinity and
    /// classifier-lifecycle wrappers, as the legacy selection runs: the
    /// prefill host resolves the session's prefill target, the decode host
    /// its decode target, and each target steers its own set's stages (a
    /// pin in hard mode, a preference in soft mode). The decode hold and
    /// lifecycle stay with the flow for the decode dispatch.
    async fn schedule_first(
        &mut self,
        multistage: &Arc<dyn Router>,
        shared: &Shared,
    ) -> Result<()> {
        let prefill_request = self
            .prefill_request
            .as_ref()
            .ok_or_else(|| anyhow!("prefill request already consumed"))?;
        let decode_request = self
            .decode_request
            .as_ref()
            .ok_or_else(|| anyhow!("decode request already consumed"))?;
        let prefill_host = &self.binding.router;
        let decode_host = self.decode_host;
        let budget = &self.budget;
        let exhausted = &self.migration_exhausted;
        let engine_ctx = decode_request.context();
        let engine_ctx: &dyn AsyncEngineContext = engine_ctx.as_ref();
        let select = || {
            prefill_host.select_with_session_affinity(
                prefill_request,
                RequestPhase::Prefill,
                false,
                budget,
                |prefill_target| {
                    decode_host.select_with_session_affinity(
                        decode_request,
                        RequestPhase::Decode,
                        false,
                        budget,
                        move |decode_target| async move {
                            let mut guard = shared.lock().await;
                            let (routing, plan) = &mut *guard;
                            let plan = plan.as_mut().expect("plan set before scheduling");
                            steer(
                                routing,
                                plan,
                                WorkerType::Prefill,
                                prefill_target,
                                prefill_host,
                            )?;
                            steer(
                                routing,
                                plan,
                                WorkerType::Decode,
                                decode_target,
                                decode_host,
                            )?;
                            // Nothing is dispatched yet: a stopped client
                            // withdraws the queued admission, as the host's
                            // own admission does.
                            await_with_cleanup_policy(
                                engine_ctx,
                                RequestPhase::Decode,
                                StagedKv::Absent,
                                "plan schedule",
                                budget,
                                multistage.schedule(routing, plan),
                            )
                            .await?
                            .map_err(|error| {
                                migration_exhaustion(plan, exhausted, decode_request.content())
                                    .unwrap_or_else(|| frontend_error(error))
                            })?;
                            Ok(decode_worker(plan))
                        },
                    )
                },
            )
        };
        let (((decode_worker, decode_hold), prefill_hold), lifecycle) = decode_host
            .with_request_lifecycle(
                decode_request,
                RequestPhase::Decode,
                select,
                |((w, _), _)| *w,
            )
            .await
            .map_err(|error| self.routing_error(error))?;
        self.lifecycle_selected = decode_worker.is_some();
        self.lifecycle = lifecycle;
        self.affinity = decode_hold;
        self.prefill_affinity = prefill_hold;
        Ok(())
    }

    /// A migration retry may not reuse a worker a previous attempt failed on.
    /// Each host whose set the plan uses applies its one eligibility rule
    /// (allowed set, constraints, recorded failures); the plan records the
    /// exclusions on that set's pending stages so the core selects around
    /// them. A set with nothing eligible left is remembered: only a stage
    /// that still needs it after the skip decisions fails for it.
    fn exclude_migration_failures(
        &mut self,
        routing: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<()> {
        let body = self
            .decode_request
            .as_ref()
            .ok_or_else(|| anyhow!("decode request already consumed"))?
            .content();
        let Some(state) = body.migration_state.as_ref() else {
            return Ok(());
        };
        let phases = [
            (
                WorkerType::Prefill,
                RequestPhase::Prefill,
                &self.binding.router,
            ),
            (WorkerType::Decode, RequestPhase::Decode, self.decode_host),
        ];
        let mut exhausted = Vec::new();
        for (set, phase, host) in phases {
            if !has_stage(plan, set) {
                continue;
            }
            let Some(eligible) = host.migration_eligible_workers(
                body,
                phase,
                &routing.routing_constraints,
                routing.allowed_worker_ids.as_ref(),
            ) else {
                continue;
            };
            if eligible.is_empty() {
                exhausted.push(set);
            }
            for worker_id in state.excluded_worker_ids() {
                exclude_set(plan, set, worker_id)?;
            }
        }
        self.migration_exhausted = exhausted;
        Ok(())
    }

    /// A worker the caller named for a phase pins that phase's stage, so the
    /// conditional decision keeps remote prefill for an explicit prefill
    /// worker (as the legacy path does) and the host still validates the
    /// target. The hint precedence and the rank resolution are the host's
    /// own: an omitted rank is the worker's unique rank or a rejection, never
    /// an invented zero. Each pin goes to its own set only.
    fn pin_caller_targets(&self, plan: &mut Plan) -> Result<()> {
        let body = self
            .decode_request
            .as_ref()
            .ok_or_else(|| anyhow!("decode request already consumed"))?
            .content();
        let routing = body.routing.as_ref();
        let phases = [
            (
                WorkerType::Prefill,
                RequestPhase::Prefill,
                &self.binding.router,
            ),
            (WorkerType::Decode, RequestPhase::Decode, self.decode_host),
        ];
        for (set, phase, host) in phases {
            let Some((worker_id, dp_rank)) = pinned_worker_hint(phase, routing) else {
                continue;
            };
            let unique_rank = host
                .kv_router_if_enabled()
                .and_then(|router| router.unique_dp_rank_for_worker(worker_id));
            let worker = resolve_pinned_worker_rank(worker_id, dp_rank, unique_rank)?;
            pin_set(plan, set, worker)?;
        }
        Ok(())
    }

    async fn run_prefill(
        &mut self,
        plan: &mut Plan,
        k: usize,
        engine_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Result<Step> {
        let attempt = plan.mark_dispatched(k)?;
        let selection = stage_selection(plan, k, None)?;
        let mut prefill_request = self
            .prefill_request
            .take()
            .ok_or_else(|| anyhow!("prefill stage executed twice"))?;
        let phase_barrier = self.tracker.set_phase(RequestPhase::Prefill).await;
        let worker = selection.worker;
        let target = AffinityTarget::new(worker.worker_id, Some(worker.dp_rank));
        let prepared = self.router.prepare_prefill_dispatch(
            &mut prefill_request,
            target,
            &self.binding.endpoint_id,
        )?;
        let prefill_affinity = self.prefill_affinity.take();
        let prefill_stream = match self
            .binding
            .router
            .dispatch_worker_selection(
                prefill_request,
                selection,
                None,
                prefill_affinity,
                &self.budget,
            )
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                plan.mark_failed(k, attempt, failure(&error))?;
                return Err(self.prefill_error(error));
            }
        };
        self.dispatched = true;
        // Decode must now be dispatched even if the client disconnects: only
        // its worker's KV-transfer path frees what prefill staged.
        self.staged_kv_cleanup = true;

        let outcome = if let Some(bootstrap_info) = prepared.bootstrap_info.clone() {
            self.prefill_completion = Some(self.router.spawn_prefill_task(
                prefill_stream,
                Some(Arc::clone(&self.tracker)),
                phase_barrier,
            ));
            PrefillOutcome::Bootstrap {
                bootstrap_info,
                worker_id: prepared.worker_id,
                prefill_dp_rank: prepared.prefill_dp_rank,
            }
        } else {
            drop(phase_barrier);
            let completion = match PrefillRouter::consume_prefill_stream(
                prefill_stream,
                Some(Arc::clone(&self.tracker)),
                self.router.task_guard.clone(),
            )
            .await
            {
                Ok(completion) => completion,
                Err(error) => {
                    let error = anyhow::Error::from(error);
                    plan.mark_failed(k, attempt, failure(&error))?;
                    return Err(self.prefill_error(error));
                }
            };
            match completion {
                PrefillCompletion::Handoff {
                    result,
                    worker_link,
                    completion,
                } => {
                    self.prefill_completion = completion;
                    handoff_outcome(result, worker_link, &prepared)
                }
                PrefillCompletion::Terminal { output } => {
                    // The context step finished the request: nothing to hand
                    // off and no decode to select.
                    plan.mark_completed(k, attempt, worker)?;
                    let output = strip_terminal_disaggregated_params(*output);
                    return Ok(Step::Done(ResponseStream::new(
                        Box::pin(stream::once(async move { output })),
                        Arc::clone(engine_ctx),
                    )));
                }
            }
        };
        self.outcome = Some(outcome);
        // Bootstrap info or the first output unlocks decode; the prefill task
        // reports completion or a late failure through `forward_decode`.
        plan.mark_handed_off(k, attempt)?;
        Ok(Step::Continue)
    }

    async fn run_decode(
        &mut self,
        plan: &mut Plan,
        k: usize,
        conditional: bool,
        engine_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Result<Step> {
        plan.mark_dispatched(k)?;
        let mut lifecycle = self.lifecycle.take();
        if let (Some(lifecycle), Some(worker)) = (lifecycle.as_mut(), plan.worker(k))
            && !self.lifecycle_selected
        {
            lifecycle.selected(worker);
        }
        let selection = stage_selection(plan, k, lifecycle)?;
        let decode_request = self
            .decode_request
            .take()
            .ok_or_else(|| anyhow!("decode stage executed twice"))?;
        // Decode prefills locally unless a prefill stage ran ahead of it: a
        // skipped prefill (conditional bypass) or a plan with no prefill
        // stage at all (a custom router's decode-only shape).
        let local_prefill = !(0..k).any(|j| {
            plan.stage(j)
                .is_some_and(|stage| stage.set == WorkerType::Prefill)
                && plan.state_of(j) != Some(&StageState::Skipped)
        });
        // In the bootstrap path this waits until the spawned prefill task
        // releases its phase barrier, keeping worker attribution correct.
        let _decode_permit = self.tracker.set_phase(RequestPhase::Decode).await;
        let staged_kv_cleanup = self.staged_kv_cleanup;
        let decode_request = if local_prefill {
            tracing::info!(
                request_id = %self.request_id,
                worker_id = selection.worker.worker_id,
                dp_rank = selection.worker.dp_rank,
                cached_tokens = selection.cached_tokens,
                "Conditional disagg routing to decode worker"
            );
            decode_request.map(|mut request| {
                request
                    .annotations
                    .push(BYPASS_REMOTE_PREFILL_ANNOTATION.to_string());
                request
            })
        } else {
            let outcome = self
                .outcome
                .take()
                .ok_or_else(|| anyhow!("decode selected before the prefill handoff"))?;
            // NVBugs 5969206: decode routing proceeds even after the client
            // context stopped, so the KV transfer has a receiver.
            if engine_ctx.is_stopped() || engine_ctx.is_killed() {
                tracing::debug!(
                    "Context {} killed/stopped after prefill, allowing decode routing for KV transfer",
                    engine_ctx.id()
                );
            }
            let original_max_tokens = self.original_max_tokens;
            decode_request.map(|request| {
                let mut decode = into_decode_request(request, outcome);
                decode.stop_conditions.max_tokens = original_max_tokens;
                let existing = decode.router_config_override.take();
                decode.router_config_override =
                    Some(build_decode_router_override(existing, conditional));
                decode
            })
        };
        // A remote prefill staged KV for this leg: decode's dispatch and
        // cleanup go on through a client disconnect.
        let decode_request = decode_request.map(|mut request| {
            request.staged_kv_cleanup = staged_kv_cleanup;
            request
        });
        self.dispatched = true;
        let affinity = self.affinity.take();
        let dispatch = self.decode_host.dispatch_worker_selection(
            decode_request,
            selection,
            None,
            affinity,
            &self.budget,
        );
        let response = match self.prefill_completion.take() {
            Some(completion) => {
                completion
                    .forward_decode(dispatch, Arc::clone(engine_ctx))
                    .await?
            }
            None => dispatch.await?,
        };
        if local_prefill {
            let ctx = response.context();
            let annotation = Annotated::<LLMEngineOutput>::from_annotation(
                BYPASS_REMOTE_PREFILL_ANNOTATION,
                &true,
            )?;
            let merged = stream::once(async move { annotation }).chain(response);
            return Ok(Step::Done(ResponseStream::new(Box::pin(merged), ctx)));
        }
        Ok(Step::Done(response))
    }

    fn routing_error(&self, error: anyhow::Error) -> anyhow::Error {
        if match_error_chain(
            error.as_ref(),
            &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
            &[],
        ) {
            tracing::warn!(
                request_id = %self.request_id,
                error = %error,
                "request rejected during stage selection (at capacity)"
            );
        } else {
            tracing::error!(
                request_id = %self.request_id,
                error = %error,
                "Stage selection failed, failing request"
            );
        }
        error
    }

    fn prefill_error(&self, error: anyhow::Error) -> anyhow::Error {
        if match_error_chain(
            error.as_ref(),
            &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
            &[],
        ) {
            tracing::warn!(
                error = %error,
                "request rejected by prefill worker (at capacity)"
            );
        } else {
            tracing::error!(error = %error, "Remote prefill failed, failing request");
        }
        error
    }
}

fn failure(error: &anyhow::Error) -> Failure {
    Failure {
        is_retryable: false,
        reason: error.to_string(),
    }
}

/// Steer a set's stages toward the session's affinity target the way the
/// host's own selection does: hard affinity pins the stage, soft affinity
/// prefers the worker; a stage the caller already pinned keeps its pin.
fn steer(
    routing: &mut SelectAndReserveRequest,
    plan: &mut Plan,
    set: WorkerType,
    target: Option<AffinityTarget>,
    host: &RoutingHost,
) -> Result<()> {
    let Some(target) = target else {
        return Ok(());
    };
    match host.session_affinity_mode() {
        SessionAffinityMode::Hard => {
            let unique_rank = host
                .kv_router_if_enabled()
                .and_then(|router| router.unique_dp_rank_for_worker(target.worker_id));
            let worker = resolve_pinned_worker_rank(target.worker_id, target.dp_rank, unique_rank)?;
            pin_set(plan, set, worker)
        }
        SessionAffinityMode::Soft => {
            if set == WorkerType::Decode {
                routing.affinity_target =
                    Some(WorkerAffinityTarget::new(target.worker_id, target.dp_rank));
            }
            Ok(())
        }
    }
}

/// Pin every pending stage of `set` to `worker`, unless it is pinned already.
fn pin_set(plan: &mut Plan, set: WorkerType, worker: WorkerWithDpRank) -> Result<()> {
    for k in 0..plan.stage_count() {
        let Some(stage) = plan.stage(k) else { continue };
        if stage.set != set
            || !plan.state_of(k).is_some_and(StageState::is_pending)
            || stage
                .constraints
                .iter()
                .any(|constraint| matches!(constraint, Constraint::Pin(_)))
        {
            continue;
        }
        plan.constrain(k, Constraint::Pin(worker))?;
    }
    Ok(())
}

fn has_stage(plan: &Plan, set: WorkerType) -> bool {
    (0..plan.stage_count()).any(|k| plan.stage(k).is_some_and(|stage| stage.set == set))
}

/// A schedule failure while a pending (not skipped) stage belongs to a set
/// the migration retry has exhausted is the migration's own answer: the
/// request cannot be placed anywhere it has not already failed.
fn migration_exhaustion(
    plan: &Plan,
    exhausted: &[WorkerType],
    body: &PreprocessedRequest,
) -> Option<anyhow::Error> {
    let needs_exhausted_set = (0..plan.stage_count()).any(|k| {
        plan.stage(k)
            .is_some_and(|stage| exhausted.contains(&stage.set))
            && plan.state_of(k).is_some_and(StageState::is_pending)
    });
    if !needs_exhausted_set {
        return None;
    }
    body.migration_state
        .as_ref()
        .and_then(|state| state.exhausted_error())
        .map(anyhow::Error::from)
}

/// Exclude `worker_id` from every pending stage of `set`.
fn exclude_set(plan: &mut Plan, set: WorkerType, worker_id: u64) -> Result<()> {
    for k in 0..plan.stage_count() {
        let Some(stage) = plan.stage(k) else { continue };
        if stage.set != set || !plan.state_of(k).is_some_and(StageState::is_pending) {
            continue;
        }
        plan.constrain(k, Constraint::Exclude(worker_id))?;
    }
    Ok(())
}

fn decode_worker(plan: &Plan) -> Option<WorkerWithDpRank> {
    (0..plan.stage_count())
        .find(|&k| {
            plan.stage(k)
                .is_some_and(|stage| stage.set == WorkerType::Decode)
        })
        .and_then(|k| plan.worker(k))
}

/// The host's view of a booked stage, read off the plan: the worker, the
/// booking handle (now the host's to release through its cleanup), the
/// scheduler's answer for tracking, and the hint to forward.
fn stage_selection(
    plan: &mut Plan,
    k: usize,
    request_lifecycle: Option<Box<RequestLifecycle>>,
) -> Result<WorkerSelection> {
    let worker = plan
        .worker(k)
        .ok_or_else(|| anyhow!("stage {k} has no booked worker"))?;
    let booking = match plan.take_booking(k)? {
        Booking::Owned(handle) => Some(handle),
        // Someone else owns the release (an exported booking); nothing for
        // the cleanup to hold.
        _ => None,
    };
    let scheduling = plan.scheduling(k);
    Ok(WorkerSelection {
        worker,
        booking,
        overlap_amount: scheduling.map_or(0, |s| s.effective_overlap_blocks.round() as u32),
        effective_overlap_blocks: scheduling.map_or(0.0, |s| s.effective_overlap_blocks),
        cached_tokens: scheduling.map_or(0, |s| s.cached_tokens),
        selected_raw_cached_tokens: scheduling.and_then(|s| s.selected_raw_cached_tokens),
        max_raw_cached_tokens: scheduling.and_then(|s| s.max_raw_cached_tokens),
        potential_decode_blocks: scheduling.map_or(0, |s| s.potential_decode_blocks as u64),
        selected_worker_load: None,
        routing_hashes: None,
        kv_hint: plan.kv_hint(k).cloned(),
        request_lifecycle,
    })
}

fn handoff_outcome(
    result: PrefillResult,
    worker_link: Option<TraceLink>,
    prepared: &PreparedPrefill,
) -> PrefillOutcome {
    if let Some(bootstrap_info) = extract_bootstrap_info(&result.disaggregated_params) {
        PrefillOutcome::Bootstrap {
            bootstrap_info,
            worker_id: prepared.worker_id,
            prefill_dp_rank: prepared.prefill_dp_rank,
        }
    } else {
        PrefillOutcome::Completed {
            result,
            worker_id: prepared.worker_id,
            worker_link,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anyhow::Error;
    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{
            AddressedRequest, AsyncEngine, AsyncEngineContextProvider, Context, ManyIn, ManyOut,
            Operator, PushRouter, ResponseStream, RouterMode, SingleIn, StreamingDispatch,
            async_trait,
        },
        protocols::{annotated::Annotated, maybe_error::MaybeError},
    };
    use futures::{StreamExt, stream};
    use tokio::sync::watch;

    use super::super::{PrefillBinding, PrefillLifecycleState, PrefillRouter};
    use super::force_plan_host;
    use crate::{
        discovery::{ModelManager, WorkerSetTargetId},
        kv_router::{KvRouter, RoutingHost, SelectionPolicySource},
        local_model::runtime_config::ModelRuntimeConfig,
        protocols::common::{
            StopConditions,
            llm_backend::{FinishReason, LLMEngineOutput, PreprocessedRequest},
        },
    };
    use dynamo_kv_router::WorkerType;
    use dynamo_kv_router::router::{Plan, Router};

    /// A worker that answers each request with the next scripted frame list
    /// and records what it was asked.
    #[derive(Default)]
    struct ScriptedWorker {
        frames: Mutex<VecDeque<Vec<Annotated<LLMEngineOutput>>>>,
        seen: Mutex<Vec<(u64, PreprocessedRequest)>>,
        /// Runs once, inside the next dispatch: a client that cancels while
        /// the worker is being reached.
        on_dispatch: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    }

    impl ScriptedWorker {
        fn script(&self, frames: Vec<Annotated<LLMEngineOutput>>) {
            self.frames.lock().unwrap().push_back(frames);
        }
        fn seen(&self) -> Vec<(u64, PreprocessedRequest)> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for ScriptedWorker {
        async fn generate(
            &self,
            request: SingleIn<AddressedRequest<PreprocessedRequest>>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            tokio::task::yield_now().await;
            let (addressed, context) = request.transfer(());
            let (body, _, instance) = addressed.into_parts();
            self.seen
                .lock()
                .unwrap()
                .push((instance.expect("selected worker").id(), body));
            let frames = self
                .frames
                .lock()
                .unwrap()
                .pop_front()
                .expect("a scripted answer for every dispatch");
            if let Some(on_dispatch) = self.on_dispatch.lock().unwrap().take() {
                on_dispatch();
                assert!(
                    !context.context().is_stopped(),
                    "remote prefill keeps its independent context"
                );
            }
            Ok(ResponseStream::new(
                Box::pin(stream::iter(frames)),
                context.context(),
            ))
        }

        async fn generate_bidirectional(
            &self,
            _instance: dynamo_runtime::component::Instance,
            _address: String,
            _input: ManyIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            unreachable!("the routing host dispatches unary requests")
        }
    }

    /// The operator's `next`: the planned path dispatches decode through the
    /// decode host, never through the pipeline.
    struct NeverNext;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for NeverNext
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            panic!("the planned path must dispatch decode through the decode RoutingHost")
        }
    }

    struct KvSet {
        host: Arc<RoutingHost>,
        chooser: Arc<KvRouter>,
        worker: Arc<ScriptedWorker>,
        worker_id: u64,
        /// The set's admitted membership, as discovery publishes it.
        workers_tx: watch::Sender<HashMap<u64, ModelRuntimeConfig>>,
    }

    async fn kv_set(
        distributed: &DistributedRuntime,
        namespace: &str,
        role: &'static str,
        conditional: bool,
        worker_config: ModelRuntimeConfig,
    ) -> KvSet {
        // The deployment's flag reaches each set's router, as the model
        // manager passes it: a decode set that may run prefill locally keeps
        // cache-aware selection, a plain one scores on load alone.
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            conditional_disagg_enabled: conditional,
            ..Default::default()
        };
        kv_set_with_config(distributed, namespace, role, config, worker_config).await
    }

    async fn kv_set_with_config(
        distributed: &DistributedRuntime,
        namespace: &str,
        role: &'static str,
        config: KvRouterConfig,
        worker_config: ModelRuntimeConfig,
    ) -> KvSet {
        kv_set_with_policy(
            distributed,
            namespace,
            role,
            config,
            worker_config,
            SelectionPolicySource::Registry,
        )
        .await
    }

    /// A set whose core runs `policy` for worker selection.
    async fn kv_set_with_policy(
        distributed: &DistributedRuntime,
        namespace: &str,
        role: &'static str,
        config: KvRouterConfig,
        worker_config: ModelRuntimeConfig,
        policy: SelectionPolicySource,
    ) -> KvSet {
        let endpoint = distributed
            .namespace(namespace.to_string())
            .unwrap()
            .component(role.to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let (workers_tx, workers) = watch::channel(HashMap::from([(worker_id, worker_config)]));
        // The role types the core's `Router`: it books only the stages of
        // its own set, as production's discovery-managed hosts do.
        let worker_role = match role {
            "prefill" => WorkerType::Prefill,
            "decode" => WorkerType::Decode,
            other => panic!("unknown role {other}"),
        };
        let chooser = Arc::new(
            KvRouter::new_with_worker_role(
                endpoint,
                client.clone(),
                workers,
                None,
                16,
                policy,
                Some(config),
                None,
                Some(worker_role),
                role,
                None,
                false,
                None,
                None,
            )
            .await
            .unwrap(),
        );
        let worker = Arc::new(ScriptedWorker::default());
        let inner = PushRouter::from_client_with_dispatch(
            client,
            RouterMode::KV,
            Arc::clone(&worker) as Arc<dyn StreamingDispatch<_, _>>,
        )
        .await
        .unwrap();
        let host = Arc::new(RoutingHost::new(inner, Arc::clone(&chooser), None).unwrap());
        KvSet {
            host,
            chooser,
            worker,
            worker_id,
            workers_tx,
        }
    }

    struct Fixture {
        router: Arc<PrefillRouter>,
        prefill: KvSet,
        decode: KvSet,
        runtime: Runtime,
    }

    async fn fixture(namespace: &str) -> Fixture {
        fixture_with(namespace, false).await
    }

    /// `conditional`: the ISL-bounding policy is on, so an uncached prompt
    /// still goes to remote prefill and a cached one bypasses it.
    async fn fixture_with(namespace: &str, conditional: bool) -> Fixture {
        fixture_configs(
            namespace,
            conditional,
            ModelRuntimeConfig::default(),
            ModelRuntimeConfig::default(),
        )
        .await
    }

    async fn fixture_configs(
        namespace: &str,
        conditional: bool,
        prefill_config: ModelRuntimeConfig,
        decode_config: ModelRuntimeConfig,
    ) -> Fixture {
        force_plan_host(true);
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let prefill = kv_set(
            &distributed,
            namespace,
            "prefill",
            conditional,
            prefill_config,
        )
        .await;
        let decode = kv_set(
            &distributed,
            namespace,
            "decode",
            conditional,
            decode_config,
        )
        .await;
        let endpoint_id = dynamo_runtime::protocols::EndpointId {
            namespace: namespace.to_string(),
            component: "prefill".to_string(),
            name: "generate".to_string(),
        };
        let router = if conditional {
            let (_activation_tx, activation_rx) = tokio::sync::oneshot::channel();
            std::mem::forget(_activation_tx);
            PrefillRouter::new(
                activation_rx,
                Arc::new(ModelManager::new()),
                RouterMode::KV,
                16,
                Some(KvRouterConfig {
                    conditional_disagg_enabled: true,
                    ..Default::default()
                }),
                None,
                None,
                crate::session_affinity::SessionAffinityMode::Hard,
                "model".to_string(),
                namespace.to_string(),
                crate::discovery::LoadThresholdHandle::new(Default::default()),
                tokio_util::sync::CancellationToken::new(),
            )
        } else {
            PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None)
        };
        router.binding.store(Some(Arc::new(PrefillBinding {
            target_id: WorkerSetTargetId::Legacy(endpoint_id.clone()),
            endpoint_id,
            router: Arc::clone(&prefill.host),
            prefill_router_mode: RouterMode::KV,
        })));
        router.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        router
            .set_decode_routing_host(Arc::clone(&decode.host))
            .unwrap();
        Fixture {
            router,
            prefill,
            decode,
            runtime,
        }
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids((1..=64).collect::<Vec<u32>>())
            .stop_conditions(StopConditions {
                max_tokens: Some(32),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    fn bootstrap_frame() -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            disaggregated_params: Some(serde_json::json!({
                "bootstrap_host": "10.0.0.5",
                "bootstrap_port": 8998,
                "bootstrap_room": 42,
            })),
            ..Default::default()
        })
    }

    fn token_frame(token: u32) -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            token_ids: vec![token],
            ..Default::default()
        })
    }

    fn stop_frame() -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        })
    }

    async fn active_requests(chooser: &KvRouter, worker_id: u64) -> usize {
        chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap()
            .iter()
            .find(|load| load.worker_id == worker_id)
            .map_or(0, |load| load.active_requests)
    }

    async fn wait_until_released(fixture: &Fixture) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let prefill =
                    active_requests(&fixture.prefill.chooser, fixture.prefill.worker_id).await;
                let decode =
                    active_requests(&fixture.decode.chooser, fixture.decode.worker_id).await;
                if prefill == 0 && decode == 0 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("both bookings released");
    }

    #[tokio::test]
    async fn the_planned_path_runs_prefill_then_decode_and_hands_decode_the_bootstrap() {
        let fixture = fixture("plan-host-pd").await;
        fixture.prefill.worker.script(vec![bootstrap_frame()]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);

        let response = fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
            .expect("routed");
        let frames: Vec<_> = response.collect().await;
        assert_eq!(frames.len(), 2);
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );

        let prefill_seen = fixture.prefill.worker.seen();
        let decode_seen = fixture.decode.worker.seen();
        assert_eq!(prefill_seen.len(), 1);
        assert_eq!(decode_seen.len(), 1);
        let (prefill_worker, prefill_body) = &prefill_seen[0];
        let (decode_worker, decode_body) = &decode_seen[0];
        assert_eq!(*prefill_worker, fixture.prefill.worker_id);
        assert_eq!(*decode_worker, fixture.decode.worker_id);
        assert_eq!(
            prefill_body.stop_conditions.max_tokens,
            Some(1),
            "prefill runs one token"
        );
        assert_eq!(
            decode_body.stop_conditions.max_tokens,
            Some(32),
            "decode gets the caller's budget back"
        );
        assert_eq!(
            decode_body
                .routing
                .as_ref()
                .and_then(|r| r.prefill_worker_id),
            Some(fixture.prefill.worker_id),
            "decode knows which prefill worker holds the blocks"
        );
        assert!(decode_body.staged_kv_cleanup);
        assert!(decode_body.bootstrap_info.is_some());
        let decode_override = decode_body
            .router_config_override
            .as_ref()
            .expect("decode override");
        assert_eq!(decode_override.track_prefill_tokens, Some(false));
        assert_eq!(decode_override.assume_kv_reuse, Some(false));
        assert!(
            !decode_body
                .annotations
                .iter()
                .any(|a| a == super::BYPASS_REMOTE_PREFILL_ANNOTATION),
            "remote prefill ran, so no bypass annotation"
        );
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    #[tokio::test]
    async fn an_initial_prefill_failure_fails_the_request_before_decode_is_dispatched() {
        let fixture = fixture("plan-host-prefill-fails").await;
        fixture
            .prefill
            .worker
            .script(vec![Annotated::from_error("prefill connection lost")]);

        let error = fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
            .expect_err("prefill failed");
        assert!(
            error.to_string().contains("prefill connection lost"),
            "{error}"
        );
        assert!(
            fixture.decode.worker.seen().is_empty(),
            "decode was never dispatched"
        );
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    #[tokio::test]
    async fn a_late_prefill_failure_after_the_handoff_replaces_decode_success() {
        let fixture = fixture("plan-host-late-fail").await;
        fixture.prefill.worker.script(vec![
            bootstrap_frame(),
            Annotated::from_error("prefill connection lost"),
        ]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);

        // The handoff unblocked decode, so the plan dispatched it; the late
        // failure then reaches the client either before decode resolves (an
        // error return) or after (an error frame ending the stream), as on
        // the legacy path.
        let message = match fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
        {
            Err(error) => error.to_string(),
            Ok(response) => {
                let frames: Vec<_> = response.collect().await;
                let last = frames.last().expect("at least the error frame");
                last.err()
                    .expect("the stream ends with the prefill error")
                    .to_string()
            }
        };
        assert!(message.contains("prefill connection lost"), "{message}");
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    #[tokio::test]
    async fn conditional_disagg_with_nothing_cached_still_runs_remote_prefill() {
        let fixture = fixture_with("plan-host-conditional-remote", true).await;
        fixture.prefill.worker.script(vec![bootstrap_frame()]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);

        let response = fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
            .expect("routed");
        let frames: Vec<_> = response.collect().await;
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );
        assert_eq!(
            fixture.prefill.worker.seen().len(),
            1,
            "the preview found no cache: remote prefill"
        );
        let decode_seen = fixture.decode.worker.seen();
        assert_eq!(decode_seen.len(), 1);
        let (_, decode_body) = &decode_seen[0];
        assert!(
            !decode_body
                .annotations
                .iter()
                .any(|a| a == super::BYPASS_REMOTE_PREFILL_ANNOTATION),
            "no bypass annotation on the remote path"
        );
        // Conditional disaggregation leaves decode's overlap credit to the
        // base config; plain disaggregation zeroes it.
        let decode_override = decode_body
            .router_config_override
            .as_ref()
            .expect("decode override");
        assert_eq!(decode_override.overlap_score_credit, None);
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    #[tokio::test]
    async fn an_explicit_prefill_pin_keeps_remote_prefill_even_with_decode_cached() {
        let fixture = fixture_with("plan-host-pinned-prefill", true).await;
        seed_prefix(&fixture.decode, &(1..=64).collect::<Vec<u32>>()).await;
        fixture.prefill.worker.script(vec![bootstrap_frame()]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);
        let mut body = request();
        body.routing_mut().prefill_worker_id = Some(fixture.prefill.worker_id);
        body.routing_mut().prefill_dp_rank = Some(0);

        let response = fixture
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await
            .expect("routed");
        let frames: Vec<_> = response.collect().await;
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );
        let prefill_seen = fixture.prefill.worker.seen();
        assert_eq!(prefill_seen.len(), 1, "the named prefill worker ran once");
        assert_eq!(prefill_seen[0].0, fixture.prefill.worker_id);
        let decode_seen = fixture.decode.worker.seen();
        assert_eq!(decode_seen.len(), 1);
        assert!(
            !decode_seen[0]
                .1
                .annotations
                .iter()
                .any(|a| a == super::BYPASS_REMOTE_PREFILL_ANNOTATION),
            "no bypass with an explicit prefill worker"
        );
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    /// A custom `Router` a deployment might install: prompts under a size go
    /// straight to decode, which prefills them locally; everything else is
    /// the default shape. It wraps the default and delegates scheduling.
    struct ShortPromptRouter {
        inner: Arc<dyn Router>,
        partition: dynamo_kv_router::identity::RoutingPartitionId,
        max_tokens: usize,
    }

    #[async_trait]
    impl Router for ShortPromptRouter {
        async fn select(
            &self,
            req: dynamo_kv_router::services::selection::SelectRequest,
        ) -> Result<
            dynamo_kv_router::services::selection::SelectResponse,
            dynamo_kv_router::services::selection::SelectionError,
        > {
            self.inner.select(req).await
        }

        fn plan(
            &self,
            req: &dynamo_kv_router::services::selection::SelectAndReserveRequest,
        ) -> Result<Plan, dynamo_kv_router::services::selection::SelectionError> {
            let prompt = req.prompt.token_ids.as_ref().map_or(0, Vec::len);
            if prompt < self.max_tokens {
                Plan::new(
                    dynamo_kv_router::router::PlanId::from(
                        req.selection_id.clone().unwrap_or_default(),
                    ),
                    self.partition.clone(),
                    vec![dynamo_kv_router::router::Stage::new(WorkerType::Decode)],
                )
                .map_err(|error| {
                    dynamo_kv_router::services::selection::SelectionError::BadRequest(
                        error.to_string(),
                    )
                })
            } else {
                self.inner.plan(req)
            }
        }

        async fn schedule(
            &self,
            req: &dynamo_kv_router::services::selection::SelectAndReserveRequest,
            plan: &mut Plan,
        ) -> Result<(), dynamo_kv_router::services::selection::SelectionError> {
            self.inner.schedule(req, plan).await
        }
    }

    #[tokio::test]
    async fn a_custom_router_installed_through_the_factory_drives_the_same_host_loop() {
        let fixture = fixture("plan-host-custom-router").await;
        let partition = fixture.decode.chooser.selection.partition_key().clone();
        fixture
            .router
            .set_plan_router_factory(Arc::new(move |parts| {
                let inner = super::default_plan_router(parts)?;
                Ok(Arc::new(ShortPromptRouter {
                    inner,
                    partition: partition.clone(),
                    max_tokens: 8,
                }) as Arc<dyn Router>)
            }))
            .unwrap();

        // Four tokens: the custom shape, decode only, prefilled locally.
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);
        let mut short = request();
        short.token_ids = Arc::new(vec![1, 2, 3, 4]);
        let response = fixture
            .router
            .generate(Context::new(short), Arc::new(NeverNext))
            .await
            .expect("routed");
        let frames: Vec<_> = response.collect().await;
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );
        assert!(
            fixture.prefill.worker.seen().is_empty(),
            "no prefill stage in the plan"
        );
        let decode_seen = fixture.decode.worker.seen();
        assert_eq!(decode_seen.len(), 1);
        assert!(
            decode_seen[0]
                .1
                .annotations
                .iter()
                .any(|a| a == super::BYPASS_REMOTE_PREFILL_ANNOTATION),
            "decode prefills locally"
        );
        wait_until_released(&fixture).await;

        // Sixty-four tokens: the default shape through the same loop.
        fixture.prefill.worker.script(vec![bootstrap_frame()]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);
        let response = fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
            .expect("routed");
        let frames: Vec<_> = response.collect().await;
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );
        assert_eq!(
            fixture.prefill.worker.seen().len(),
            1,
            "prefill ran remotely"
        );
        assert_eq!(fixture.decode.worker.seen().len(), 2);
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    /// Record the whole prompt as cached on `worker_id`, so the decode
    /// preview sees a full prefix hit there.
    /// The decode worker requires its KV-transfer peers in zone b; the
    /// prefill worker publishes no zone. The forward constraints prefill
    /// was selected under cannot see decode's requirement, so the core
    /// checks the pair before recording decode and frees the refused
    /// booking; prefill's booking stays with the plan for the host.
    #[tokio::test]
    async fn an_incompatible_transfer_pair_is_refused_and_nothing_is_kept() {
        use dynamo_kv_router::protocols::KvTransferEnforcement;
        use dynamo_kv_router::router::{ClassTable, MultiStageRouter, StageList};
        use dynamo_kv_router::services::selection::SelectionError;
        use std::collections::HashSet;

        use crate::kv_router::plan_host::routing_request;

        let fixture = fixture_configs(
            "plan-host-reverse-transfer",
            false,
            ModelRuntimeConfig::default(),
            ModelRuntimeConfig {
                taints: HashSet::from(["dynamo.topology/zone=b".to_string()]),
                topology_domains: HashMap::from([("zone".to_string(), "b".to_string())]),
                kv_transfer_domain: Some("zone".to_string()),
                kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
                ..ModelRuntimeConfig::default()
            },
        )
        .await;
        let request = Context::new(request());
        let req = routing_request(&request, &fixture.decode.chooser.partition(), None, false);
        let router = MultiStageRouter::builder()
            .set(WorkerType::Prefill, fixture.prefill.chooser.plan_router())
            .set(WorkerType::Decode, fixture.decode.chooser.plan_router())
            .classes(ClassTable::new(StageList::prefill_decode()))
            .build()
            .unwrap();
        let mut plan = router.plan(&req).unwrap();
        let result = router.schedule(&req, &mut plan).await;
        assert!(
            matches!(result, Err(SelectionError::Conflict(_))),
            "the pair is refused as a placement conflict: {result:?}"
        );
        assert!(
            plan.state_of(1).unwrap().is_pending(),
            "decode was not recorded"
        );
        assert!(
            plan.booking(0).is_some(),
            "prefill stays booked for the host"
        );
        plan.release().await.unwrap();
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    /// A decode worker whose metadata discovery has withdrawn is not a
    /// placement, even though its endpoint instance is still registered and
    /// its last-known facts were compatible with the prefill side. The set
    /// has no admitted worker, so the plan is refused before anything is
    /// dispatched: no prefill is spent on a request that has nowhere to
    /// decode, and nothing is left booked.
    #[tokio::test]
    async fn a_decode_worker_without_metadata_is_not_a_placement() {
        use dynamo_kv_router::protocols::KvTransferEnforcement;
        use std::collections::HashSet;

        let zone_b = || ModelRuntimeConfig {
            taints: HashSet::from(["dynamo.topology/zone=b".to_string()]),
            topology_domains: HashMap::from([("zone".to_string(), "b".to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
            ..ModelRuntimeConfig::default()
        };
        let fixture = fixture_configs(
            "plan-host-decode-metadata-withdrawn",
            false,
            zone_b(),
            zone_b(),
        )
        .await;
        fixture.prefill.worker.script(vec![bootstrap_frame()]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);

        // Discovery withdraws the decode worker's metadata; its endpoint
        // instance stays registered.
        fixture.decode.workers_tx.send_replace(HashMap::new());
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let loads = fixture
                    .decode
                    .chooser
                    .get_potential_loads(&[], None, None, None, None)
                    .await
                    .unwrap();
                if !loads
                    .iter()
                    .any(|load| load.worker_id == fixture.decode.worker_id)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the decode set observes the withdrawal");

        let result = fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await;
        let error = match result {
            Err(error) => error,
            Ok(stream) => {
                let frames: Vec<_> = stream.collect().await;
                panic!("a worker without metadata was placed: {frames:?}");
            }
        };
        assert!(
            fixture.decode.worker.seen().is_empty(),
            "decode was never dispatched: {error:#}"
        );
        assert!(
            fixture.prefill.worker.seen().is_empty(),
            "the plan is refused before prefill is dispatched: {error:#}"
        );
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    /// The two answers that are the client's, through the real frontend
    /// loop: a request withdrawn before it was planned, and a decode worker
    /// the set does not have. Each leaves with the class a direct admission
    /// reports (499 and 400 at the HTTP layer).
    #[tokio::test]
    async fn client_cancellation_and_validation_keep_their_class_through_the_loop() {
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::timing::RequestPhase;
        use dynamo_runtime::error::DynamoError;

        let fixture = fixture("plan-host-client-errors").await;
        let class_of = |error: &anyhow::Error| {
            error
                .downcast_ref::<DynamoError>()
                .unwrap_or_else(|| panic!("a canonical frontend error: {error:?}"))
                .error_type()
        };
        let direct = |request: Context<PreprocessedRequest>| {
            let host = Arc::clone(&fixture.decode.host);
            async move {
                host.admit_kv_route(
                    &request,
                    RequestPhase::Decode,
                    None,
                    CleanupBudget::default(),
                    None,
                )
                .await
                .err()
                .expect("the direct admission is refused")
            }
        };

        // Withdrawn before planning.
        let cancelled = || {
            let request = Context::new(request());
            request.context().stop();
            request
        };
        let Err(planned) = fixture
            .router
            .generate(cancelled(), Arc::new(NeverNext))
            .await
        else {
            panic!("a withdrawn request is not planned");
        };
        assert_eq!(class_of(&planned), class_of(&direct(cancelled()).await));

        // Pinned to a decode worker the set does not have.
        let unknown = || {
            let mut body = request();
            body.routing_mut().decode_worker_id = Some(fixture.decode.worker_id + 1_000);
            body.routing_mut().dp_rank = Some(0);
            Context::new(body)
        };
        let Err(planned) = fixture
            .router
            .generate(unknown(), Arc::new(NeverNext))
            .await
        else {
            panic!("an unknown decode worker is refused");
        };
        assert_eq!(class_of(&planned), class_of(&direct(unknown()).await));
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    /// A caller naming a prefill worker without a rank gets that worker's
    /// unique rank, as the legacy preview resolves it; rank 0 does not exist
    /// on this worker.
    #[tokio::test]
    async fn a_worker_only_prefill_pin_resolves_the_workers_unique_rank() {
        let fixture = fixture_configs(
            "plan-host-pin-rank",
            false,
            ModelRuntimeConfig {
                data_parallel_size: 1,
                data_parallel_start_rank: 3,
                ..ModelRuntimeConfig::default()
            },
            ModelRuntimeConfig::default(),
        )
        .await;
        fixture.prefill.worker.script(vec![bootstrap_frame()]);
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);
        let mut body = request();
        body.routing_mut().prefill_worker_id = Some(fixture.prefill.worker_id);

        let response = fixture
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await
            .expect("the unique rank is resolved");
        let frames: Vec<_> = response.collect().await;
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );
        let prefill_seen = fixture.prefill.worker.seen();
        assert_eq!(prefill_seen.len(), 1, "the named prefill worker ran once");
        assert_eq!(
            prefill_seen[0]
                .1
                .routing
                .as_ref()
                .and_then(|r| r.prefill_dp_rank),
            Some(3),
            "the pin carries the worker's own rank"
        );
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    async fn seed_prefix(set: &KvSet, tokens: &[u32]) {
        use dynamo_kv_router::indexer::KvIndexerInterface;
        use dynamo_kv_router::protocols::{
            BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
            KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        };
        use dynamo_kv_router::services::indexer::backend::Indexer;
        let local_hashes = compute_block_hash_for_seq(tokens, 16, BlockHashOptions::default());
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
        let indexer = set.chooser.indexer();
        indexer
            .apply_event_routed(RouterEvent::with_storage_tier(
                set.worker_id,
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
        if let Indexer::Single { primary, .. } = indexer {
            primary.flush().await;
        }
    }

    #[tokio::test]
    async fn conditional_disagg_with_the_prompt_cached_on_decode_bypasses_remote_prefill() {
        let fixture = fixture_with("plan-host-conditional-bypass", true).await;
        let tokens: Vec<u32> = (1..=64).collect();
        seed_prefix(&fixture.decode, &tokens).await;
        fixture
            .decode
            .worker
            .script(vec![token_frame(7), stop_frame()]);

        let response = fixture
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
            .expect("routed");
        let frames: Vec<_> = response.collect().await;
        assert!(
            frames.iter().all(|frame| frame.err().is_none()),
            "{frames:?}"
        );
        assert!(
            frames[0]
                .event
                .as_deref()
                .is_some_and(|event| event == super::BYPASS_REMOTE_PREFILL_ANNOTATION),
            "the bypass annotation leads the stream: {frames:?}"
        );
        assert!(
            fixture.prefill.worker.seen().is_empty(),
            "prefill was skipped"
        );
        let decode_seen = fixture.decode.worker.seen();
        assert_eq!(decode_seen.len(), 1);
        let (worker, decode_body) = &decode_seen[0];
        assert_eq!(
            *worker, fixture.decode.worker_id,
            "decode went to the previewed worker"
        );
        assert!(
            decode_body
                .annotations
                .iter()
                .any(|a| a == super::BYPASS_REMOTE_PREFILL_ANNOTATION),
            "decode runs the prefill locally"
        );
        assert_eq!(decode_body.stop_conditions.max_tokens, Some(32));
        wait_until_released(&fixture).await;
        fixture.runtime.shutdown();
    }

    /// The reviewer's acceptance probe: a hard prefill session bound through
    /// the planned path names the same worker a direct admission binds.
    #[tokio::test]
    async fn reviewer_direct_host_preserves_prefill_affinity_binding() {
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::extensions::{
            SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
        };
        use crate::protocols::common::timing::RequestPhase;
        use crate::session_affinity::{AffinityCoordinator, SessionAffinityMode};

        let mut f = fixture("review-direct-prefill-affinity").await;
        let affinity =
            AffinityCoordinator::new(Duration::from_secs(30), SessionAffinityMode::Hard).unwrap();
        let inner = PushRouter::from_client_with_dispatch(
            f.prefill.chooser.client().clone(),
            RouterMode::KV,
            Arc::clone(&f.prefill.worker) as Arc<dyn StreamingDispatch<_, _>>,
        )
        .await
        .unwrap();
        f.prefill.host = Arc::new(RoutingHost::new_with_coordinator(
            inner,
            Arc::clone(&f.prefill.chooser),
            Some(affinity.clone()),
        ));
        let previous = f.router.binding.load_full().unwrap();
        f.router.binding.store(Some(Arc::new(PrefillBinding {
            target_id: previous.target_id.clone(),
            endpoint_id: previous.endpoint_id.clone(),
            router: Arc::clone(&f.prefill.host),
            prefill_router_mode: RouterMode::KV,
        })));
        f.prefill.worker.script(vec![bootstrap_frame()]);
        f.decode.worker.script(vec![token_frame(7), stop_frame()]);
        let planned_id = SessionAffinityId::new("planned-prefill-session");
        let mut planned = Context::new(request());
        planned.insert(SESSION_AFFINITY_CONTEXT_KEY, planned_id.clone());
        let response = f
            .router
            .generate(planned, Arc::new(NeverNext))
            .await
            .unwrap();
        let frames: Vec<_> = response.collect().await;
        assert!(frames.iter().all(|frame| frame.err().is_none()));
        wait_until_released(&f).await;
        let planned_target = affinity.query_target(&planned_id, None).unwrap();

        f.prefill.worker.script(vec![bootstrap_frame()]);
        let direct_id = SessionAffinityId::new("direct-prefill-session");
        let mut direct = Context::new(request());
        direct.insert(SESSION_AFFINITY_CONTEXT_KEY, direct_id.clone());
        let route = f
            .prefill
            .host
            .admit_kv_route(
                &direct,
                RequestPhase::Prefill,
                None,
                CleanupBudget::default(),
                None,
            )
            .await
            .unwrap();
        let response = f
            .prefill
            .host
            .dispatch_kv_plan(direct, route)
            .await
            .unwrap();
        let _: Vec<_> = response.collect().await;
        wait_until_released(&f).await;
        let direct_target = affinity.query_target(&direct_id, None).unwrap();
        f.runtime.shutdown();
        assert_eq!(
            direct_target.map(|target| target.worker_id),
            Some(f.prefill.worker_id)
        );
        assert_eq!(
            planned_target, direct_target,
            "planned prefill dispatch must bind the same affinity target as direct dispatch"
        );
    }

    /// The reviewer's acceptance probe: a migration retry never reaches the
    /// decode worker the previous attempt failed on.
    #[tokio::test]
    async fn reviewer_direct_host_preserves_migration_exclusions() {
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::timing::RequestPhase;
        use dynamo_runtime::error::{DynamoError, ErrorType};

        let f = fixture("review-direct-migration-exclusions").await;
        let mut body = request();
        body.migration_state = Some(Default::default());
        body.migration_state.as_ref().unwrap().record_failure(
            f.decode.worker_id,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("the previous attempt failed on this decode worker")
                    .build(),
            ),
        );
        let direct = f
            .decode
            .host
            .admit_kv_route(
                &Context::new(body.clone()),
                RequestPhase::Decode,
                None,
                CleanupBudget::default(),
                None,
            )
            .await;
        assert!(
            direct.is_err(),
            "the direct control rejects its only excluded decode worker"
        );
        f.prefill.worker.script(vec![bootstrap_frame()]);
        f.decode.worker.script(vec![token_frame(7), stop_frame()]);
        let response = f
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await;
        if let Ok(response) = response {
            let _: Vec<_> = response.collect().await;
        }
        wait_until_released(&f).await;
        let sent = f.decode.worker.seen();
        f.runtime.shutdown();
        assert!(
            sent.is_empty(),
            "the planned host must not dispatch to the decode worker excluded by the prior migration attempt"
        );
    }

    /// The reviewer's acceptance probe: a planned admission waiting in a
    /// saturated queue withdraws when the client stops, as a direct
    /// admission does, without the task being aborted from outside.
    #[tokio::test]
    async fn reviewer_direct_host_cancels_queued_admission() {
        use crate::kv_router::plan_host::routing_request;
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::timing::RequestPhase;
        use dynamo_kv_router::router::{ClassTable, MultiStageRouter, Stage, StageList};
        use dynamo_kv_router::scheduling::config::RouterPrefillLoadModel;

        let original = fixture("review-direct-queued-cancellation").await;
        let distributed =
            DistributedRuntime::new(original.runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let policy =
            std::env::temp_dir().join(format!("review-direct-cancel-{}.yaml", std::process::id()));
        std::fs::write(
            &policy,
            "default_policy_family: regular\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: all\npolicy_classes:\n  - name: saturated\n    policy_family: regular\n    cache_bucket: all\n    quantum: 1\n    prefill_busy_threshold_frac: 0.0\n    request_queue_limit_per_worker: 1\n",
        )
        .unwrap();
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            router_prefill_load_model: RouterPrefillLoadModel::Ais,
            router_policy_config: Some(policy.to_string_lossy().into_owned()),
            ..Default::default()
        };
        let decode = kv_set_with_config(
            &distributed,
            "review-direct-queued-cancellation",
            "decode",
            config,
            ModelRuntimeConfig {
                max_num_batched_tokens: Some(8192),
                total_kv_blocks: Some(1_000_000),
                ..Default::default()
            },
        )
        .await;
        let binding = original.router.binding.load_full().unwrap();
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        router.binding.store(Some(Arc::new(PrefillBinding {
            target_id: binding.target_id.clone(),
            endpoint_id: binding.endpoint_id.clone(),
            router: Arc::clone(&original.prefill.host),
            prefill_router_mode: RouterMode::KV,
        })));
        router.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        router
            .set_decode_routing_host(Arc::clone(&decode.host))
            .unwrap();
        router
            .set_plan_router_factory(Arc::new(|parts| {
                Ok(Arc::new(
                    MultiStageRouter::builder()
                        .set(WorkerType::Decode, parts.decode)
                        .classes(ClassTable::new(StageList::new(vec![Stage::new(
                            WorkerType::Decode,
                        )])))
                        .build()?,
                ) as Arc<dyn Router>)
            }))
            .unwrap();
        let f = Fixture {
            router,
            prefill: original.prefill,
            decode,
            runtime: original.runtime,
        };
        // Hold the worker's one slot so the next admissions queue.
        let context = Context::new(request());
        let core = f.decode.chooser.plan_router();
        let req = routing_request(&context, &f.decode.chooser.partition(), None, false);
        let mut held = core.plan(&req).unwrap();
        core.schedule(&req, &mut held).await.unwrap();

        let chooser = Arc::clone(&f.decode.chooser);
        let wait_for_pending = |expected: usize| {
            let chooser = Arc::clone(&chooser);
            async move {
                tokio::time::timeout(Duration::from_secs(2), async {
                    while chooser.pending_count() != expected {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap_or_else(|_| panic!("the queue did not reach {expected} pending"));
            }
        };

        // Direct control: stop withdraws the queued admission.
        let direct_context = Context::new(request());
        let direct_engine = direct_context.context();
        let host = Arc::clone(&f.decode.host);
        let mut direct = tokio::spawn(async move {
            host.admit_kv_route(
                &direct_context,
                RequestPhase::Decode,
                None,
                CleanupBudget::default(),
                None,
            )
            .await
        });
        wait_for_pending(1).await;
        direct_engine.stop();
        let direct_cancelled =
            match tokio::time::timeout(Duration::from_millis(250), &mut direct).await {
                Ok(result) => result.unwrap().is_err(),
                Err(_) => {
                    direct.abort();
                    let _ = direct.await;
                    false
                }
            };
        wait_for_pending(0).await;

        // Planned: the same stop withdraws the same queue entry.
        let planned_context = Context::new(request());
        let planned_engine = planned_context.context();
        let router = Arc::clone(&f.router);
        let mut planned =
            tokio::spawn(
                async move { router.generate(planned_context, Arc::new(NeverNext)).await },
            );
        wait_for_pending(1).await;
        planned_engine.stop();
        let planned_cancelled =
            match tokio::time::timeout(Duration::from_millis(250), &mut planned).await {
                Ok(result) => result.unwrap().is_err(),
                Err(_) => {
                    planned.abort();
                    let _ = planned.await;
                    false
                }
            };
        wait_for_pending(0).await;
        held.release().await.unwrap();
        wait_until_released(&f).await;
        assert!(f.prefill.worker.seen().is_empty() && f.decode.worker.seen().is_empty());
        std::fs::remove_file(policy).unwrap();
        f.runtime.shutdown();
        assert!(direct_cancelled, "direct control responds to stop");
        assert!(
            planned_cancelled,
            "planned admission must retract on stop without external task abortion"
        );
    }

    /// A fixture whose prefill set runs in its own runtime, so the two sets
    /// have distinct worker ids and a failure recorded against one cannot
    /// alias the other.
    async fn distinct_prefill_fixture(namespace: &str, conditional: bool) -> Fixture {
        let mut f = fixture_with(namespace, conditional).await;
        let distributed =
            DistributedRuntime::new(f.runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let prefill_namespace = format!("{namespace}-separate-prefill");
        f.prefill = kv_set(
            &distributed,
            &prefill_namespace,
            "prefill",
            conditional,
            ModelRuntimeConfig::default(),
        )
        .await;
        let endpoint_id = dynamo_runtime::protocols::EndpointId {
            namespace: prefill_namespace,
            component: "prefill".to_string(),
            name: "generate".to_string(),
        };
        f.router.binding.store(Some(Arc::new(PrefillBinding {
            target_id: WorkerSetTargetId::Legacy(endpoint_id.clone()),
            endpoint_id,
            router: Arc::clone(&f.prefill.host),
            prefill_router_mode: RouterMode::KV,
        })));
        assert_ne!(
            f.prefill.worker_id, f.decode.worker_id,
            "independent worker runtimes must have distinct ids"
        );
        f
    }

    /// The reviewer's probe: a decode-only plan has no prefill stage, so an
    /// exhausted prefill set is not this request's concern.
    #[tokio::test]
    async fn reviewer_monitor_decode_only_ignores_excluded_prefill_pool() {
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::timing::RequestPhase;
        use dynamo_kv_router::router::{ClassTable, MultiStageRouter, Stage, StageList};
        use dynamo_runtime::error::{DynamoError, ErrorType};

        let f = distinct_prefill_fixture("monitor-unused-prefill-exclusion", false).await;
        f.router
            .set_plan_router_factory(Arc::new(|parts| {
                Ok(Arc::new(
                    MultiStageRouter::builder()
                        .set(WorkerType::Decode, parts.decode)
                        .classes(ClassTable::new(StageList::new(vec![Stage::new(
                            WorkerType::Decode,
                        )])))
                        .build()?,
                ) as Arc<dyn Router>)
            }))
            .unwrap();
        let mut body = request();
        body.migration_state = Some(Default::default());
        body.migration_state.as_ref().unwrap().record_failure(
            f.prefill.worker_id,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("the previous prefill attempt failed")
                    .build(),
            ),
        );
        let direct = f
            .decode
            .host
            .admit_kv_route(
                &Context::new(body.clone()),
                RequestPhase::Decode,
                None,
                CleanupBudget::default(),
                None,
            )
            .await
            .expect("the healthy decode pool is eligible");
        direct.abort().await;
        wait_until_released(&f).await;
        f.decode.worker.script(vec![token_frame(7), stop_frame()]);
        let response = f
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await;
        let succeeded = match response {
            Ok(response) => {
                let frames: Vec<_> = response.collect().await;
                frames.iter().all(|frame| frame.err().is_none())
            }
            Err(_) => false,
        };
        wait_until_released(&f).await;
        let prefill_seen = f.prefill.worker.seen();
        let decode_seen = f.decode.worker.seen();
        f.runtime.shutdown();
        assert!(prefill_seen.is_empty());
        assert!(
            succeeded,
            "a decode-only plan must ignore exhaustion in an absent prefill stage"
        );
        assert_eq!(decode_seen.len(), 1);
    }

    /// The reviewer's probe: a conditional bypass skips prefill, so an
    /// exhausted prefill set does not refuse the cached decode.
    #[tokio::test]
    async fn reviewer_monitor_cached_bypass_ignores_excluded_prefill_pool() {
        use dynamo_runtime::error::{DynamoError, ErrorType};

        let f = distinct_prefill_fixture("monitor-skipped-prefill-exclusion", true).await;
        seed_prefix(&f.decode, &(1..=64).collect::<Vec<u32>>()).await;
        f.decode.worker.script(vec![token_frame(7), stop_frame()]);
        let control = f
            .router
            .generate(Context::new(request()), Arc::new(NeverNext))
            .await
            .unwrap();
        let frames: Vec<_> = control.collect().await;
        assert!(
            frames[0]
                .event
                .as_deref()
                .is_some_and(|event| event == super::BYPASS_REMOTE_PREFILL_ANNOTATION)
        );
        assert!(f.prefill.worker.seen().is_empty());
        wait_until_released(&f).await;

        let mut body = request();
        body.migration_state = Some(Default::default());
        body.migration_state.as_ref().unwrap().record_failure(
            f.prefill.worker_id,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("the previous prefill attempt failed")
                    .build(),
            ),
        );
        f.decode.worker.script(vec![token_frame(8), stop_frame()]);
        let response = f
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await;
        let succeeded = match response {
            Ok(response) => {
                let frames: Vec<_> = response.collect().await;
                frames.iter().all(|frame| frame.err().is_none())
            }
            Err(_) => false,
        };
        wait_until_released(&f).await;
        let prefill_seen = f.prefill.worker.seen();
        let decode_seen = f.decode.worker.seen();
        f.runtime.shutdown();
        assert!(prefill_seen.is_empty());
        assert!(
            succeeded,
            "a cached decode bypass must not be refused because its unused prefill pool is exhausted"
        );
        assert_eq!(decode_seen.len(), 2);
    }

    /// The reviewer's control: a client that stops or is killed while prefill
    /// is being dispatched still gets its decode leg, which carries the
    /// staged-KV cleanup, and every projected charge clears afterwards.
    #[tokio::test]
    async fn reviewer_monitor_after_prefill_cancellation_keeps_cleanup() {
        use dynamo_kv_router::router::{ClassTable, MultiStageRouter, StageList};

        for (deferred, kill) in [(false, false), (false, true), (true, false), (true, true)] {
            let namespace = match (deferred, kill) {
                (false, false) => "review-upfront-prefill-stop",
                (false, true) => "review-upfront-prefill-kill",
                (true, false) => "review-deferred-prefill-stop",
                (true, true) => "review-deferred-prefill-kill",
            };
            let fixture = fixture(namespace).await;
            if deferred {
                fixture
                    .router
                    .set_plan_router_factory(Arc::new(|parts| {
                        let router = MultiStageRouter::builder()
                            .set(WorkerType::Prefill, parts.prefill)
                            .set(WorkerType::Decode, parts.decode)
                            .classes(ClassTable::new(StageList::prefill_decode_deferred(
                                Duration::from_secs(1),
                            )))
                            .build()?;
                        Ok(Arc::new(router) as Arc<dyn Router>)
                    }))
                    .unwrap();
            }
            fixture.prefill.worker.script(vec![bootstrap_frame()]);
            fixture.decode.worker.script(vec![stop_frame()]);
            let context = Context::new(request());
            let client = context.context();
            let cancel = Arc::clone(&client);
            *fixture.prefill.worker.on_dispatch.lock().unwrap() = Some(Box::new(move || {
                if kill {
                    cancel.kill();
                } else {
                    cancel.stop_generating();
                }
            }));
            let response = tokio::time::timeout(
                Duration::from_secs(5),
                fixture.router.generate(context, Arc::new(NeverNext)),
            )
            .await
            .expect("routing terminates");
            if let Ok(response) = response {
                tokio::time::timeout(Duration::from_secs(5), response.collect::<Vec<_>>())
                    .await
                    .expect("cleanup stream terminates");
            }
            assert!(
                client.is_stopped() || client.is_killed(),
                "client cancellation happened during prefill dispatch"
            );
            let prefill_seen = fixture.prefill.worker.seen();
            let decode_seen = fixture.decode.worker.seen();
            wait_until_released(&fixture).await;
            for set in [&fixture.prefill, &fixture.decode] {
                let loads = set
                    .chooser
                    .get_potential_loads(&[], None, None, None, None)
                    .await
                    .unwrap();
                assert!(
                    loads.iter().all(|load| load.active_requests == 0
                        && load.potential_prefill_tokens == 0
                        && load.potential_decode_blocks == 0),
                    "all projected charges clear after the cancelled flow: {loads:?}"
                );
            }
            fixture.runtime.shutdown();
            assert_eq!(
                prefill_seen.len(),
                1,
                "prefill was dispatched before cancellation"
            );
            assert_eq!(
                decode_seen.len(),
                1,
                "decode must still receive the staged KV cleanup request"
            );
            let (worker, body) = &decode_seen[0];
            assert_eq!(*worker, fixture.decode.worker_id);
            assert!(body.staged_kv_cleanup);
            assert!(
                body.bootstrap_info.is_some(),
                "cleanup dispatch carries the prefill handoff"
            );
            assert_eq!(
                body.routing.as_ref().and_then(|r| r.prefill_worker_id),
                Some(fixture.prefill.worker_id)
            );
        }
    }

    /// The reviewer's probe: the aggregated fallback keeps a migration
    /// retry's excluded decode worker excluded, so a request that fell back
    /// once does not reach the worker it then failed on.
    #[tokio::test]
    async fn reviewer_independent_fallback_preserves_failed_decode_exclusion() {
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::preprocessor::RoutingHints;
        use crate::protocols::common::timing::RequestPhase;
        use dynamo_kv_router::router::{ClassTable, Fallback, MultiStageRouter, StageList};
        use dynamo_runtime::error::{DynamoError, ErrorType};

        let f = distinct_prefill_fixture("review-fallback-exclusion", false).await;
        f.router
            .set_plan_router_factory(Arc::new(|parts| {
                Ok(Arc::new(
                    MultiStageRouter::builder()
                        .set(WorkerType::Prefill, parts.prefill)
                        .set(WorkerType::Decode, parts.decode)
                        .classes(ClassTable::new(
                            StageList::prefill_decode().with_fallback(Fallback::Aggregated),
                        ))
                        .build()?,
                ) as Arc<dyn Router>)
            }))
            .unwrap();
        let mut body = request();
        body.routing = Some(RoutingHints {
            allowed_worker_ids: Some(HashSet::from([f.decode.worker_id])),
            ..Default::default()
        });
        f.decode.worker.script(vec![token_frame(7), stop_frame()]);
        let control = f
            .router
            .generate(Context::new(body.clone()), Arc::new(NeverNext))
            .await
            .unwrap();
        let frames: Vec<_> = control.collect().await;
        assert!(frames.iter().all(|frame| frame.err().is_none()));
        wait_until_released(&f).await;
        assert!(
            f.prefill.worker.seen().is_empty(),
            "the control used the decode fallback"
        );
        assert_eq!(f.decode.worker.seen().len(), 1);

        body.migration_state = Some(Default::default());
        body.migration_state.as_ref().unwrap().record_failure(
            f.decode.worker_id,
            Some(
                DynamoError::builder()
                    .error_type(ErrorType::WorkerOverloaded)
                    .message("decode already failed")
                    .build(),
            ),
        );
        let direct = f
            .decode
            .host
            .admit_kv_route(
                &Context::new(body.clone()),
                RequestPhase::Decode,
                None,
                CleanupBudget::default(),
                None,
            )
            .await;
        assert!(direct.is_err(), "direct decode refuses the failed worker");
        f.decode.worker.script(vec![token_frame(8), stop_frame()]);
        let result = f
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await;
        if let Ok(response) = result {
            let _: Vec<_> = response.collect().await;
        }
        wait_until_released(&f).await;
        let seen = f.decode.worker.seen();
        f.runtime.shutdown();
        assert_eq!(
            seen.len(),
            1,
            "the fallback must not dispatch a second request to the excluded decode worker"
        );
    }

    /// A worker picker that records the session context each selection
    /// hands it: what a compiled custom policy gets to decide on.
    struct SessionRecordingPicker {
        observed: Arc<Mutex<Vec<Option<dynamo_kv_router::SessionContext>>>>,
    }

    impl dynamo_kv_router::plugins::worker_selection::WorkerPicker for SessionRecordingPicker {
        fn pick(
            &mut self,
            context: &dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext<'_>,
            _input: dynamo_kv_router::plugins::worker_selection::WorkerInputView<'_>,
        ) -> std::result::Result<
            usize,
            dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyError,
        > {
            self.observed
                .lock()
                .unwrap()
                .push(context.session_context().cloned());
            Ok(0)
        }
    }

    /// The reviewer's probe: a compiled custom picker on the decode core sees
    /// the same agent session context through the planned path as through a
    /// direct admission.
    #[tokio::test]
    async fn reviewer_independent_custom_picker_receives_agent_context() {
        use crate::kv_router::routing_host::CleanupBudget;
        use crate::protocols::common::extensions::{AgentContextBuilder, InputTrigger};
        use crate::protocols::common::timing::RequestPhase;

        let mut f = fixture("review-custom-session-context").await;
        let drt = DistributedRuntime::new(f.runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let capture = Arc::clone(&observed);
        let factory: dynamo_kv_router::WorkerSelectionPolicyFactory =
            Arc::new(move |config, role, _| {
                dynamo_kv_router::WorkerSelectionPolicy::new(
                    config.clone(),
                    role.as_str(),
                    Vec::new(),
                    Box::new(SessionRecordingPicker {
                        observed: Arc::clone(&capture),
                    }),
                )
            });
        f.decode = kv_set_with_policy(
            &drt,
            "review-custom-session-context-decode",
            "decode",
            KvRouterConfig {
                skip_initial_worker_wait: true,
                use_kv_events: false,
                router_track_active_blocks: false,
                ..Default::default()
            },
            ModelRuntimeConfig::default(),
            SelectionPolicySource::Factory(factory),
        )
        .await;
        // The decode host is set once per router: rebuild the router around
        // the recording decode set.
        let binding = f.router.binding.load_full().unwrap();
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        router.binding.store(Some(Arc::new(PrefillBinding {
            target_id: binding.target_id.clone(),
            endpoint_id: binding.endpoint_id.clone(),
            router: Arc::clone(&f.prefill.host),
            prefill_router_mode: RouterMode::KV,
        })));
        router.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        router
            .set_decode_routing_host(Arc::clone(&f.decode.host))
            .unwrap();
        f.router = router;

        let mut agent = AgentContextBuilder::default();
        agent.session_id("session-a".to_string());
        agent.parent_session_id("parent-a".to_string());
        agent.session_final(false);
        agent.input_trigger(InputTrigger::ToolResult);
        let mut body = request();
        body.agent_context = Some(agent.build().unwrap());
        let expected = crate::kv_router::to_worker_selection_session_context(
            body.agent_context.as_ref().unwrap(),
        );
        let control = f
            .decode
            .host
            .admit_kv_route(
                &Context::new(body.clone()),
                RequestPhase::Decode,
                None,
                CleanupBudget::default(),
                None,
            )
            .await
            .unwrap();
        control.abort().await;
        wait_until_released(&f).await;
        {
            let seen = observed.lock().unwrap();
            assert!(!seen.is_empty());
            assert!(
                seen.iter().all(|item| item.as_ref() == Some(&expected)),
                "the direct picker sees the session metadata: {seen:?}"
            );
        }
        observed.lock().unwrap().clear();

        f.prefill.worker.script(vec![bootstrap_frame()]);
        f.decode.worker.script(vec![token_frame(8), stop_frame()]);
        let response = f
            .router
            .generate(Context::new(body), Arc::new(NeverNext))
            .await
            .unwrap();
        let frames: Vec<_> = response.collect().await;
        assert!(frames.iter().all(|frame| frame.err().is_none()));
        wait_until_released(&f).await;
        let seen = observed.lock().unwrap().clone();
        f.runtime.shutdown();
        assert!(!seen.is_empty(), "the planned admission called the picker");
        assert!(
            seen.iter().all(|item| item.as_ref() == Some(&expected)),
            "the planned picker sees the same agent context as a direct admission: {seen:?}"
        );
    }

    /// A rolling replacement as the planned path sees it: the set's admitted
    /// membership is the one input it routes by, intersected with discovered
    /// endpoints as the direct path does. A plan booked on the retiring
    /// worker keeps its booking and releases it after the worker has left
    /// the membership. Once withdrawn, the worker's still-present endpoint
    /// instance is not a placement: with no admitted worker left a new plan
    /// is refused rather than routed to the retired instance or to an
    /// instance without metadata.
    #[tokio::test]
    async fn membership_withdrawal_keeps_the_retiring_booking_and_refuses_new_plans() {
        use crate::kv_router::plan_host::routing_request;
        use dynamo_kv_router::services::selection::SelectionError;

        let f = fixture("plan-host-membership-withdrawal").await;
        let retiring = f.decode.worker_id;
        let router = f.decode.chooser.plan_router();
        let partition = f.decode.chooser.partition();
        let loads = || async {
            f.decode
                .chooser
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap()
        };

        // In flight on the retiring worker.
        let before = Context::new(request());
        let before_req = routing_request(&before, &partition, None, false);
        let mut in_flight = router.plan(&before_req).unwrap();
        router.schedule(&before_req, &mut in_flight).await.unwrap();
        assert_eq!(in_flight.worker(0).unwrap().worker_id, retiring);

        // Discovery withdraws the worker from the admitted membership; its
        // endpoint instance is still registered.
        f.decode.workers_tx.send_modify(|members| {
            members.remove(&retiring);
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while loads().await.iter().any(|load| load.worker_id == retiring) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the core follows the admitted membership");
        assert!(
            f.decode
                .chooser
                .client()
                .instances()
                .iter()
                .any(|instance| instance.id() == retiring),
            "the endpoint instance itself is still discovered"
        );

        // A new plan is refused: the retired instance is not a placement.
        let after = Context::new(request());
        let after_req = routing_request(&after, &partition, None, false);
        let mut fresh = router.plan(&after_req).unwrap();
        let error = router.schedule(&after_req, &mut fresh).await.unwrap_err();
        assert!(
            matches!(error, SelectionError::NotReady(_)),
            "no admitted worker to schedule on, not a placement: {error}"
        );
        assert!(fresh.booking(0).is_none());

        // The retiring booking is still the plan's to release.
        in_flight.release().await.unwrap();
        assert!(
            loads().await.iter().all(|load| load.active_requests == 0),
            "nothing is charged after the release"
        );
        f.runtime.shutdown();
    }
}
