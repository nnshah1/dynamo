// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill/decode routing with the frontend as the host of a Plan-returning
//! `Router`.
//!
//! A `MultiStageRouter` over one `HostSetRouter` per worker set decides the
//! stage list, the conditional-disagg bypass (preview, decode-busy gate,
//! previewed pin), placement constraints and per-stage accounting. This
//! module is the host: it dispatches what the plan books, observes the
//! prefill handoff and reports it into the plan, and lets the existing
//! `RoutePlan` cleanups, the prefill background task and `forward_decode`
//! own the request's lifecycle, exactly as the legacy path does. Once decode
//! is dispatched the plan is informational: its bookings already live in
//! the route plans' cleanups, so dropping it frees nothing.
//!
//! `DYN_ROUTER_PLAN_HOST=0` falls back to the legacy hand-composed path for
//! one release.
// TODO(v1.7): remove the switch and the legacy KV prefill/decode path.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::conditional_disagg::ConditionalDisaggPolicy;
use dynamo_kv_router::router::{
    ClassTable, Constraint, Failure, MultiStageRouter, Outcome, Plan, Router, StageList, StageState,
};
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    pipeline::{
        AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};

use super::{
    BYPASS_REMOTE_PREFILL_ANNOTATION, PrefillBinding, PrefillCompletion, PrefillOutcome,
    PrefillRouter, PreparedPrefill, build_decode_router_override, extract_bootstrap_info,
    handoff::PrefillTask, independent_prefill_context, into_decode_request,
    strip_terminal_disaggregated_params,
};
use crate::kv_router::RoutingHost;
use crate::kv_router::plan_host::{BusyThresholds, HostSetRouter, frontend_error, routing_request};
use crate::kv_router::routing_host::kv_selection::{
    pinned_worker_hint, resolve_pinned_worker_rank,
};
use crate::protocols::common::{
    extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::{PrefillResult, TraceLink},
    timing::{RequestPhase, RequestTracker},
};
use crate::session_affinity::AffinityTarget;

/// What a [`PlanRouterFactory`] receives: one `Router` per worker set over
/// this request's hosts, and the conditional-disaggregation policy with
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
        let thresholds = BusyThresholds {
            prefill: self.conditional_disagg_prefill_busy_threshold,
            decode: self.conditional_disagg_decode_busy_threshold,
        };
        let prefill_set = Arc::new(HostSetRouter::new(
            Arc::clone(&binding.router),
            WorkerType::Prefill,
            RequestPhase::Prefill,
            &prefill_request,
            conditional,
            thresholds,
        ));
        let decode_set = Arc::new(HostSetRouter::new(
            Arc::clone(&decode_host),
            WorkerType::Decode,
            RequestPhase::Decode,
            &decode_request,
            conditional,
            thresholds,
        ));
        let routing = routing_request(&decode_request, &decode_set.partition(), policy_class);

        let mut flow = PlannedFlow {
            router: self,
            binding: &binding,
            decode_host: &decode_host,
            tracker,
            request_id,
            original_max_tokens,
            prefill_request: Some(prefill_request),
            decode_request: Some(decode_request),
            prefill_set,
            decode_set,
            outcome: None,
            prefill_completion: None,
            dispatched: false,
        };
        match flow.run(&routing, conditional).await {
            Ok(response) => Ok(response),
            // A failed conditional decision falls back to remote prefill, as
            // before, unless the request itself was cancelled or invalid.
            Err(error)
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
                flow.run(&routing, false).await
            }
            Err(error) => Err(error),
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
    prefill_set: Arc<HostSetRouter>,
    decode_set: Arc<HostSetRouter>,
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

impl PlannedFlow<'_> {
    /// The `Router` for this request: the configured factory's, or the
    /// default `MultiStageRouter`, over this request's set routers.
    fn plan_router(&self, conditional: bool) -> Result<Arc<dyn Router>> {
        let parts = PlanRouterParts {
            prefill: Arc::clone(&self.prefill_set) as Arc<dyn Router>,
            decode: Arc::clone(&self.decode_set) as Arc<dyn Router>,
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

    async fn run(
        &mut self,
        routing: &dynamo_kv_router::services::selection::SelectAndReserveRequest,
        conditional: bool,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let multistage = self.plan_router(conditional)?;
        let mut plan = multistage.plan(routing)?;
        self.pin_caller_targets(&mut plan)?;
        let engine_ctx = self
            .decode_request
            .as_ref()
            .ok_or_else(|| anyhow!("decode request already consumed"))?
            .context();
        loop {
            if let Err(error) = multistage.schedule(routing, &mut plan).await {
                return Err(self.routing_error(frontend_error(error)));
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
                WorkerType::Prefill => self.run_prefill(&mut plan, k, &engine_ctx).await?,
                WorkerType::Decode => {
                    self.run_decode(&mut plan, k, conditional, &engine_ctx)
                        .await?
                }
                other => return Err(anyhow!("plan dispatched unexpected set {other:?}")),
            };
            if let Step::Done(response) = step {
                return Ok(response);
            }
        }
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
            for k in 0..plan.stage_count() {
                if plan.stage(k).is_some_and(|stage| stage.set == set) {
                    plan.constrain(k, Constraint::Pin(worker))?;
                }
            }
        }
        Ok(())
    }

    async fn run_prefill(
        &mut self,
        plan: &mut Plan,
        k: usize,
        engine_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Result<Step> {
        let attempt = plan.dispatch(k)?;
        let side = self
            .prefill_set
            .take_side(k)
            .ok_or_else(|| anyhow!("prefill stage has no admitted route"))?;
        let mut prefill_request = self
            .prefill_request
            .take()
            .ok_or_else(|| anyhow!("prefill stage executed twice"))?;
        let phase_barrier = self.tracker.set_phase(RequestPhase::Prefill).await;
        let worker = side.plan.worker();
        let target = AffinityTarget::new(worker.worker_id, Some(worker.dp_rank));
        let prepared = self.router.prepare_prefill_dispatch(
            &mut prefill_request,
            target,
            &self.binding.endpoint_id,
        )?;
        let prefill_stream = match self
            .binding
            .router
            .dispatch_kv_plan(prefill_request, side.plan)
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                plan.fail(k, attempt, failure(&error))?;
                return Err(self.prefill_error(error));
            }
        };
        self.dispatched = true;
        // Decode must now be selected even if the client disconnects: only
        // its worker's KV-transfer path frees what prefill staged.
        self.decode_set.set_staged_kv_cleanup(true);

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
                    plan.fail(k, attempt, failure(&error))?;
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
                    plan.complete(
                        k,
                        attempt,
                        Outcome {
                            worker,
                            kv_hint: None,
                        },
                    )?;
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
        plan.handoff(k, attempt)?;
        Ok(Step::Continue)
    }

    async fn run_decode(
        &mut self,
        plan: &mut Plan,
        k: usize,
        conditional: bool,
        engine_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Result<Step> {
        plan.dispatch(k)?;
        let side = self
            .decode_set
            .take_side(k)
            .ok_or_else(|| anyhow!("decode stage has no admitted route"))?;
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
        let decode_request = if local_prefill {
            let signals = side.signals;
            tracing::info!(
                request_id = %self.request_id,
                worker_id = signals.worker.worker_id,
                dp_rank = signals.worker.dp_rank,
                cached_tokens = signals.cached_tokens,
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
        self.dispatched = true;
        let dispatch = self.decode_host.dispatch_kv_plan(decode_request, side.plan);
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
    use std::collections::{HashMap, VecDeque};
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
    }

    async fn kv_set(
        distributed: &DistributedRuntime,
        namespace: &str,
        role: &'static str,
        worker_config: ModelRuntimeConfig,
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
        let (_workers_tx, workers) = watch::channel(HashMap::from([(worker_id, worker_config)]));
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let chooser = Arc::new(
            KvRouter::new(
                endpoint,
                client.clone(),
                workers,
                None,
                16,
                SelectionPolicySource::Registry,
                Some(config),
                None,
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
        let prefill = kv_set(&distributed, namespace, "prefill", prefill_config).await;
        let decode = kv_set(&distributed, namespace, "decode", decode_config).await;
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
    /// was selected under cannot see decode's requirement, so the host checks
    /// the pair before recording decode and aborts the refused route.
    #[tokio::test]
    async fn an_incompatible_transfer_pair_is_refused_and_nothing_is_kept() {
        use dynamo_kv_router::protocols::KvTransferEnforcement;
        use dynamo_kv_router::router::{ClassTable, MultiStageRouter, StageList};
        use dynamo_kv_router::services::selection::SelectionError;
        use std::collections::HashSet;

        use crate::kv_router::plan_host::{BusyThresholds, HostSetRouter, routing_request};
        use crate::protocols::common::timing::RequestPhase;

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
        let prefill = Arc::new(HostSetRouter::new(
            Arc::clone(&fixture.prefill.host),
            WorkerType::Prefill,
            RequestPhase::Prefill,
            &request,
            false,
            BusyThresholds::default(),
        ));
        let decode = Arc::new(HostSetRouter::new(
            Arc::clone(&fixture.decode.host),
            WorkerType::Decode,
            RequestPhase::Decode,
            &request,
            false,
            BusyThresholds::default(),
        ));
        let req = routing_request(&request, &decode.partition(), None);
        let router = MultiStageRouter::builder()
            .set(WorkerType::Prefill, Arc::clone(&prefill) as Arc<dyn Router>)
            .set(WorkerType::Decode, Arc::clone(&decode) as Arc<dyn Router>)
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
            decode.take_side(1).is_none(),
            "the refused route was not kept"
        );
        // Prefill's admitted route stays with the host for the caller to
        // abort, as any booked stage does when a later one fails.
        prefill
            .take_side(0)
            .expect("prefill route")
            .plan
            .abort()
            .await;
        plan.release().await.unwrap();
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
}
