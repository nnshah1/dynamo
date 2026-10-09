// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A [`Router`] over a bare [`KvRouter`] for a host that drives bookings by
//! id: the gateway EPP.
//!
//! The EPP has no request pipeline, only a `KvRouter` per worker set and a
//! wire whose later callbacks carry an id (`mark_prefill_completed(id)`,
//! `free(id)`). So each stage is booked in *export mode*: the admission is a
//! lease under the stage's booking id, enrolled in the router's request-lease
//! manager exactly as a public `find_best_match_details_with_policy_class`
//! admission is, and the plan records it as `Committed`. Dropping the plan
//! frees nothing; the host frees by the exported id, and a host that never
//! does is reaped on lease expiry.
//!
//! The booking id follows the library's export rule: the plan id for a
//! one-stage first attempt, else `"{plan}/{k}/{attempt}"`.

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::RouterConfigOverride;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::router::{
    Booking, Budget, Constraint, Plan, PlanId, Router, Stage, StageAttempt, StageWork, WorkerFacts,
};
use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_kv_router::services::overlap::MooncakeOverlapSummary;
use dynamo_kv_router::services::selection::{
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionError, SelectionWorkerLoad,
};
use dynamo_runtime::pipeline::{Context, SingleIn, async_trait};

use super::host_error;
use crate::kv_router::{FindBestMatchOutcome, KvRouter};
use crate::protocols::common::{llm_backend::PreprocessedRequest, timing::RequestPhase};

/// A [`Router`] over one worker set's [`KvRouter`], booking in export mode.
pub struct KvRouterSetRouter {
    router: Arc<KvRouter>,
    set: WorkerType,
    phase: RequestPhase,
    /// The request to select with: shares the caller's controller (id,
    /// cancellation) and metadata; each stage renders onto a clone of it.
    base: Context<PreprocessedRequest>,
}

/// The selection arguments a rendered request body carries.
struct SelectionArgs<'a> {
    tokens: &'a [u32],
    block_mm_infos: Option<&'a [Option<BlockExtraInfo>]>,
    router_config_override: Option<&'a RouterConfigOverride>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    priority_jump: f64,
    strict_priority: u32,
    expected_output_tokens: Option<u32>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    routing_constraints: RoutingConstraints,
}

impl<'a> SelectionArgs<'a> {
    fn from_body(body: &'a PreprocessedRequest) -> Self {
        let (tokens, block_mm_infos) = body.block_mm_routing_info();
        let routing = body.routing.as_ref();
        Self {
            tokens,
            block_mm_infos,
            router_config_override: body.router_config_override.as_ref(),
            lora_name: routing.and_then(|hints| hints.lora_name.clone()),
            cache_namespace: routing.and_then(|hints| hints.cache_namespace.clone()),
            priority_jump: routing.and_then(|hints| hints.priority_jump).unwrap_or(0.0),
            strict_priority: routing.and_then(|hints| hints.strict_priority).unwrap_or(0),
            expected_output_tokens: routing.and_then(|hints| hints.expected_output_tokens),
            allowed_worker_ids: routing.and_then(|hints| hints.allowed_worker_ids.clone()),
            routing_constraints: routing
                .and_then(|hints| hints.routing_constraints.clone())
                .unwrap_or_default(),
        }
    }
}

/// The wire id of stage `k`'s booking, as the library's export admission
/// names it: the plan id for a one-stage first attempt, else
/// `"{plan}/{k}/{attempt}"`.
pub fn booking_id(plan: &Plan, k: usize) -> String {
    let attempt = plan.attempt(k).unwrap_or(StageAttempt::FIRST);
    if plan.stage_count() == 1 && attempt == StageAttempt::FIRST {
        plan.id().to_string()
    } else {
        format!("{}/{k}/{attempt}", plan.id())
    }
}

impl KvRouterSetRouter {
    pub fn new(
        router: Arc<KvRouter>,
        set: WorkerType,
        phase: RequestPhase,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Self {
        Self {
            router,
            set,
            phase,
            base: request.fork(request.content().clone()),
        }
    }

    pub fn partition(&self) -> RoutingPartitionId {
        self.router.selection.partition_key().clone()
    }

    fn facts(&self, worker_id: WorkerId) -> WorkerFacts {
        self.router
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

    fn exclude(&self, body: &mut PreprocessedRequest, excluded: &HashSet<WorkerId>) {
        if excluded.is_empty() {
            return;
        }
        let all: HashSet<WorkerId> = self
            .router
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

    /// The worker a rendered body pins for this phase, with its rank resolved
    /// the way the routing host resolves a caller's hint.
    fn pinned_worker(
        &self,
        body: &PreprocessedRequest,
    ) -> Result<Option<WorkerWithDpRank>, SelectionError> {
        let Some(routing) = body.routing.as_ref() else {
            return Ok(None);
        };
        let (worker_id, dp_rank) = match self.phase {
            RequestPhase::Prefill => (
                routing.prefill_worker_id.or(routing.backend_instance_id),
                routing.prefill_dp_rank.or(routing.dp_rank),
            ),
            RequestPhase::Decode | RequestPhase::Aggregated => (
                routing.decode_worker_id.or(routing.backend_instance_id),
                routing.dp_rank,
            ),
        };
        let Some(worker_id) = worker_id else {
            return Ok(None);
        };
        let dp_rank = dp_rank
            .or_else(|| self.router.unique_dp_rank_for_worker(worker_id))
            .ok_or_else(|| {
                SelectionError::BadRequest(format!(
                    "pinned worker {worker_id} has several dp ranks and none was given"
                ))
            })?;
        Ok(Some(WorkerWithDpRank::new(worker_id, dp_rank)))
    }

    /// The selection body for stage `k`: the base request with the stage's
    /// accounting, derived placement constraints, pins and exclusions.
    fn render(
        &self,
        req: &SelectAndReserveRequest,
        plan: &Plan,
        k: usize,
    ) -> Result<PreprocessedRequest, SelectionError> {
        let mut body = self.base.content().clone();
        let work = plan
            .work_of(k)
            .ok_or_else(|| SelectionError::Internal(format!("plan has no stage {k}")))?;
        // Accounting by the stage's work, from the library's one definition.
        // The gateway's decode always routes load-only, as its legacy
        // decode override did.
        let mut config = body.router_config_override.take().unwrap_or_default();
        let mut expected_output_tokens = body
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        work.apply_to(&mut config, &mut expected_output_tokens);
        if let Some(expected) = expected_output_tokens {
            body.routing_mut().expected_output_tokens = Some(expected);
        }
        if work == StageWork::DecodeOnly {
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
        Ok(body)
    }
}

#[async_trait]
impl Router for KvRouterSetRouter {
    /// An advisory pick: the scheduler's choice and that worker's load,
    /// without admission or a booking.
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
        let pinned = self.pinned_worker(&body)?;
        let args = SelectionArgs::from_body(&body);
        let prompt_tokens = args.tokens.len();
        let admitted = self
            .router
            .preview_best_match_details_with_policy_class(
                Some(self.base.id()),
                args.tokens,
                args.block_mm_infos,
                args.router_config_override,
                args.lora_name,
                args.cache_namespace,
                args.priority_jump,
                args.strict_priority,
                None,
                None,
                args.expected_output_tokens,
                pinned,
                args.allowed_worker_ids,
                args.routing_constraints,
            )
            .await
            .map_err(|error| host_error(error, pinned.map(|worker| worker.worker_id)))?;
        let worker_load = admitted.advisory_load.map(|load| SelectionWorkerLoad {
            active_prefill_tokens: load.active_prefill_tokens,
            prefill_token_capacity: load.prefill_token_capacity,
            total_kv_blocks: load.total_kv_blocks.map(|blocks| blocks as u64),
            prefill_busy: None,
        });
        match admitted.outcome {
            FindBestMatchOutcome::Routed {
                worker,
                cached_tokens,
                potential_decode_blocks,
                ..
            } => {
                let key = self.partition();
                let cached = cached_tokens as u32;
                Ok(SelectResponse {
                    selection_id: None,
                    sequence_hashes: None,
                    isl_tokens: Some(prompt_tokens),
                    track_prefill_tokens: None,
                    model_name: key.model_name.clone(),
                    routing_group: key.routing_group.clone(),
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    endpoint: String::new(),
                    block_size: self.router.block_size(),
                    overlap: MooncakeOverlapSummary {
                        longest_matched: cached,
                        gpu: cached,
                        ..MooncakeOverlapSummary::default()
                    },
                    effective_prefill_tokens: prompt_tokens.saturating_sub(cached_tokens),
                    potential_decode_blocks,
                    decode_busy: None,
                    worker_load,
                    kv_hint: None,
                })
            }
            FindBestMatchOutcome::QueueRejected { rejection } => Err(SelectionError::Scheduler(
                KvSchedulerError::QueueRejected(rejection),
            )),
        }
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
            let body = self.render(req, plan, k)?;
            let id = booking_id(plan, k);
            // The stage's wait is the queue's hold budget, as in the core.
            let hold_budget = match plan.stage(k).expect("stage exists").wait {
                Budget::Full => None,
                Budget::Immediate => Some(std::time::Duration::ZERO),
                Budget::Bounded(budget) => Some(budget),
            };
            let pinned = self.pinned_worker(&body)?;
            let args = SelectionArgs::from_body(&body);
            let admitted = self
                .router
                .find_best_match_details_with_policy_class_admitted(
                    Some(&id),
                    args.tokens,
                    args.block_mm_infos,
                    args.router_config_override,
                    true,
                    false,
                    args.lora_name,
                    args.cache_namespace,
                    args.priority_jump,
                    args.strict_priority,
                    req.policy_class.clone(),
                    None,
                    args.expected_output_tokens,
                    pinned,
                    args.allowed_worker_ids,
                    args.routing_constraints,
                    hold_budget,
                )
                .await
                .map_err(|error| host_error(error, pinned.map(|worker| worker.worker_id)))?;
            let (outcome, handle) = admitted.into_parts();
            let (worker, kv_hint) = match outcome {
                FindBestMatchOutcome::Routed {
                    worker, kv_hint, ..
                } => (worker, kv_hint),
                FindBestMatchOutcome::QueueRejected { rejection } => {
                    // Nothing was booked; a handle, if any, frees itself on drop.
                    drop(handle);
                    return Err(SelectionError::Scheduler(KvSchedulerError::QueueRejected(
                        rejection,
                    )));
                }
            };
            let handle = handle.ok_or_else(|| {
                SelectionError::Internal("booked selection returned no booking handle".to_string())
            })?;
            let descriptor = handle.descriptor().clone();
            // Export: the lease manager retains the booking under its id until
            // the host frees it or it expires, as for a public admission.
            self.router
                .enroll_public_request_attempt(handle, None)
                .await?;
            // The chosen worker's own requirement on its earlier peers is the
            // one check the forward constraints it was selected under cannot
            // make; a refused pair frees the exported booking by its id.
            let facts = self.facts(worker.worker_id);
            if let Err(error) = plan
                .check_placement(k, &facts)
                .and_then(|()| plan.book(k, Booking::Committed(descriptor), facts, kv_hint))
            {
                if let Err(free_error) = self.router.free(&id).await {
                    tracing::debug!(booking_id = %id, %free_error, "freeing a booking the plan refused");
                }
                return Err(error.into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_kv_router::router::StageState;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::sync::watch;

    use super::*;
    use crate::kv_router::SelectionPolicySource;
    use crate::kv_router::plan_host::routing_request;
    use crate::local_model::runtime_config::ModelRuntimeConfig;

    /// A KV router over one process-local worker, with no event source.
    async fn kv_router(namespace: &str) -> (Arc<KvRouter>, u64, Runtime) {
        kv_router_with(namespace, ModelRuntimeConfig::default()).await
    }

    async fn kv_router_with(
        namespace: &str,
        worker_config: ModelRuntimeConfig,
    ) -> (Arc<KvRouter>, u64, Runtime) {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace(namespace.to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let workers = HashMap::from([(worker_id, worker_config)]);
        let (_workers_tx, workers) = watch::channel(workers);
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let router = KvRouter::new(
            endpoint,
            client,
            workers,
            None,
            16,
            SelectionPolicySource::Registry,
            Some(config),
            None,
            "decode",
            None,
            false,
            None,
            None,
        )
        .await
        .unwrap();
        (Arc::new(router), worker_id, runtime)
    }

    fn request(tokens: Vec<u32>) -> Context<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test".to_string())
                .token_ids(tokens)
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .unwrap(),
        )
    }

    async fn active_requests(router: &KvRouter, worker_id: u64) -> usize {
        router
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap()
            .iter()
            .find(|load| load.worker_id == worker_id && load.dp_rank == 0)
            .expect("the worker is reported")
            .active_requests
    }

    fn export_request(
        request: &Context<PreprocessedRequest>,
        partition: &RoutingPartitionId,
    ) -> SelectAndReserveRequest {
        let mut req = routing_request(request, partition, None);
        req.export_bookings = true;
        req
    }

    #[tokio::test]
    async fn exported_booking_is_driven_by_id_while_the_plan_is_alive() {
        let (router, worker_id, _runtime) = kv_router("plan-wire-books").await;
        let request = request((1..=40).collect());
        let set_router = KvRouterSetRouter::new(
            Arc::clone(&router),
            WorkerType::Aggregated,
            RequestPhase::Aggregated,
            &request,
        );
        let req = export_request(&request, &set_router.partition());

        let mut plan = set_router.plan(&req).unwrap();
        assert_eq!(plan.stage_count(), 1);
        set_router.schedule(&req, &mut plan).await.unwrap();
        assert_eq!(plan.state_of(0), Some(&StageState::Booked));
        assert_eq!(plan.worker(0).map(|w| w.worker_id), Some(worker_id));
        let booking = plan.booking(0).expect("the stage records its booking");
        assert!(!booking.is_owned(), "an exported booking is the wire's");
        let id = booking.id().to_string();
        assert_eq!(
            id,
            request.id(),
            "a one-stage first attempt exports the plan id"
        );
        assert_eq!(active_requests(&router, worker_id).await, 1);

        // The id-based lifecycle works while the plan is alive.
        router.mark_prefill_completed(&id).await.unwrap();
        router.free(&id).await.unwrap();
        assert!(
            router.free(&id).await.is_err(),
            "a second free finds no booking"
        );
        assert_eq!(active_requests(&router, worker_id).await, 0);

        // The plan releases nothing it does not own.
        plan.release().await.unwrap();
        assert_eq!(active_requests(&router, worker_id).await, 0);
    }

    #[tokio::test]
    async fn dropping_the_plan_keeps_the_exported_booking_for_the_wire() {
        let (router, worker_id, _runtime) = kv_router("plan-wire-drop").await;
        let request = request((1..=40).collect());
        let set_router = KvRouterSetRouter::new(
            Arc::clone(&router),
            WorkerType::Aggregated,
            RequestPhase::Aggregated,
            &request,
        );
        let req = export_request(&request, &set_router.partition());
        let mut plan = set_router.plan(&req).unwrap();
        set_router.schedule(&req, &mut plan).await.unwrap();
        let id = plan.booking(0).unwrap().id().to_string();

        drop(plan);
        assert_eq!(
            active_requests(&router, worker_id).await,
            1,
            "the wire still holds the booking"
        );
        router.free(&id).await.unwrap();
        assert_eq!(active_requests(&router, worker_id).await, 0);
    }

    /// A custom router: short prompts get a one-stage plan of its own
    /// making; everything else delegates to the wrapped router.
    struct ShortPromptRouter(Arc<dyn Router>);

    #[async_trait]
    impl Router for ShortPromptRouter {
        async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
            self.0.select(req).await
        }

        fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
            let prompt_tokens = req.prompt.token_ids.as_ref().map_or(0, Vec::len);
            if prompt_tokens >= 8 {
                return self.0.plan(req);
            }
            Plan::new(
                PlanId::from(req.selection_id.clone().unwrap_or_default()),
                RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone()),
                vec![Stage::new(WorkerType::Aggregated)],
            )
            .map_err(|error| SelectionError::BadRequest(error.to_string()))
        }

        async fn schedule(
            &self,
            req: &SelectAndReserveRequest,
            plan: &mut Plan,
        ) -> Result<(), SelectionError> {
            self.0.schedule(req, plan).await
        }
    }

    #[tokio::test]
    async fn a_custom_router_books_through_the_adapter_unchanged() {
        let (router, worker_id, _runtime) = kv_router("plan-wire-custom").await;
        for tokens in [(1..=4).collect::<Vec<u32>>(), (1..=40).collect()] {
            let request = request(tokens);
            let set_router: Arc<dyn Router> = Arc::new(KvRouterSetRouter::new(
                Arc::clone(&router),
                WorkerType::Aggregated,
                RequestPhase::Aggregated,
                &request,
            ));
            let custom = ShortPromptRouter(set_router);
            let partition = router.selection.partition_key().clone();
            let req = export_request(&request, &partition);

            let mut plan = custom.plan(&req).unwrap();
            custom.schedule(&req, &mut plan).await.unwrap();
            assert_eq!(plan.state_of(0), Some(&StageState::Booked));
            assert_eq!(plan.worker(0).map(|w| w.worker_id), Some(worker_id));
            let id = plan.booking(0).unwrap().id().to_string();
            assert_eq!(id, request.id());
            assert_eq!(active_requests(&router, worker_id).await, 1);

            router.mark_prefill_completed(&id).await.unwrap();
            router.free(&id).await.unwrap();
            assert_eq!(active_requests(&router, worker_id).await, 0);
            plan.release().await.unwrap();
        }
    }

    /// The decode router's worker requires its KV-transfer peers in zone b;
    /// the prefill router's worker publishes no zone. The wire adapter checks
    /// the chosen worker's own requirement before recording it and frees the
    /// exported booking it refused.
    #[tokio::test]
    async fn an_incompatible_transfer_pair_is_refused_and_its_export_freed() {
        use std::collections::HashSet;

        use dynamo_kv_router::protocols::KvTransferEnforcement;
        use dynamo_kv_router::router::{ClassTable, MultiStageRouter, StageList};

        let (prefill_router, prefill_worker, _prefill_runtime) =
            kv_router("plan-wire-placement-prefill").await;
        let (decode_router, decode_worker, _decode_runtime) = kv_router_with(
            "plan-wire-placement-decode",
            ModelRuntimeConfig {
                taints: HashSet::from(["dynamo.topology/zone=b".to_string()]),
                topology_domains: HashMap::from([("zone".to_string(), "b".to_string())]),
                kv_transfer_domain: Some("zone".to_string()),
                kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
                ..ModelRuntimeConfig::default()
            },
        )
        .await;
        let request = request((1..=40).collect());
        let prefill = KvRouterSetRouter::new(
            Arc::clone(&prefill_router),
            WorkerType::Prefill,
            RequestPhase::Prefill,
            &request,
        );
        let decode = KvRouterSetRouter::new(
            Arc::clone(&decode_router),
            WorkerType::Decode,
            RequestPhase::Decode,
            &request,
        );
        let req = export_request(&request, &decode.partition());
        let router = MultiStageRouter::builder()
            .set(WorkerType::Prefill, Arc::new(prefill) as Arc<dyn Router>)
            .set(WorkerType::Decode, Arc::new(decode) as Arc<dyn Router>)
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
        assert_eq!(
            active_requests(&decode_router, decode_worker).await,
            0,
            "the refused decode export was freed"
        );
        // Prefill's exported booking is the host's to free, as for any booked
        // stage when a later one fails.
        let prefill_id = plan.booking(0).expect("prefill booked").id().to_string();
        prefill_router.free(&prefill_id).await.unwrap();
        assert_eq!(active_requests(&prefill_router, prefill_worker).await, 0);
        plan.release().await.unwrap();
    }
}
