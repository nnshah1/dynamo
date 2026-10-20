// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `SelectionCore` as the one-set [`Router`]: `schedule` books every
//! schedulable stage of its worker set with a `Lease` admission and hands
//! the armed booking to the plan.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::super::types::resolve_session_context;
use super::run::session_binding;
use super::*;
use crate::router::{
    Booking, Budget, Constraint, Plan, PlanId, Router, RouterSignals, Stage, WorkerConfig,
};

#[async_trait]
impl Router for SelectionCore {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        SelectionCore::select(self, req).await
    }

    fn signals(&self) -> RouterSignals {
        RouterSignals {
            decode_busy: self
                .kv_router_config
                .conditional_disagg_decode_busy_threshold
                .is_some(),
            prefill_busy: self
                .kv_router_config
                .conditional_disagg_prefill_busy_threshold
                .is_some(),
        }
    }

    fn create_plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        let id = req
            .selection_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut stage = Stage::new(self.set_id());
        if req.all_now {
            stage.wait = Budget::Immediate;
        }
        Plan::new(PlanId::from(id), partition_of(req), vec![stage])
            .map_err(|error| SelectionError::BadRequest(error.to_string()))
    }

    async fn schedule(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
    ) -> Result<(), SelectionError> {
        let key = partition_of(req);
        if plan.partition() != &key {
            return Err(SelectionError::BadRequest(format!(
                "plan {} belongs to {}, request names {key}",
                plan.id(),
                plan.partition()
            )));
        }
        let session_context =
            resolve_session_context(req.session_context.clone(), req.session_id.clone());
        let set = self.set_id();
        loop {
            // This core's stages: the ones that name its set.
            let next = plan
                .schedulable()
                .find(|&k| plan.stage(k).is_some_and(|stage| stage.set == set));
            let Some(k) = next else { break };
            self.book_stage(req, plan, k, session_context.as_ref())
                .await?;
        }
        Ok(())
    }
}

fn partition_of(req: &SelectAndReserveRequest) -> RoutingPartitionId {
    RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone())
}

impl SelectionCore {
    /// One stage's classify → order → place → reserve, with the stage's wait
    /// budget as the queue's hold budget and its constraints folded into the
    /// request's. On any error the stage stays pending and nothing is held.
    async fn book_stage(
        &self,
        req: &SelectAndReserveRequest,
        plan: &mut Plan,
        k: usize,
        session_context: Option<&SessionContext>,
    ) -> Result<(), SelectionError> {
        let no_such_stage = || SelectionError::Internal(format!("plan has no stage {k}"));
        let stage = plan.stage(k).ok_or_else(no_such_stage)?;
        let attempt = plan.attempt(k).ok_or_else(no_such_stage)?;
        let mut pinned_worker = req.pinned_worker;
        let mut excluded = HashSet::new();
        for constraint in &stage.constraints {
            match constraint {
                Constraint::Pin(worker) | Constraint::Previewed(worker) => {
                    if pinned_worker.is_some_and(|pinned| pinned != *worker) {
                        return Err(SelectionError::BadRequest(format!(
                            "stage {k} pins {worker:?} but the request pins {pinned_worker:?}"
                        )));
                    }
                    pinned_worker = Some(*worker);
                }
                Constraint::Exclude(worker_id) => {
                    excluded.insert(*worker_id);
                }
                Constraint::TransferCompatible(_) | Constraint::SameDomain { .. } => {}
            }
        }
        let routing_constraints = plan.placement_constraints(k, &req.routing_constraints)?;
        let hold_budget = match stage.wait {
            Budget::Full => None,
            Budget::Immediate => Some(Duration::ZERO),
            Budget::Bounded(budget) => Some(budget),
        };
        // What this stage's booking is charged for, as the scheduler's
        // existing inputs express it. The stage's work is the per-stage form
        // of the request's `track_prefill_tokens` override and of the core's
        // role default, and replaces both: a decode core configured not to
        // track prefill still pays for a prompt it prefills locally.
        let work = plan.work_of(k).ok_or_else(no_such_stage)?;
        let mut router_config_override = req.router_config_override.clone();
        let mut expected_output_tokens = req.expected_output_tokens;
        work.apply_to(
            router_config_override.get_or_insert_with(Default::default),
            &mut expected_output_tokens,
        );
        // How this stage's candidates are scored: a decode that receives its
        // KV from a prefill stage is placed on load alone.
        plan.scoring_of(k)
            .ok_or_else(no_such_stage)?
            .apply_to(router_config_override.get_or_insert_with(Default::default));
        let track_active_blocks = work.tracks_active_blocks();
        let key = plan.partition().clone();
        let entry = self.ready_entry(&key)?;
        // The exclusion complement is the partition's workers at this instant;
        // a worker joining mid-selection is simply not a candidate.
        let allowed_worker_ids = if excluded.is_empty() {
            req.allowed_worker_ids.clone()
        } else {
            let allowed: HashSet<WorkerId> = match &req.allowed_worker_ids {
                Some(allowed) => allowed.difference(&excluded).copied().collect(),
                None => entry
                    .workers_tx
                    .borrow()
                    .keys()
                    .filter(|worker_id| !excluded.contains(worker_id))
                    .copied()
                    .collect(),
            };
            Some(allowed)
        };
        // The wire's id for the one-stage, first-attempt case; stage and
        // attempt are appended only when they disambiguate.
        let booking_id = if plan.stage_count() == 1 && attempt == crate::router::StageAttempt::FIRST
        {
            plan.id().to_string()
        } else {
            format!("{}/{k}/{attempt}", plan.id())
        };
        let is_steerable = req.affinity_target.is_none() && pinned_worker.is_none();
        let session = session_binding(session_context, is_steerable, true);
        let run = self
            .run_selection(SelectionOperation {
                key,
                prompt: req.prompt.view(),
                router_config_override,
                expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class: req.policy_class.clone(),
                session_context: session_context.cloned(),
                session,
                affinity_target: req.affinity_target,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                admission: if req.export_bookings {
                    SelectionAdmission::Book {
                        selection_id: booking_id.clone(),
                    }
                } else {
                    SelectionAdmission::Lease {
                        request_id: booking_id.clone(),
                    }
                },
                track_active_blocks,
                return_routing_hashes: false,
                replay_id: None,
                hold_budget,
            })
            .await;
        let selected = match run.result? {
            SelectionOutcome::Selected(selected) => selected,
            SelectionOutcome::QueueRejected { rejection } => {
                return Err(SelectionError::Scheduler(KvSchedulerError::QueueRejected(
                    rejection,
                )));
            }
        };
        // An exported booking lives in the core's reservation index under its
        // id, for a host that completes and frees by id; the plan records
        // the descriptor and never releases it. A leased booking is the
        // plan's own.
        let booking = match (
            req.export_bookings,
            selected.booking,
            selected.booking_descriptor,
        ) {
            (true, _, Some(descriptor)) => Booking::Committed(descriptor),
            (false, Some(handle), _) => Booking::Owned(handle),
            (true, _, None) => {
                return Err(SelectionError::Internal(
                    "book admission returned no booking descriptor".to_string(),
                ));
            }
            (false, None, _) => {
                return Err(SelectionError::Internal(
                    "lease admission returned no booking handle".to_string(),
                ));
            }
        };
        let worker = selected.response.best_worker;
        let is_exported = matches!(booking, Booking::Committed(_));
        let recorded = Self::record_stage(
            plan,
            k,
            &entry,
            booking,
            worker,
            selected.kv_hint,
            selected.response,
        );
        // Until the plan holds it, the booking is this call's to roll back. A
        // leased handle freed itself when `booking` dropped above; an
        // exported one sits in the reservation index and is freed here.
        if recorded.is_err()
            && is_exported
            && let Err(error) = self.free_reservation(&booking_id).await
        {
            tracing::warn!(
                booking_id,
                %error,
                "exported booking rejected by the plan could not be freed"
            );
        }
        recorded
    }

    /// Read the booked worker's config, check placement against the stages it
    /// reads, and hand the booking to the plan. Consumes the booking either
    /// way: on an error the caller rolls it back.
    fn record_stage(
        plan: &mut Plan,
        k: usize,
        entry: &SelectionEntry,
        booking: Booking,
        worker: WorkerWithDpRank,
        kv_hint: Option<crate::kv_hints::KvHint>,
        scheduling: crate::scheduling::SchedulingResponse,
    ) -> Result<(), SelectionError> {
        // The worker was in the snapshot the selector read an instant ago;
        // missing now means it just left, and a booking on it is useless.
        let config: WorkerConfig = entry
            .workers_tx
            .borrow()
            .get(&worker.worker_id)
            .map(|config| Arc::new(config.clone()) as WorkerConfig)
            .ok_or_else(|| {
                SelectionError::Internal(format!(
                    "selected worker {} left the partition before it could be booked",
                    worker.worker_id
                ))
            })?;
        plan.check_placement(k, &config)?;
        plan.book(k, booking, config, kv_hint, Some(scheduling))
            .map_err(|error| SelectionError::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;

    use super::super::super::affinity::SessionAffinityMode;
    use super::super::tests::{
        core_with_session_affinity_mode, default_key, hint_capable, hint_fixture, local_core,
        reserve_request, saturated_core, select_request, test_config, wait_until, worker,
    };
    use super::*;
    use crate::RouterConfigOverride;
    use crate::protocols::KvTransferEnforcement;
    use crate::router::StageWork;
    use crate::router::{Failure, StageState, topology_taint};

    fn zoned(worker_id: WorkerId, zone: &str) -> WorkerRequest {
        WorkerRequest {
            taints: HashSet::from([topology_taint("zone", zone)]),
            topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
            ..worker(worker_id)
        }
    }

    fn transfer_zoned(
        worker_id: WorkerId,
        zone: &str,
        enforcement: KvTransferEnforcement,
    ) -> WorkerRequest {
        WorkerRequest {
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(enforcement),
            kv_transfer_preferred_weight: Some(0.5),
            ..zoned(worker_id, zone)
        }
    }

    /// Two stages on this core's own set: the first pinned to `first`, the
    /// second placed against it.
    fn two_stage_plan(id: &str, first: WorkerId, second: Stage) -> Plan {
        Plan::new(
            PlanId::from(id),
            default_key(),
            vec![
                Stage {
                    constraints: vec![Constraint::Pin(WorkerWithDpRank::new(first, 0))],
                    ..Stage::new(WorkerType::Aggregated)
                },
                second,
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn a_one_stage_plan_is_todays_select_and_reserve() {
        let core = local_core(test_config(false));
        core.upsert_worker(zoned(1, "a"))
            .await
            .expect("worker upsert");
        let entry = core.entry(&default_key()).expect("entry");

        let req = reserve_request("sel-1");
        let mut plan = Router::create_plan(&core, &req).expect("plan");
        assert_eq!(plan.stage_count(), 1);
        Router::schedule(&core, &req, &mut plan)
            .await
            .expect("schedule");
        assert_eq!(plan.state_of(0), Some(&StageState::Booked));
        let booking = plan.booking(0).expect("booked");
        assert_eq!(
            booking.id(),
            "sel-1",
            "the booking id is the wire's selection_id"
        );
        assert_eq!(booking.worker().worker_id, 1);
        assert!(booking.is_owned());
        assert!(entry.scheduler.has_request("sel-1"));
        assert_eq!(
            plan.worker_config(0)
                .and_then(|c| c.topology_domains()?.get("zone").cloned()),
            Some("a".to_string())
        );
        assert!(
            core.reservation_index.read().is_empty(),
            "the plan owns the booking; no index row"
        );

        plan.release().await.expect("release");
        wait_until("booking release", || !entry.scheduler.has_request("sel-1")).await;
    }

    #[tokio::test]
    async fn retry_rebooks_on_another_worker_under_a_new_booking_id() {
        let core = local_core(test_config(false));
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        let entry = core.entry(&default_key()).expect("entry");

        let req = reserve_request("sel-2");
        let mut plan = Router::create_plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan)
            .await
            .expect("schedule");
        let first_worker = plan.worker(0).expect("booked");
        let attempt = plan.mark_dispatched(0).unwrap();
        plan.mark_failed(
            0,
            attempt,
            Failure {
                is_retryable: true,
                reason: "worker died".to_string(),
            },
        )
        .unwrap();
        wait_until("failed booking release", || {
            !entry.scheduler.has_request("sel-2")
        })
        .await;
        plan.retry(0).unwrap();

        Router::schedule(&core, &req, &mut plan)
            .await
            .expect("re-schedule");
        let booking = plan.booking(0).expect("re-booked");
        assert_ne!(
            booking.worker(),
            first_worker,
            "the failed worker is excluded"
        );
        assert_eq!(booking.id(), "sel-2/0/1");
        assert!(entry.scheduler.has_request("sel-2/0/1"));
        drop(plan);
        wait_until("drop releases", || {
            !entry.scheduler.has_request("sel-2/0/1")
        })
        .await;
    }

    #[tokio::test]
    async fn an_immediate_budget_rejects_instead_of_holding() {
        let core = saturated_core();
        let core: &SelectionCore = &core;
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let entry = core.entry(&default_key()).expect("entry");

        let first = reserve_request("first");
        let mut held = Router::create_plan(core, &first).unwrap();
        Router::schedule(core, &first, &mut held)
            .await
            .expect("first booking");
        assert!(entry.scheduler.has_request("first"));

        let mut immediate = Plan::new(
            PlanId::from("second"),
            default_key(),
            vec![Stage {
                wait: Budget::Immediate,
                ..Stage::new(WorkerType::Aggregated)
            }],
        )
        .unwrap();
        let error = Router::schedule(core, &reserve_request("second"), &mut immediate)
            .await
            .expect_err("the queue would hold this request");
        assert!(
            matches!(
                error,
                SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
            ),
            "{error}"
        );
        assert!(!entry.scheduler.has_request("second"));
        assert!(
            immediate.state_of(0).unwrap().is_pending(),
            "the host keeps the plan"
        );

        held.release().await.expect("release");
        wait_until("release", || !entry.scheduler.has_request("first")).await;
    }

    #[tokio::test]
    async fn a_bounded_budget_expires_while_parked() {
        let core = saturated_core();
        let core: &SelectionCore = &core;
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let first = reserve_request("first");
        let mut held = Router::create_plan(core, &first).unwrap();
        Router::schedule(core, &first, &mut held).await.unwrap();

        let mut bounded = Plan::new(
            PlanId::from("third"),
            default_key(),
            vec![Stage {
                wait: Budget::Bounded(Duration::from_millis(50)),
                ..Stage::new(WorkerType::Aggregated)
            }],
        )
        .unwrap();
        let error = Router::schedule(core, &reserve_request("third"), &mut bounded)
            .await
            .expect_err("the hold budget expires");
        assert!(
            matches!(
                error,
                SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
            ),
            "{error}"
        );
        held.abort();
    }

    #[tokio::test]
    async fn derived_constraints_place_the_second_stage_and_an_error_keeps_the_first() {
        let core = local_core(test_config(false));
        core.upsert_worker(transfer_zoned(1, "a", KvTransferEnforcement::Required))
            .await
            .unwrap();
        core.upsert_worker(zoned(2, "b")).await.unwrap();
        core.upsert_worker(zoned(3, "a")).await.unwrap();
        let entry = core.entry(&default_key()).expect("entry");

        let req = reserve_request("pair");
        let mut plan = two_stage_plan(
            "pair",
            1,
            Stage {
                constraints: vec![Constraint::TransferCompatible(0), Constraint::Exclude(1)],
                ..Stage::new(WorkerType::Aggregated)
            },
        );
        Router::schedule(&core, &req, &mut plan)
            .await
            .expect("schedule");
        assert_eq!(plan.worker(0).unwrap().worker_id, 1);
        assert_eq!(
            plan.worker(1).unwrap().worker_id,
            3,
            "the only zone-a worker other than the excluded one"
        );
        plan.abort();
        wait_until("release", || !entry.scheduler.has_request("pair/1/0")).await;

        // No zone-a peer left: stage 1 cannot be placed; stage 0 stays booked.
        core.delete_worker(3).await.unwrap();
        let req = reserve_request("pair-2");
        let mut plan = two_stage_plan(
            "pair-2",
            1,
            Stage {
                constraints: vec![Constraint::TransferCompatible(0), Constraint::Exclude(1)],
                ..Stage::new(WorkerType::Aggregated)
            },
        );
        let error = Router::schedule(&core, &req, &mut plan)
            .await
            .expect_err("no compatible worker");
        assert!(matches!(error, SelectionError::Scheduler(_)), "{error}");
        assert_eq!(plan.state_of(0), Some(&StageState::Booked));
        assert!(plan.state_of(1).unwrap().is_pending());
        assert!(entry.scheduler.has_request("pair-2/0/0"));
        assert!(!entry.scheduler.has_request("pair-2/1/0"));
        plan.abort();
    }

    #[tokio::test]
    async fn a_later_worker_requiring_its_own_transfer_domain_is_checked_against_the_earlier() {
        let core = local_core(test_config(false));
        core.upsert_worker(zoned(2, "b")).await.unwrap();
        core.upsert_worker(transfer_zoned(1, "a", KvTransferEnforcement::Required))
            .await
            .unwrap();
        let entry = core.entry(&default_key()).expect("entry");

        // Stage 0 lands on the zone-b worker (no transfer policy of its own);
        // stage 1 may only pick worker 1, whose policy requires zone-a peers.
        let req = reserve_request("rev");
        let mut plan = two_stage_plan(
            "rev",
            2,
            Stage {
                constraints: vec![Constraint::TransferCompatible(0), Constraint::Exclude(2)],
                ..Stage::new(WorkerType::Aggregated)
            },
        );
        let error = Router::schedule(&core, &req, &mut plan)
            .await
            .expect_err("the pair cannot transfer KV");
        assert!(matches!(error, SelectionError::Conflict(_)), "{error}");
        assert!(plan.state_of(1).unwrap().is_pending());
        wait_until("rejected booking release", || {
            !entry.scheduler.has_request("rev/1/0")
        })
        .await;
        assert!(entry.scheduler.has_request("rev/0/0"));
        plan.abort();
    }

    #[tokio::test]
    async fn an_exported_booking_the_plan_rejects_is_freed_from_the_index() {
        let core = local_core(test_config(false));
        core.upsert_worker(zoned(2, "b")).await.unwrap();
        core.upsert_worker(transfer_zoned(1, "a", KvTransferEnforcement::Required))
            .await
            .unwrap();
        let entry = core.entry(&default_key()).expect("entry");

        let mut req = reserve_request("rev-export");
        req.export_bookings = true;
        let mut plan = two_stage_plan(
            "rev-export",
            2,
            Stage {
                constraints: vec![Constraint::TransferCompatible(0), Constraint::Exclude(2)],
                ..Stage::new(WorkerType::Aggregated)
            },
        );
        let error = Router::schedule(&core, &req, &mut plan)
            .await
            .expect_err("the pair cannot transfer KV");
        assert!(matches!(error, SelectionError::Conflict(_)), "{error}");
        assert!(plan.state_of(1).unwrap().is_pending());
        assert!(plan.booking(1).is_none());
        // The rejected stage's exported reservation is gone, not orphaned.
        wait_until("rejected export release", || {
            !entry.scheduler.has_request("rev-export/1/0")
        })
        .await;
        assert!(
            core.free_reservation("rev-export/1/0").await.is_err(),
            "nothing left to free by id"
        );
        wait_until("no load on the rejected worker", || {
            core.loads(None, None).iter().all(|model| {
                model
                    .loads
                    .iter()
                    .filter(|load| load.worker_id == 1)
                    .all(|load| load.active_requests == 0)
            })
        })
        .await;
        // The earlier stage keeps its exported booking under the host's id.
        assert!(entry.scheduler.has_request("rev-export/0/0"));
        core.free_reservation("rev-export/0/0").await.unwrap();
        plan.release().await.unwrap();
    }

    #[tokio::test]
    async fn a_session_binds_on_the_first_booking_and_steers_the_next() {
        let core = core_with_session_affinity_mode(SessionAffinityMode::Hard);
        core.upsert_worker(worker(1)).await.unwrap();
        core.upsert_worker(worker(2)).await.unwrap();
        let entry = core.entry(&default_key()).expect("entry");

        let mut first = reserve_request("s-1");
        first.session_id = Some("session".to_string());
        let mut plan = Router::create_plan(&core, &first).unwrap();
        Router::schedule(&core, &first, &mut plan)
            .await
            .expect("a session-bearing request books on the Lease path");
        let bound = plan.worker(0).unwrap();
        plan.release().await.unwrap();
        wait_until("release", || !entry.scheduler.has_request("s-1")).await;

        // The second request of the session lands on the bound worker.
        let mut second = reserve_request("s-2");
        second.session_id = Some("session".to_string());
        let mut plan = Router::create_plan(&core, &second).unwrap();
        Router::schedule(&core, &second, &mut plan).await.unwrap();
        assert_eq!(plan.worker(0), Some(bound));
        plan.abort();
    }

    #[tokio::test]
    async fn a_session_rebinds_when_its_worker_leaves_and_releases_with_the_plan() {
        let core = core_with_session_affinity_mode(SessionAffinityMode::Hard);
        core.upsert_worker(worker(1)).await.unwrap();
        core.upsert_worker(worker(2)).await.unwrap();
        let entry = core.entry(&default_key()).expect("entry");

        let mut req = reserve_request("s-1");
        req.session_id = Some("session".to_string());
        let mut plan = Router::create_plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        let bound = plan.worker(0).unwrap();
        // Cancelled mid-flight: aborting the plan releases the booking and
        // the session lease it carried; the binding itself survives.
        plan.abort();
        wait_until("abort releases", || !entry.scheduler.has_request("s-1")).await;

        core.delete_worker(bound.worker_id).await.unwrap();
        let mut req = reserve_request("s-2");
        req.session_id = Some("session".to_string());
        let mut plan = Router::create_plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan)
            .await
            .expect("a departed binding is re-initialised, not an error");
        assert_ne!(plan.worker(0).unwrap(), bound);
        plan.release().await.unwrap();
    }

    #[tokio::test]
    async fn all_now_makes_the_one_stage_plan_immediate() {
        let core = saturated_core();
        let core: &SelectionCore = &core;
        core.upsert_worker(worker(1)).await.unwrap();
        let entry = core.entry(&default_key()).expect("entry");
        let first = reserve_request("held");
        let mut held = Router::create_plan(core, &first).unwrap();
        Router::schedule(core, &first, &mut held).await.unwrap();

        let mut req = reserve_request("now");
        req.all_now = true;
        let mut plan = Router::create_plan(core, &req).unwrap();
        assert_eq!(plan.stage(0).unwrap().wait, Budget::Immediate);
        let error = Router::schedule(core, &req, &mut plan)
            .await
            .expect_err("cannot park");
        assert!(
            matches!(
                error,
                SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded)
            ),
            "{error}"
        );
        assert!(!entry.scheduler.has_request("now"));
        held.abort();
    }

    #[tokio::test]
    async fn exported_bookings_live_in_the_reservation_index() {
        let core = local_core(test_config(false));
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let entry = core.entry(&default_key()).expect("entry");
        let mut req = reserve_request("exported");
        req.export_bookings = true;
        let mut plan = core.create_plan(&req).unwrap();
        core.schedule(&req, &mut plan).await.unwrap();
        let booking = plan.booking(0).expect("booked");
        assert_eq!(
            booking.id(),
            "exported",
            "a one-stage first attempt keeps the plan id"
        );
        assert!(!booking.is_owned(), "the reservation index owns it");
        assert!(entry.scheduler.has_request("exported"));
        // The wire lifecycle works while the plan is alive ...
        core.prefill_complete("exported").await.expect("indexed");
        core.free_reservation("exported").await.expect("indexed");
        // ... and releasing the plan afterwards is a no-op, not a double free.
        plan.release().await.unwrap();
        wait_until("the exported booking is gone", || {
            core.loads(None, None)
                .iter()
                .all(|model| model.loads.iter().all(|load| load.active_requests == 0))
        })
        .await;

        // A leased booking is not indexed.
        let req = reserve_request("leased");
        let mut plan = core.create_plan(&req).unwrap();
        core.schedule(&req, &mut plan).await.unwrap();
        assert!(plan.booking(0).unwrap().is_owned());
        assert!(core.free_reservation("leased").await.is_err());
        plan.release().await.unwrap();
    }

    #[tokio::test]
    async fn stage_work_decides_what_the_booking_is_charged_for() {
        // A decode core whose role default tracks no prefill: the stage's
        // work, not the role, decides what each booking pays for.
        let mut config = test_config(false);
        config.router_track_prefill_tokens = false;
        let core = super::super::tests::core_with(
            config,
            SelectionHost::default(),
            None,
            WorkerType::Decode,
            None,
        );
        core.upsert_worker(worker(1)).await.unwrap();
        let load = || {
            core.loads(Some("model"), Some("default"))
                .first()
                .and_then(|model| model.loads.first().cloned())
                .expect("worker load")
        };
        wait_until("slot tracker sees the worker", || {
            core.loads(Some("model"), Some("default"))
                .first()
                .is_some_and(|model| !model.loads.is_empty())
        })
        .await;
        let one_stage = |id: &str, work: StageWork| {
            Plan::new(
                PlanId::from(id),
                default_key(),
                vec![Stage {
                    work: Some(work),
                    ..Stage::new(WorkerType::Decode)
                }],
            )
            .unwrap()
        };

        let req = reserve_request("decode-only");
        let mut plan = one_stage("decode-only", StageWork::DecodeOnly);
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        assert_eq!(
            load().potential_prefill_tokens,
            0,
            "a remote-prefill decode is not charged for the prompt"
        );
        assert_eq!(load().potential_decode_blocks, 1, "but holds its KV blocks");
        plan.abort();
        wait_until("release", || load().active_requests == 0).await;

        let req = reserve_request("local");
        let mut plan = one_stage("local", StageWork::PrefillAndDecode);
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        assert_eq!(
            load().potential_prefill_tokens,
            4,
            "a local-prefill decode is charged for the prompt, role default notwithstanding"
        );
        plan.abort();
        wait_until("release", || load().active_requests == 0).await;

        let req = reserve_request("none");
        let mut plan = one_stage("none", StageWork::None);
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        let none = load();
        assert_eq!(none.potential_prefill_tokens, 0);
        assert_eq!(
            none.potential_decode_blocks, 0,
            "no prompt KV blocks either"
        );
        assert_eq!(none.active_requests, 1, "the booking itself is live");
        plan.abort();
    }

    #[tokio::test]
    async fn the_stages_work_outranks_the_requests_tracking_override() {
        let core = local_core(test_config(false));
        core.upsert_worker(worker(1)).await.unwrap();
        wait_until("slot tracker sees the worker", || {
            core.loads(Some("model"), Some("default"))
                .first()
                .is_some_and(|model| !model.loads.is_empty())
        })
        .await;
        let mut req = reserve_request("override");
        req.router_config_override = Some(RouterConfigOverride {
            track_prefill_tokens: Some(false),
            ..Default::default()
        });
        let mut plan = Router::create_plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        let load = core.loads(Some("model"), Some("default"))[0].loads[0].potential_prefill_tokens;
        assert_eq!(
            load, 4,
            "an aggregated stage prefills the prompt whatever the request's override says"
        );
        plan.abort();
    }

    #[tokio::test]
    async fn request_and_plan_must_agree_on_pins_and_partition() {
        let core = local_core(test_config(false));
        core.upsert_worker(worker(1)).await.unwrap();
        core.upsert_worker(worker(2)).await.unwrap();

        let mut req = reserve_request("pin");
        req.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        let mut plan = Plan::new(
            PlanId::from("pin"),
            default_key(),
            vec![Stage {
                constraints: vec![Constraint::Pin(WorkerWithDpRank::new(2, 0))],
                ..Stage::new(WorkerType::Aggregated)
            }],
        )
        .unwrap();
        assert!(matches!(
            Router::schedule(&core, &req, &mut plan).await,
            Err(SelectionError::BadRequest(_))
        ));

        let mut other = reserve_request("elsewhere");
        other.routing_group = "other".to_string();
        let mut plan = Router::create_plan(&core, &reserve_request("here")).unwrap();
        assert!(matches!(
            Router::schedule(&core, &other, &mut plan).await,
            Err(SelectionError::BadRequest(_))
        ));
    }

    /// The hint the scheduler attaches to a booking (here: fetch the prefix
    /// worker 1 holds) is recorded on the plan's stage, where the host reads
    /// it to dispatch.
    #[tokio::test]
    async fn a_booked_stage_records_the_schedulers_kv_hint() {
        let (core, _entry, tokens) = hint_fixture(hint_capable).await;
        let mut req = reserve_request("hinted");
        req.prompt.token_ids = Some(tokens.clone());
        req.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        let mut plan = Router::create_plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        let hint = plan
            .kv_hint(0)
            .expect("the booking on worker 2 carries a hint");
        assert_eq!(
            hint.message_id, "hinted",
            "the hint is addressed by the booking id"
        );
        assert_eq!(hint.actions[0].action_type, "kv.fetch");
        // Worker 1 holds the prefix itself: nothing to fetch, no hint.
        let mut req = reserve_request("unhinted");
        req.prompt.token_ids = Some(tokens);
        req.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        let mut other = Router::create_plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut other).await.unwrap();
        assert!(other.kv_hint(0).is_none());
        plan.release().await.unwrap();
        other.release().await.unwrap();
    }

    /// Worker 2 holds the whole prompt and one active request; worker 1 is
    /// idle with nothing cached. A stage that decodes after remote prefill
    /// is placed on load alone, so it goes to worker 1 despite worker 2's
    /// cache; the same stage told to score cache-aware goes to worker 2, as
    /// a plain advisory selection does.
    #[tokio::test]
    async fn a_decode_only_stage_is_placed_on_load_not_cache() {
        use crate::router::StageScoring;
        let core = local_core(test_config(false));
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        let tokens: Vec<u32> = (1..=64).collect();

        // Book the prompt on worker 2: the approximate primary records its
        // prefix there, and the booking is its load.
        let mut held = reserve_request("held");
        held.prompt.token_ids = Some(tokens.clone());
        held.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        core.select_and_reserve(held).await.expect("reserve");
        let advisory = || {
            let mut req = select_request();
            req.prompt.token_ids = Some(tokens.clone());
            req.advisory = true;
            req
        };
        let baseline = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let response = core.select(advisory()).await.expect("select");
                if response.overlap.longest_matched > 0 {
                    return response;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the booked prompt is credited");
        assert_eq!(baseline.worker_id, 2, "cache-aware: the cached worker wins");

        let stage = |scoring: Option<StageScoring>| {
            Plan::new(
                PlanId::from(match scoring {
                    None => "load-only",
                    Some(_) => "cache-aware",
                }),
                default_key(),
                vec![Stage {
                    work: Some(StageWork::DecodeOnly),
                    scoring,
                    ..Stage::new(WorkerType::Aggregated)
                }],
            )
            .unwrap()
        };
        let mut req = reserve_request("load-only");
        req.prompt.token_ids = Some(tokens.clone());
        let mut plan = stage(None);
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        assert_eq!(
            plan.worker(0).unwrap().worker_id,
            1,
            "decode after remote prefill ignores the cache and takes the idle worker"
        );

        let mut req = reserve_request("cache-aware");
        req.prompt.token_ids = Some(tokens.clone());
        let mut other = stage(Some(StageScoring::CacheAware));
        Router::schedule(&core, &req, &mut other).await.unwrap();
        assert_eq!(
            other.worker(0).unwrap().worker_id,
            2,
            "the stage's own scoring puts it back on the cached worker"
        );
        plan.release().await.unwrap();
        other.release().await.unwrap();
        core.free_reservation("held").await.unwrap();
    }

    /// A core named as one worker set of its role books only the stages
    /// that name that set; the other set's stage is left for its own core,
    /// and the core's one-stage plan names its set.
    #[tokio::test]
    async fn a_named_core_books_only_its_own_sets_stages() {
        let core = local_core(test_config(false));
        core.name_set("canary").unwrap();
        assert!(core.name_set("main").is_err(), "a core is one set");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let req = reserve_request("pooled");
        let own = Router::create_plan(&core, &req).unwrap();
        assert_eq!(own.stage(0).unwrap().set.name.as_deref(), Some("canary"));

        let mut plan = Plan::new(
            PlanId::from("pooled"),
            default_key(),
            vec![
                Stage::new((WorkerType::Aggregated, "canary")),
                Stage::new(WorkerType::Aggregated),
            ],
        )
        .unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        assert_eq!(plan.state_of(0), Some(&StageState::Booked));
        assert!(
            plan.state_of(1).unwrap().is_pending(),
            "the unnamed set's stage is not this core's"
        );
        plan.release().await.unwrap();
    }

    /// Precedence through a real admission: a request that asks for full
    /// cache credit still has its decode-only stage placed on load alone
    /// (the stage list wins over the request's override), and the rule holds
    /// across a retry after that worker fails.
    #[tokio::test]
    async fn a_stages_scoring_wins_over_the_requests_override_and_survives_retry() {
        use crate::router::Failure;
        let core = local_core(test_config(false));
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        // Worker 2 holds a 96-token request past its prefill; the 64-token
        // request below is a prefix of it. With the default weights that
        // makes both choices strict: cache-aware prefers 2 (24 vs 32 blocks),
        // load-only prefers an idle worker (32 vs 40).
        let held_tokens: Vec<u32> = (1..=96).collect();
        let tokens: Vec<u32> = (1..=64).collect();
        let mut held = reserve_request("held");
        held.prompt.token_ids = Some(held_tokens);
        held.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        core.select_and_reserve(held).await.expect("reserve");
        core.prefill_complete("held")
            .await
            .expect("held is decoding");
        let cached_on_2 = || {
            let mut req = select_request();
            req.prompt.token_ids = Some(tokens.clone());
            req.advisory = true;
            req
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            while core
                .select(cached_on_2())
                .await
                .expect("select")
                .overlap
                .longest_matched
                == 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the booked prompt is credited");

        let mut req = reserve_request("override");
        req.prompt.token_ids = Some(tokens.clone());
        req.router_config_override = Some(RouterConfigOverride {
            overlap_score_credit: Some(1.0),
            ..RouterConfigOverride::default()
        });
        let mut plan = Plan::new(
            PlanId::from("override"),
            default_key(),
            vec![Stage {
                work: Some(StageWork::DecodeOnly),
                ..Stage::new(WorkerType::Aggregated)
            }],
        )
        .unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        assert_eq!(
            plan.worker(0).unwrap().worker_id,
            1,
            "the stage's load-only scoring beats the request's full cache credit"
        );

        // Worker 1 fails. A third, idle worker joins, so the retry has two
        // eligible candidates: 2 (holds the cached prefix) and 3 (idle). A
        // caller-only request with the same eligibility and the same full
        // cache credit picks the cached worker; the retried stage still
        // places on load alone and picks the idle one.
        let attempt = plan.mark_dispatched(0).unwrap();
        plan.mark_failed(
            0,
            attempt,
            Failure {
                is_retryable: true,
                reason: "worker died".to_string(),
            },
        )
        .unwrap();
        core.upsert_worker(worker(3)).await.expect("worker upsert");
        wait_until("the third worker is visible", || {
            core.loads(None, None)
                .iter()
                .any(|model| model.loads.iter().any(|load| load.worker_id == 3))
        })
        .await;
        let mut control = cached_on_2();
        control.router_config_override = req.router_config_override.clone();
        control.allowed_worker_ids = Some(HashSet::from([2, 3]));
        let caller_only = core.select(control).await.unwrap().worker_id;
        plan.retry(0).unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        let retried = plan.worker(0).unwrap().worker_id;
        plan.release().await.unwrap();
        core.free_reservation("held").await.unwrap();
        assert_eq!(
            caller_only, 2,
            "the caller-only control prefers the cached worker"
        );
        assert_eq!(
            retried, 3,
            "the retried stage keeps load-only scoring with two eligible workers"
        );
    }
}
