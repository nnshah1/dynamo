// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `SelectionCore` as the one-set [`Router`]: `schedule` books every
//! schedulable stage of its worker set with a `Lease` admission and hands
//! the armed booking to the plan.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;

use super::super::types::resolve_session_context;
use super::run::session_binding;
use super::*;
use crate::router::{
    Booking, Budget, Constraint, Plan, PlanError, PlanId, Router, Stage, StageWork, WorkerFacts,
};

#[async_trait]
impl Router for SelectionCore {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        SelectionCore::select(self, req).await
    }

    fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        let id = req
            .selection_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut stage = Stage::new(self.worker_type);
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
        loop {
            let next = plan.schedulable().find(|&k| {
                plan.stage(k)
                    .is_some_and(|stage| stage.set == self.worker_type)
            });
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

fn placement_error(error: PlanError) -> SelectionError {
    match error {
        PlanError::Placement { .. } => SelectionError::Conflict(error.to_string()),
        other => SelectionError::BadRequest(other.to_string()),
    }
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
        let routing_constraints = plan
            .placement_constraints(k, &req.routing_constraints)
            .map_err(placement_error)?;
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
        let config = router_config_override.get_or_insert_with(Default::default);
        config.track_prefill_tokens = Some(matches!(
            work,
            StageWork::PrefillAndDecode | StageWork::PrefillOnly
        ));
        match work {
            StageWork::PrefillAndDecode => {}
            StageWork::PrefillOnly => expected_output_tokens = Some(1),
            StageWork::DecodeOnly => config.assume_kv_reuse = Some(false),
            StageWork::None => {}
        }
        // An encoder's booking is admission and lifecycle only: it holds no
        // prompt KV blocks either.
        let track_active_blocks = work != StageWork::None;
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
                        selection_id: booking_id,
                    }
                } else {
                    SelectionAdmission::Lease {
                        request_id: booking_id,
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
        // The worker was in the snapshot the selector read an instant ago;
        // missing now means it just left, and a booking on it is useless.
        // `handle` drops on this path and frees the booking.
        let facts = entry
            .workers_tx
            .borrow()
            .get(&worker.worker_id)
            .map(WorkerFacts::from_config)
            .ok_or_else(|| {
                SelectionError::Internal(format!(
                    "selected worker {} left the partition before it could be booked",
                    worker.worker_id
                ))
            })?;
        plan.check_placement(k, &facts).map_err(placement_error)?;
        plan.book(k, booking, facts, selected.kv_hint)
            .map_err(|error| SelectionError::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;

    use super::super::super::affinity::SessionAffinityMode;
    use super::super::tests::{
        core_with_session_affinity_mode, default_key, local_core, reserve_request, saturated_core,
        test_config, wait_until, worker,
    };
    use super::*;
    use crate::RouterConfigOverride;
    use crate::protocols::KvTransferEnforcement;
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
        let mut plan = Router::plan(&core, &req).expect("plan");
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
        assert_eq!(plan.facts(0).unwrap().topology_value("zone"), Some("a"));
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
        let mut plan = Router::plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan)
            .await
            .expect("schedule");
        let first_worker = plan.worker(0).expect("booked");
        let attempt = plan.dispatch(0).unwrap();
        plan.fail(
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
        let mut held = Router::plan(core, &first).unwrap();
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
        let mut held = Router::plan(core, &first).unwrap();
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
    async fn a_session_binds_on_the_first_booking_and_steers_the_next() {
        let core = core_with_session_affinity_mode(SessionAffinityMode::Hard);
        core.upsert_worker(worker(1)).await.unwrap();
        core.upsert_worker(worker(2)).await.unwrap();
        let entry = core.entry(&default_key()).expect("entry");

        let mut first = reserve_request("s-1");
        first.session_id = Some("session".to_string());
        let mut plan = Router::plan(&core, &first).unwrap();
        Router::schedule(&core, &first, &mut plan)
            .await
            .expect("a session-bearing request books on the Lease path");
        let bound = plan.worker(0).unwrap();
        plan.release().await.unwrap();
        wait_until("release", || !entry.scheduler.has_request("s-1")).await;

        // The second request of the session lands on the bound worker.
        let mut second = reserve_request("s-2");
        second.session_id = Some("session".to_string());
        let mut plan = Router::plan(&core, &second).unwrap();
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
        let mut plan = Router::plan(&core, &req).unwrap();
        Router::schedule(&core, &req, &mut plan).await.unwrap();
        let bound = plan.worker(0).unwrap();
        // Cancelled mid-flight: aborting the plan releases the booking and
        // the session lease it carried; the binding itself survives.
        plan.abort();
        wait_until("abort releases", || !entry.scheduler.has_request("s-1")).await;

        core.delete_worker(bound.worker_id).await.unwrap();
        let mut req = reserve_request("s-2");
        req.session_id = Some("session".to_string());
        let mut plan = Router::plan(&core, &req).unwrap();
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
        let mut held = Router::plan(core, &first).unwrap();
        Router::schedule(core, &first, &mut held).await.unwrap();

        let mut req = reserve_request("now");
        req.all_now = true;
        let mut plan = Router::plan(core, &req).unwrap();
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
        let mut plan = core.plan(&req).unwrap();
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
        let mut plan = core.plan(&req).unwrap();
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
        let mut plan = Router::plan(&core, &req).unwrap();
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
        let mut plan = Router::plan(&core, &reserve_request("here")).unwrap();
        assert!(matches!(
            Router::schedule(&core, &other, &mut plan).await,
            Err(SelectionError::BadRequest(_))
        ));
    }
}
