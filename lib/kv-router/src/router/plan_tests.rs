// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `Plan` state machine over scripted bookings: booking versus execution
//! order, fail cascading, retry fencing, exactly-once release.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::booking::Booking;
use super::plan::{
    Budget, Constraint, Failure, Outcome, Plan, PlanError, PlanId, PlanState, Retry, SkipRule,
    Stage, StageAttempt, StageState, StageWork, When, WorkerFacts,
};
use crate::WorkerType;
use crate::identity::RoutingPartitionId;
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::AttemptId;
use crate::scheduling::queue::SchedulerBookingDescriptor;

fn worker(id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(id, 0)
}

fn booking(id: &str, w: WorkerWithDpRank, releases: &Arc<AtomicUsize>) -> Booking {
    Booking::scripted(id, w, Arc::clone(releases))
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

fn zone(name: &str) -> WorkerFacts {
    WorkerFacts {
        kv_transfer_domain: Some("zone".to_string()),
        topology_domains: HashMap::from([("zone".to_string(), name.to_string())]),
        ..WorkerFacts::default()
    }
}

fn new_plan(stages: Vec<Stage>) -> Result<Plan, PlanError> {
    let partition = RoutingPartitionId::new("model", "default");
    Plan::new(PlanId::from("plan-1"), partition, stages)
}

fn plan(stages: Vec<Stage>) -> Plan {
    new_plan(stages).expect("valid stage list")
}

/// Prefill then decode, decode reading prefill's worker and outcome.
fn prefill_decode(when: When, wait: Budget) -> Vec<Stage> {
    vec![
        Stage::new(WorkerType::Prefill),
        Stage {
            when,
            wait,
            inputs: vec![0],
            constraints: vec![Constraint::TransferCompatible(0)],
            ..Stage::new(WorkerType::Decode)
        },
    ]
}

/// Decode booked first, prefill placed against it, prefill forwarded first.
fn decode_first() -> Vec<Stage> {
    vec![
        Stage {
            inputs: vec![1],
            ..Stage::new(WorkerType::Decode)
        },
        Stage {
            constraints: vec![Constraint::TransferCompatible(0)],
            ..Stage::new(WorkerType::Prefill)
        },
    ]
}

fn state(plan: &Plan, k: usize) -> &StageState {
    plan.state_of(k).unwrap()
}

fn vec_of(iter: impl Iterator<Item = usize>) -> Vec<usize> {
    iter.collect()
}

#[test]
fn construction_rejects_forward_booking_edges_cycles_and_empty_lists() {
    assert_eq!(new_plan(Vec::new()).err(), Some(PlanError::Empty));
    let forward_when = vec![
        Stage {
            when: When::After(1),
            ..Stage::new(WorkerType::Prefill)
        },
        Stage::new(WorkerType::Decode),
    ];
    assert_eq!(
        new_plan(forward_when).err(),
        Some(PlanError::ForwardDependency {
            stage: 0,
            depends_on: 1
        })
    );
    let self_constraint = vec![Stage {
        constraints: vec![Constraint::TransferCompatible(0)],
        ..Stage::new(WorkerType::Prefill)
    }];
    assert_eq!(
        new_plan(self_constraint).err(),
        Some(PlanError::ForwardDependency {
            stage: 0,
            depends_on: 0
        })
    );
    let cycle = vec![
        Stage {
            inputs: vec![1],
            ..Stage::new(WorkerType::Prefill)
        },
        Stage {
            inputs: vec![0],
            ..Stage::new(WorkerType::Decode)
        },
    ];
    assert!(matches!(
        new_plan(cycle).err(),
        Some(PlanError::DependencyCycle { .. })
    ));
    let dangling = vec![Stage {
        inputs: vec![3],
        ..Stage::new(WorkerType::Prefill)
    }];
    assert_eq!(
        new_plan(dangling).err(),
        Some(PlanError::NoSuchStage { stage: 3 })
    );
    // `After(k)` implies `inputs: [k]`.
    let p = plan(prefill_decode(When::After(0), Budget::Full));
    assert_eq!(p.stage(1).unwrap().inputs, vec![0]);
}

#[test]
fn a_deferred_stage_is_booked_only_after_its_input_completes() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(prefill_decode(
        When::After(0),
        Budget::Bounded(Duration::from_secs(2)),
    ));
    assert_eq!(p.state(), PlanState::Planned);
    assert_eq!(vec_of(p.schedulable()), vec![0]);

    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(p.state(), PlanState::Booked);
    assert_eq!(vec_of(p.ready()), vec![0]);
    assert_eq!(vec_of(p.schedulable()), Vec::<usize>::new());
    assert_eq!(
        p.book(1, booking("early", worker(21), &releases), zone("a"), None)
            .err(),
        Some(PlanError::InputsNotReady { stage: 1 })
    );

    let attempt = p.dispatch(0).unwrap();
    assert_eq!(p.state(), PlanState::Dispatched);
    assert_eq!(vec_of(p.ready()), Vec::<usize>::new());
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    assert_eq!(p.facts(0).unwrap().topology_value("zone"), Some("a"));

    p.book(1, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(vec_of(p.ready()), vec![1]);
    assert!(!p.has_pending());
    p.abort();
    assert_eq!(
        releases.load(Ordering::SeqCst),
        3,
        "p, d, and the rejected early booking"
    );
}

#[test]
fn booking_order_and_execution_order_are_independent() {
    let releases = Arc::new(AtomicUsize::new(0));
    // Booked together, forwarded prefill then decode.
    let mut p = plan(prefill_decode(When::Now, Budget::Immediate));
    assert_eq!(
        vec_of(p.schedulable()),
        vec![0],
        "decode reads prefill's worker"
    );
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    p.book(1, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(vec_of(p.ready()), vec![0]);
    assert_eq!(
        p.dispatch(1).err(),
        Some(PlanError::InputsNotReady { stage: 1 }),
        "decode waits for prefill's outcome"
    );
    let attempt = p.dispatch(0).unwrap();
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    assert_eq!(vec_of(p.ready()), vec![1]);

    // Decode first: booked first, forwarded last.
    let mut p = plan(decode_first());
    assert_eq!(vec_of(p.schedulable()), vec![0]);
    p.book(0, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    assert_eq!(
        vec_of(p.ready()),
        Vec::<usize>::new(),
        "decode waits for prefill"
    );
    p.book(1, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(vec_of(p.ready()), vec![1]);
    let attempt = p.dispatch(1).unwrap();
    p.complete(1, attempt, outcome(worker(11))).unwrap();
    assert_eq!(vec_of(p.ready()), vec![0]);

    // A skipped input counts as settled.
    let mut p = plan(vec![
        Stage {
            skip: Some(SkipRule::NoMultimodal),
            ..Stage::new(WorkerType::Encode)
        },
        Stage {
            when: When::After(0),
            ..Stage::new(WorkerType::Prefill)
        },
    ]);
    p.skip(0).unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    p.book(1, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(vec_of(p.ready()), vec![1]);
}

#[test]
fn fail_releases_the_stage_and_every_booked_stage_that_reads_it() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(vec![
        Stage::new(WorkerType::Encode),
        Stage {
            constraints: vec![Constraint::SameDomain {
                stage: 0,
                key: "zone".to_string(),
                mode: super::plan::DomainMode::Required,
            }],
            ..Stage::new(WorkerType::Prefill)
        },
        Stage {
            constraints: vec![Constraint::TransferCompatible(1)],
            ..Stage::new(WorkerType::Decode)
        },
    ]);
    p.book(0, booking("e", worker(31), &releases), zone("a"), None)
        .unwrap();
    p.book(1, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    p.book(2, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    let attempt = p.dispatch(0).unwrap();

    p.fail(0, attempt, failure(true)).unwrap();
    assert_eq!(
        releases.load(Ordering::SeqCst),
        3,
        "the stage and both transitive dependents"
    );
    assert!(matches!(state(&p, 0), StageState::Failed(a, _) if *a == attempt));
    assert!(state(&p, 1).is_pending());
    assert!(
        state(&p, 2).is_pending(),
        "stage 2 read stage 1's old placement"
    );
    assert_eq!(
        p.attempt(2),
        Some(StageAttempt::FIRST.next()),
        "a re-placed dependent is a new attempt"
    );
    assert_eq!(p.failures(2), Some(0), "but not a failure of its own");
    assert_eq!(p.state(), PlanState::Planned);

    // A dispatched dependent is in flight and is kept.
    let mut p = plan(decode_first());
    p.book(0, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    p.book(1, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    let prefill = p.dispatch(1).unwrap();
    p.complete(1, prefill, outcome(worker(11))).unwrap();
    let decode = p.dispatch(0).unwrap();
    // The host reports the prefill KV lost after decode started; only
    // decode could still read it and decode is already dispatched.
    assert_eq!(
        p.fail(1, prefill, failure(false)).err(),
        Some(PlanError::InvalidTransition {
            stage: 1,
            state: "completed",
            action: "fail"
        })
    );
    assert!(matches!(state(&p, 0), StageState::Dispatched(a) if *a == decode));
}

#[test]
fn retry_excludes_the_failed_worker_and_fences_the_old_attempt() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(vec![Stage {
        constraints: vec![Constraint::Previewed(worker(11))],
        ..Stage::new(WorkerType::Prefill)
    }]);
    assert!(matches!(
        p.retry(0),
        Err(PlanError::InvalidTransition { .. })
    ));
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    let first = p.dispatch(0).unwrap();
    p.fail(0, first, failure(true)).unwrap();

    assert_eq!(
        p.retry(0).unwrap(),
        Retry {
            stage: 0,
            attempt: first.next(),
            excluded: 11
        }
    );
    assert!(state(&p, 0).is_pending());
    assert_eq!(p.failures(0), Some(1));
    assert_eq!(
        p.stage(0).unwrap().constraints,
        vec![Constraint::Exclude(11)],
        "the router's own pin to the failed worker is dropped with it"
    );
    assert_eq!(
        p.book(0, booking("again", worker(11), &releases), zone("a"), None)
            .err(),
        Some(PlanError::ConstraintViolated {
            stage: 0,
            worker: worker(11)
        })
    );
    p.book(0, booking("p2", worker(12), &releases), zone("a"), None)
        .unwrap();
    let second = p.dispatch(0).unwrap();
    assert_eq!(second, first.next());

    let stale = PlanError::StaleAttempt {
        stage: 0,
        event: first,
        current: second,
    };
    assert_eq!(
        p.complete(0, first, outcome(worker(11))),
        Err(stale.clone())
    );
    assert_eq!(p.fail(0, first, failure(true)), Err(stale));
    assert!(matches!(state(&p, 0), StageState::Dispatched(a) if *a == second));
    assert_eq!(
        releases.load(Ordering::SeqCst),
        2,
        "the failed booking and the rejected re-book"
    );

    // A non-retryable failure cannot be retried.
    p.fail(0, second, failure(false)).unwrap();
    assert!(matches!(
        p.retry(0),
        Err(PlanError::InvalidTransition { .. })
    ));

    // A caller's pin is not relaxed: the stage cannot be retried elsewhere.
    let mut p = plan(vec![Stage {
        constraints: vec![Constraint::Pin(worker(11))],
        ..Stage::new(WorkerType::Prefill)
    }]);
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    let attempt = p.dispatch(0).unwrap();
    p.fail(0, attempt, failure(true)).unwrap();
    assert!(matches!(
        p.retry(0),
        Err(PlanError::InvalidTransition { .. })
    ));
}

#[test]
fn fail_follows_inputs_that_point_forward() {
    // Stage 0 waits for 1, stage 1 waits for 2. Failing the last stage
    // resets both, not just its direct reader.
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(vec![
        Stage {
            inputs: vec![1],
            ..Stage::new(WorkerType::Decode)
        },
        Stage {
            inputs: vec![2],
            ..Stage::new(WorkerType::Prefill)
        },
        Stage::new(WorkerType::Encode),
    ]);
    for (k, w) in [(0, 21), (1, 11), (2, 31)] {
        p.book(k, booking("b", worker(w), &releases), zone("a"), None)
            .unwrap();
    }
    let attempt = p.dispatch(2).unwrap();
    p.fail(2, attempt, failure(true)).unwrap();
    assert!(state(&p, 1).is_pending());
    assert!(
        state(&p, 0).is_pending(),
        "the reader of a reader is reset too"
    );
    assert_eq!(p.attempt(0), Some(StageAttempt::FIRST.next()));
    assert_eq!(releases.load(Ordering::SeqCst), 3);
}

#[test]
fn work_follows_the_set_unless_the_prefill_was_skipped() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(vec![
        Stage::new(WorkerType::Prefill),
        Stage {
            inputs: vec![0],
            constraints: vec![Constraint::TransferCompatible(0)],
            ..Stage::new(WorkerType::Decode)
        },
    ]);
    assert_eq!(p.work_of(0), Some(StageWork::PrefillOnly));
    assert_eq!(p.work_of(1), Some(StageWork::DecodeOnly));
    p.skip(0).unwrap();
    assert_eq!(
        p.work_of(1),
        Some(StageWork::PrefillAndDecode),
        "no remote prefill: decode does it"
    );
    assert_eq!(p.due().collect::<Vec<_>>(), vec![1]);
    p.book(1, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    let explicit = plan(vec![Stage {
        work: Some(StageWork::None),
        ..Stage::new(WorkerType::Decode)
    }]);
    assert_eq!(explicit.work_of(0), Some(StageWork::None));
}

#[test]
fn terminal_events_are_idempotent_for_the_current_attempt_and_otherwise_rejected() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(vec![Stage::new(WorkerType::Prefill)]);
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    assert!(
        matches!(
            p.complete(0, p.attempt(0).unwrap(), outcome(worker(11))),
            Err(PlanError::InvalidTransition { .. })
        ),
        "a booked stage was never dispatched"
    );
    let attempt = p.dispatch(0).unwrap();
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    assert!(matches!(
        p.fail(0, attempt, failure(true)),
        Err(PlanError::InvalidTransition { .. })
    ));

    // The host may decline to dispatch a booked stage: that is a failure.
    let mut p = plan(vec![Stage::new(WorkerType::Prefill)]);
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    let attempt = p.attempt(0).unwrap();
    p.fail(0, attempt, failure(false)).unwrap();
    p.fail(0, attempt, failure(false)).unwrap();
    assert!(matches!(
        p.complete(0, attempt, outcome(worker(11))),
        Err(PlanError::InvalidTransition { .. })
    ));
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn release_abort_and_drop_free_every_owned_booking_exactly_once() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(prefill_decode(When::Now, Budget::Immediate));
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    assert!(
        matches!(
            p.book(0, booking("dup", worker(12), &releases), zone("a"), None),
            Err(PlanError::InvalidTransition { .. })
        ),
        "a held stage keeps its booking"
    );
    assert_eq!(p.worker(0), Some(worker(11)));
    p.book(1, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    let attempt = p.dispatch(0).unwrap();
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    p.release().await.unwrap();
    assert_eq!(
        releases.load(Ordering::SeqCst),
        3,
        "p, d, and the rejected duplicate"
    );

    let mut p = plan(prefill_decode(When::Now, Budget::Immediate));
    p.book(0, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    p.abort();
    assert_eq!(releases.load(Ordering::SeqCst), 4);

    {
        let mut p = plan(prefill_decode(When::Now, Budget::Immediate));
        p.book(0, booking("p", worker(11), &releases), zone("a"), None)
            .unwrap();
        p.book(1, booking("d", worker(21), &releases), zone("a"), None)
            .unwrap();
        let _ = p.dispatch(0).unwrap();
    }
    assert_eq!(releases.load(Ordering::SeqCst), 6);

    // A committed booking's owner releases it through the wire.
    let mut p = plan(prefill_decode(When::Now, Budget::Immediate));
    let committed = Booking::Committed(SchedulerBookingDescriptor {
        request_id: "sel-1".to_string(),
        worker: worker(11),
        attempt_id: AttemptId::new(1),
    });
    assert!(!committed.is_owned());
    p.book(0, committed, zone("a"), None).unwrap();
    p.book(1, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    assert_eq!(p.booking(0).unwrap().id(), "sel-1");
    p.release().await.unwrap();
    assert_eq!(releases.load(Ordering::SeqCst), 7);
}

#[test]
fn a_cancelled_request_can_still_route_its_held_decode_for_kv_cleanup() {
    // DEP #15457 §6: the client is gone after prefill was dispatched, but
    // prefill is staging KV for one decode worker. The host keeps the plan,
    // forwards decode when its input settles, then releases everything.
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(decode_first());
    p.book(0, booking("d", worker(21), &releases), zone("a"), None)
        .unwrap();
    p.book(1, booking("p", worker(11), &releases), zone("a"), None)
        .unwrap();
    let prefill = p.dispatch(1).unwrap();
    // Cancellation arrives here. Nothing on the plan changes.
    assert_eq!(state(&p, 0), &StageState::Booked);
    p.complete(1, prefill, outcome(worker(11))).unwrap();
    assert_eq!(vec_of(p.ready()), vec![0]);
    let decode = p.dispatch(0).unwrap();
    p.complete(0, decode, outcome(worker(21))).unwrap();
    drop(p);
    assert_eq!(releases.load(Ordering::SeqCst), 2);
}

#[test]
fn a_taken_booking_belongs_to_the_host_and_is_released_once() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut plan = plan(prefill_decode(When::After(0), Budget::Full));
    plan.book(
        0,
        booking("p", worker(1), &releases),
        WorkerFacts::default(),
        None,
    )
    .unwrap();
    let taken = plan.take_booking(0).unwrap();
    assert_eq!(taken.worker(), worker(1));
    assert!(plan.booking(0).is_none(), "the plan no longer holds it");
    assert_eq!(
        plan.worker(0),
        Some(worker(1)),
        "but still knows the worker"
    );
    assert_eq!(state(&plan, 0), &StageState::Booked);
    // Failing the stage afterwards must not touch the moved booking.
    let attempt = plan.dispatch(0).unwrap();
    plan.fail(0, attempt, failure(true)).unwrap();
    assert_eq!(releases.load(Ordering::SeqCst), 0);
    drop(plan);
    assert_eq!(
        releases.load(Ordering::SeqCst),
        0,
        "drop frees nothing it does not own"
    );
    drop(taken);
    assert_eq!(
        releases.load(Ordering::SeqCst),
        1,
        "the host's drop frees it once"
    );
}

#[test]
fn take_booking_needs_a_booked_or_dispatched_stage() {
    let mut plan = plan(vec![Stage::new(WorkerType::Aggregated)]);
    assert!(matches!(
        plan.take_booking(0),
        Err(PlanError::InvalidTransition { .. })
    ));
    assert!(matches!(
        plan.take_booking(5),
        Err(PlanError::NoSuchStage { stage: 5 })
    ));
}

#[test]
fn an_untracked_booking_records_the_worker_without_a_handle() {
    let mut plan = plan(vec![
        Stage::new(WorkerType::Encode),
        Stage {
            when: When::After(0),
            inputs: vec![0],
            ..Stage::new(WorkerType::Prefill)
        },
    ]);
    plan.book_untracked(0, worker(31), zone("b")).unwrap();
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(plan.worker(0), Some(worker(31)));
    assert!(plan.booking(0).is_none());
    assert_eq!(plan.facts(0), Some(&zone("b")));
    assert!(matches!(
        plan.take_booking(0),
        Err(PlanError::InvalidTransition { .. })
    ));
    let attempt = plan.dispatch(0).unwrap();
    plan.complete(0, attempt, outcome(worker(31))).unwrap();
    assert!(plan.ready().next().is_none(), "prefill is not booked yet");
    // Only a stage that does no scheduler work may be untracked.
    let mut other = new_plan(vec![Stage::new(WorkerType::Prefill)]).unwrap();
    assert!(matches!(
        other.book_untracked(0, worker(1), WorkerFacts::default()),
        Err(PlanError::InvalidTransition { .. })
    ));
}

#[test]
fn a_handed_off_prefill_unblocks_decode_and_still_accepts_a_late_failure() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut plan = plan(prefill_decode(When::After(0), Budget::Full));
    plan.book(
        0,
        booking("p", worker(1), &releases),
        WorkerFacts::default(),
        None,
    )
    .unwrap();
    let attempt = plan.dispatch(0).unwrap();
    // A rejected booking drops and releases itself; count it apart.
    let rejected = Arc::new(AtomicUsize::new(0));
    assert!(
        matches!(
            plan.book(
                1,
                booking("d", worker(2), &rejected),
                WorkerFacts::default(),
                None
            ),
            Err(PlanError::InputsNotReady { stage: 1 })
        ),
        "decode waits for the handoff"
    );
    assert_eq!(rejected.load(Ordering::SeqCst), 1);
    plan.handoff(0, attempt).unwrap();
    assert_eq!(state(&plan, 0), &StageState::HandedOff(attempt));
    plan.book(
        1,
        booking("d", worker(2), &releases),
        WorkerFacts::default(),
        None,
    )
    .unwrap();
    assert_eq!(vec_of(plan.ready()), vec![1]);
    let decode_attempt = plan.dispatch(1).unwrap();
    assert_eq!(plan.state(), PlanState::Dispatched);
    // Late prefill failure: decode is running on a worker; it is not reset.
    plan.fail(0, attempt, failure(false)).unwrap();
    assert_eq!(state(&plan, 1), &StageState::Dispatched(decode_attempt));
    assert!(matches!(state(&plan, 0), StageState::Failed(..)));
    assert_eq!(
        releases.load(Ordering::SeqCst),
        1,
        "only the prefill booking was freed"
    );

    // Completing after a handoff is also fine.
    let mut ok = new_plan(vec![Stage::new(WorkerType::Prefill)]).unwrap();
    ok.book(
        0,
        booking("p", worker(1), &releases),
        WorkerFacts::default(),
        None,
    )
    .unwrap();
    let a = ok.dispatch(0).unwrap();
    ok.handoff(0, a).unwrap();
    ok.complete(0, a, outcome(worker(1))).unwrap();
    assert!(matches!(state(&ok, 0), StageState::Completed(_)));
}

#[test]
fn handoff_is_fenced_by_attempt_and_state() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut plan = plan(vec![Stage::new(WorkerType::Prefill)]);
    plan.book(
        0,
        booking("p", worker(1), &releases),
        WorkerFacts::default(),
        None,
    )
    .unwrap();
    assert!(
        matches!(
            plan.handoff(0, StageAttempt::FIRST),
            Err(PlanError::InvalidTransition { .. })
        ),
        "not dispatched yet"
    );
    let attempt = plan.dispatch(0).unwrap();
    assert!(matches!(
        plan.handoff(0, attempt.next()),
        Err(PlanError::StaleAttempt { .. })
    ));
    plan.handoff(0, attempt).unwrap();
    plan.handoff(0, attempt).unwrap();
}

#[test]
fn an_untracked_booking_obeys_the_stages_pins_and_exclusions() {
    let encode = |constraint: Constraint| {
        vec![Stage {
            constraints: vec![constraint],
            ..Stage::new(WorkerType::Encode)
        }]
    };
    // Pinned to worker 2: worker 3 is refused and nothing changes.
    let mut plan = new_plan(encode(Constraint::Pin(worker(2)))).unwrap();
    assert_eq!(
        plan.book_untracked(0, worker(3), WorkerFacts::default()),
        Err(PlanError::ConstraintViolated {
            stage: 0,
            worker: worker(3)
        })
    );
    assert_eq!(state(&plan, 0), &StageState::Pending);
    assert_eq!(plan.worker(0), None);
    // The pin names a DP rank too.
    assert!(matches!(
        plan.book_untracked(0, WorkerWithDpRank::new(2, 1), WorkerFacts::default()),
        Err(PlanError::ConstraintViolated { .. })
    ));
    plan.book_untracked(0, worker(2), WorkerFacts::default())
        .unwrap();
    assert_eq!(plan.worker(0), Some(worker(2)));
    // An exclusion bars the worker.
    let mut plan = new_plan(encode(Constraint::Exclude(3))).unwrap();
    assert!(matches!(
        plan.book_untracked(0, worker(3), WorkerFacts::default()),
        Err(PlanError::ConstraintViolated { .. })
    ));
    assert_eq!(state(&plan, 0), &StageState::Pending);
    plan.book_untracked(0, worker(4), WorkerFacts::default())
        .unwrap();
}

#[test]
fn a_plan_keeps_the_prompts_block_hashes_per_block_size() {
    let mut plan = plan(prefill_decode(When::Now, Budget::Full));
    assert!(plan.prompt_hashes(16).is_none());
    plan.record_prompt_hashes(16, vec![1, 2, 3]);
    plan.record_prompt_hashes(16, vec![9, 9, 9]);
    assert_eq!(
        plan.prompt_hashes(16),
        Some(&[1, 2, 3][..]),
        "the first set to hash wins"
    );
    assert!(
        plan.prompt_hashes(32).is_none(),
        "another block size hashes for itself"
    );
    plan.record_prompt_hashes(32, Vec::new());
    assert!(
        plan.prompt_hashes(32).is_none(),
        "an empty prompt records nothing"
    );
}
