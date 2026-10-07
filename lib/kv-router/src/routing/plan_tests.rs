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
    Stage, StageState, When, WorkerFacts,
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

    p.book(0, booking("p", worker(11), &releases), zone("a"))
        .unwrap();
    assert_eq!(p.state(), PlanState::Booked);
    assert_eq!(vec_of(p.ready()), vec![0]);
    assert_eq!(vec_of(p.schedulable()), Vec::<usize>::new());
    assert_eq!(
        p.book(1, booking("early", worker(21), &releases), zone("a"))
            .err(),
        Some(PlanError::InputsNotReady { stage: 1 })
    );

    let attempt = p.dispatch(0).unwrap();
    assert_eq!(p.state(), PlanState::Dispatched);
    assert_eq!(vec_of(p.ready()), Vec::<usize>::new());
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    assert_eq!(p.facts(0).unwrap().topology_value("zone"), Some("a"));

    p.book(1, booking("d", worker(21), &releases), zone("a"))
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
    let mut p = plan(prefill_decode(When::Now, Budget::Zero));
    assert_eq!(
        vec_of(p.schedulable()),
        vec![0],
        "decode reads prefill's worker"
    );
    p.book(0, booking("p", worker(11), &releases), zone("a"))
        .unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    p.book(1, booking("d", worker(21), &releases), zone("a"))
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
    p.book(0, booking("d", worker(21), &releases), zone("a"))
        .unwrap();
    assert_eq!(vec_of(p.schedulable()), vec![1]);
    assert_eq!(
        vec_of(p.ready()),
        Vec::<usize>::new(),
        "decode waits for prefill"
    );
    p.book(1, booking("p", worker(11), &releases), zone("a"))
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
    p.book(1, booking("p", worker(11), &releases), zone("a"))
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
    p.book(0, booking("e", worker(31), &releases), zone("a"))
        .unwrap();
    p.book(1, booking("p", worker(11), &releases), zone("a"))
        .unwrap();
    p.book(2, booking("d", worker(21), &releases), zone("a"))
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
    assert_eq!(p.state(), PlanState::Planned);

    // A dispatched dependent is in flight and is kept.
    let mut p = plan(decode_first());
    p.book(0, booking("d", worker(21), &releases), zone("a"))
        .unwrap();
    p.book(1, booking("p", worker(11), &releases), zone("a"))
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
    let mut p = plan(vec![Stage::new(WorkerType::Prefill)]);
    assert!(matches!(
        p.retry(0),
        Err(PlanError::InvalidTransition { .. })
    ));
    p.book(0, booking("p", worker(11), &releases), zone("a"))
        .unwrap();
    let first = p.dispatch(0).unwrap();
    p.fail(0, first, failure(true)).unwrap();

    assert_eq!(
        p.retry(0).unwrap(),
        Retry {
            stage: 0,
            attempt: first.next(),
            excluded: worker(11)
        }
    );
    assert!(state(&p, 0).is_pending());
    assert_eq!(
        p.book(0, booking("again", worker(11), &releases), zone("a"))
            .err(),
        Some(PlanError::ConstraintViolated {
            stage: 0,
            worker: worker(11)
        })
    );
    p.book(0, booking("p2", worker(12), &releases), zone("a"))
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
}

#[test]
fn terminal_events_are_idempotent_for_the_current_attempt_and_otherwise_rejected() {
    let releases = Arc::new(AtomicUsize::new(0));
    let mut p = plan(vec![Stage::new(WorkerType::Prefill)]);
    p.book(0, booking("p", worker(11), &releases), zone("a"))
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
    p.book(0, booking("p", worker(11), &releases), zone("a"))
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
    let mut p = plan(prefill_decode(When::Now, Budget::Zero));
    p.book(0, booking("p", worker(11), &releases), zone("a"))
        .unwrap();
    assert!(
        matches!(
            p.book(0, booking("dup", worker(12), &releases), zone("a")),
            Err(PlanError::InvalidTransition { .. })
        ),
        "a held stage keeps its booking"
    );
    assert_eq!(p.worker(0), Some(worker(11)));
    p.book(1, booking("d", worker(21), &releases), zone("a"))
        .unwrap();
    let attempt = p.dispatch(0).unwrap();
    p.complete(0, attempt, outcome(worker(11))).unwrap();
    p.release().await.unwrap();
    assert_eq!(
        releases.load(Ordering::SeqCst),
        3,
        "p, d, and the rejected duplicate"
    );

    let mut p = plan(prefill_decode(When::Now, Budget::Zero));
    p.book(0, booking("p", worker(11), &releases), zone("a"))
        .unwrap();
    p.abort();
    assert_eq!(releases.load(Ordering::SeqCst), 4);

    {
        let mut p = plan(prefill_decode(When::Now, Budget::Zero));
        p.book(0, booking("p", worker(11), &releases), zone("a"))
            .unwrap();
        p.book(1, booking("d", worker(21), &releases), zone("a"))
            .unwrap();
        let _ = p.dispatch(0).unwrap();
    }
    assert_eq!(releases.load(Ordering::SeqCst), 6);

    // A committed booking's owner releases it through the wire.
    let mut p = plan(prefill_decode(When::Now, Budget::Zero));
    let committed = Booking::Committed(SchedulerBookingDescriptor {
        request_id: "sel-1".to_string(),
        worker: worker(11),
        attempt_id: AttemptId::new(1),
    });
    assert!(!committed.is_owned());
    p.book(0, committed, zone("a")).unwrap();
    p.book(1, booking("d", worker(21), &releases), zone("a"))
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
    p.book(0, booking("d", worker(21), &releases), zone("a"))
        .unwrap();
    p.book(1, booking("p", worker(11), &releases), zone("a"))
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
