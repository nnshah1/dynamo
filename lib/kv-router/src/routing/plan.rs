// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Plan`, `Stage` and the per-stage state machine.
//!
//! Invariants:
//! - Every owned booking is released exactly once. The plan moves a booking,
//!   never clones it; [`Booking`]'s drop is the backstop.
//! - The stage index is stable. The attempt increments on `retry`; an event
//!   naming another attempt is dropped.
//! - Booking order is list order: `when` and constraints point backward.
//!   Execution order is `inputs`, an acyclic graph over the whole list, so a
//!   stage may be booked first and forwarded last (decode-first).
//! - A failed stage takes every booked stage that reads it back to `Pending`,
//!   transitively.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::WorkerType;
use crate::identity::RoutingPartitionId;
use crate::kv_hints::KvHint;
use crate::protocols::{KvTransferEnforcement, WorkerConfigLike, WorkerWithDpRank};
use crate::sequences::SequenceError;

use super::booking::Booking;

/// Stable across `schedule` calls; a RouterService keys its copy by it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanId(String);

impl From<&str> for PlanId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl From<String> for PlanId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for PlanId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Which attempt of a stage an event belongs to; increments on `retry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StageAttempt(u32);

impl StageAttempt {
    pub const FIRST: Self = Self(0);

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for StageAttempt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// When a stage may be booked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum When {
    /// On the first `schedule` call, with the other `Now` stages.
    Now,
    /// Once stage `k` has completed or was skipped. Implies `inputs: [k]`.
    After(usize),
}

/// How long the stage's queue may hold the request while earlier bookings
/// are held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// The class's queue policy applies unchanged.
    Full,
    /// Admit or reject, nothing held across the gap: stages booked together.
    Zero,
    /// A deferred stage may wait this long while earlier work is held.
    Bounded(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainMode {
    Required,
    Preferred,
}

/// A placement rule. Rules that name a stage read that stage's booked
/// worker, so they may only name earlier stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint {
    /// The worker must be able to receive KV from stage `k`'s worker.
    TransferCompatible(usize),
    /// The worker must (or should) share topology domain `key` with stage
    /// `k`'s worker.
    SameDomain {
        stage: usize,
        key: String,
        mode: DomainMode,
    },
    Pin(WorkerWithDpRank),
    /// `retry` appends one per failed attempt.
    Exclude(WorkerWithDpRank),
}

impl Constraint {
    fn reads(&self) -> Option<usize> {
        match self {
            Self::TransferCompatible(stage) | Self::SameDomain { stage, .. } => Some(*stage),
            Self::Pin(_) | Self::Exclude(_) => None,
        }
    }
}

/// When the router skips a stage instead of booking it.
#[derive(Debug, Clone, PartialEq)]
pub enum SkipRule {
    /// No multimodal input (an encode stage).
    NoMultimodal,
    /// The decode worker already holds the prefix (a prefill stage).
    DecodeHoldsPrefix,
    /// The stage's worker set is busier than this fraction.
    SetBusy(f64),
    /// The conditional-disaggregation policy and decode-busy gate.
    ConditionalDisagg,
}

/// What a later stage's placement rules read about a booked worker. Captured
/// at booking so the plan still has them if the worker leaves discovery.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkerFacts {
    pub taints: HashSet<String>,
    pub topology_domains: HashMap<String, String>,
    pub kv_transfer_domain: Option<String>,
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub kv_transfer_preferred_weight: Option<f32>,
}

impl WorkerFacts {
    pub fn from_config<C: WorkerConfigLike + ?Sized>(config: &C) -> Self {
        Self {
            taints: config.taints().clone(),
            topology_domains: config.topology_domains().cloned().unwrap_or_default(),
            kv_transfer_domain: config.kv_transfer_domain().map(str::to_string),
            kv_transfer_enforcement: config.kv_transfer_enforcement(),
            kv_transfer_preferred_weight: config.kv_transfer_preferred_weight(),
        }
    }

    pub fn topology_value(&self, domain: &str) -> Option<&str> {
        self.topology_domains.get(domain).map(String::as_str)
    }
}

/// Routing facts a host records when a stage completes. Engine-opaque data
/// never enters the plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub worker: WorkerWithDpRank,
    pub kv_hint: Option<KvHint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub is_retryable: bool,
    pub reason: String,
}

/// What `retry` changed; the host passes the plan back to `schedule` next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Retry {
    pub stage: usize,
    pub attempt: StageAttempt,
    pub excluded: WorkerWithDpRank,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StageState {
    Pending,
    Booked,
    Dispatched(StageAttempt),
    Completed(Outcome),
    Failed(StageAttempt, Failure),
    Skipped,
}

impl StageState {
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Completed or skipped: a dependent's input is available.
    pub fn is_settled(&self) -> bool {
        matches!(self, Self::Completed(_) | Self::Skipped)
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Booked => "booked",
            Self::Dispatched(_) => "dispatched",
            Self::Completed(_) => "completed",
            Self::Failed(..) => "failed",
            Self::Skipped => "skipped",
        }
    }
}

/// One element of a plan: a kind of work on its worker set, when it is
/// booked, what it waits for before it runs, how long it may wait, and its
/// placement rules. Pure configuration; the plan keeps the state beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    pub set: WorkerType,
    pub when: When,
    /// Stages whose outcome this one needs before it is forwarded.
    pub inputs: Vec<usize>,
    pub wait: Budget,
    pub constraints: Vec<Constraint>,
    pub skip: Option<SkipRule>,
}

impl Stage {
    pub fn new(set: WorkerType) -> Self {
        Self {
            set,
            when: When::Now,
            inputs: Vec::new(),
            wait: Budget::Full,
            constraints: Vec::new(),
            skip: None,
        }
    }

    /// Every stage this one reads: `when`, `inputs` and constraints.
    fn reads(&self) -> impl Iterator<Item = usize> + '_ {
        let when = match self.when {
            When::After(stage) => Some(stage),
            When::Now => None,
        };
        when.into_iter()
            .chain(self.inputs.iter().copied())
            .chain(self.constraints.iter().filter_map(Constraint::reads))
    }
}

#[derive(Debug)]
struct Slot {
    stage: Stage,
    state: StageState,
    booking: Option<Booking>,
    facts: Option<WorkerFacts>,
    attempt: StageAttempt,
    /// The worker the last attempt was booked on; `retry` excludes it.
    last_worker: Option<WorkerWithDpRank>,
}

impl Slot {
    fn new(stage: Stage) -> Self {
        Self {
            stage,
            state: StageState::Pending,
            booking: None,
            facts: None,
            attempt: StageAttempt::FIRST,
            last_worker: None,
        }
    }

    fn reset(&mut self, state: StageState) -> Option<Booking> {
        self.state = state;
        self.facts = None;
        self.booking.take()
    }
}

/// Derived from the stages: nothing booked, something booked, something
/// forwarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanState {
    Planned,
    Booked,
    Dispatched,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    #[error("a plan needs at least one stage")]
    Empty,
    #[error("stage {stage} is out of range")]
    NoSuchStage { stage: usize },
    #[error("stage {stage} is booked against stage {depends_on}, which is not earlier")]
    ForwardDependency { stage: usize, depends_on: usize },
    #[error("stage {stage}'s inputs form a cycle")]
    DependencyCycle { stage: usize },
    #[error("stage {stage} is {state}: cannot {action}")]
    InvalidTransition {
        stage: usize,
        state: &'static str,
        action: &'static str,
    },
    #[error("stage {stage}: event for attempt {event} but the stage is on attempt {current}")]
    StaleAttempt {
        stage: usize,
        event: StageAttempt,
        current: StageAttempt,
    },
    #[error("stage {stage}: its inputs are not ready")]
    InputsNotReady { stage: usize },
    #[error("stage {stage}: worker {worker:?} violates the stage's constraints")]
    ConstraintViolated {
        stage: usize,
        worker: WorkerWithDpRank,
    },
}

/// The Router's answer for one request: ordered stages and their state.
#[derive(Debug)]
pub struct Plan {
    id: PlanId,
    partition: RoutingPartitionId,
    slots: Vec<Slot>,
}

impl Plan {
    /// Every stage pending. `when` and constraints must name earlier stages;
    /// `inputs` may name any other stage but must not form a cycle.
    pub fn new(
        id: PlanId,
        partition: RoutingPartitionId,
        mut stages: Vec<Stage>,
    ) -> Result<Self, PlanError> {
        if stages.is_empty() {
            return Err(PlanError::Empty);
        }
        let len = stages.len();
        for (k, stage) in stages.iter_mut().enumerate() {
            let when = match stage.when {
                When::After(j) => Some(j),
                When::Now => None,
            };
            if let Some(j) = when
                .into_iter()
                .chain(stage.constraints.iter().filter_map(Constraint::reads))
                .find(|j| *j >= k)
            {
                return Err(PlanError::ForwardDependency {
                    stage: k,
                    depends_on: j,
                });
            }
            if let Some(j) = when
                && !stage.inputs.contains(&j)
            {
                stage.inputs.push(j);
            }
            if let Some(j) = stage.inputs.iter().find(|j| **j >= len) {
                return Err(PlanError::NoSuchStage { stage: *j });
            }
        }
        check_acyclic(&stages)?;
        Ok(Self {
            id,
            partition,
            slots: stages.into_iter().map(Slot::new).collect(),
        })
    }

    pub fn id(&self) -> &PlanId {
        &self.id
    }

    pub fn partition(&self) -> &RoutingPartitionId {
        &self.partition
    }

    pub fn stage_count(&self) -> usize {
        self.slots.len()
    }

    pub fn stages(&self) -> impl Iterator<Item = &Stage> {
        self.slots.iter().map(|slot| &slot.stage)
    }

    pub fn stage(&self, k: usize) -> Option<&Stage> {
        self.slots.get(k).map(|slot| &slot.stage)
    }

    pub fn state_of(&self, k: usize) -> Option<&StageState> {
        self.slots.get(k).map(|slot| &slot.state)
    }

    pub fn attempt(&self, k: usize) -> Option<StageAttempt> {
        self.slots.get(k).map(|slot| slot.attempt)
    }

    pub fn booking(&self, k: usize) -> Option<&Booking> {
        self.slots.get(k)?.booking.as_ref()
    }

    /// The booked worker's placement facts, for a later stage's constraints.
    pub fn facts(&self, k: usize) -> Option<&WorkerFacts> {
        self.slots.get(k)?.facts.as_ref()
    }

    pub fn outcome(&self, k: usize) -> Option<&Outcome> {
        match &self.slots.get(k)?.state {
            StageState::Completed(outcome) => Some(outcome),
            _ => None,
        }
    }

    /// The worker stage `k` is booked on.
    pub fn worker(&self, k: usize) -> Option<WorkerWithDpRank> {
        self.booking(k).map(Booking::worker)
    }

    pub fn state(&self) -> PlanState {
        let mut state = PlanState::Planned;
        for slot in &self.slots {
            match slot.state {
                StageState::Dispatched(_) | StageState::Completed(_) => {
                    return PlanState::Dispatched;
                }
                StageState::Booked => state = PlanState::Booked,
                StageState::Pending | StageState::Failed(..) | StageState::Skipped => {}
            }
        }
        state
    }

    pub fn has_pending(&self) -> bool {
        self.slots.iter().any(|slot| slot.state.is_pending())
    }

    /// Pending stages the router can book on this `schedule` call: their
    /// `when` is satisfied and every constraint has a worker to read.
    pub fn schedulable(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.slots.len()).filter(move |&k| self.is_bookable(k))
    }

    /// Booked stages whose inputs have all completed: what the host forwards.
    pub fn ready(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.slots.len())
            .filter(move |&k| self.slots[k].state == StageState::Booked && self.inputs_settled(k))
    }

    fn inputs_settled(&self, k: usize) -> bool {
        self.slots[k]
            .stage
            .inputs
            .iter()
            .all(|&j| self.slots[j].state.is_settled())
    }

    fn is_bookable(&self, k: usize) -> bool {
        let slot = &self.slots[k];
        slot.state.is_pending()
            && match slot.stage.when {
                When::Now => true,
                When::After(j) => self.slots[j].state.is_settled(),
            }
            && slot
                .stage
                .constraints
                .iter()
                .filter_map(Constraint::reads)
                .all(|j| {
                    self.slots[j].booking.is_some() || self.slots[j].state == StageState::Skipped
                })
    }

    fn check_index(&self, k: usize) -> Result<(), PlanError> {
        if k < self.slots.len() {
            Ok(())
        } else {
            Err(PlanError::NoSuchStage { stage: k })
        }
    }

    fn invalid_transition(&self, k: usize, action: &'static str) -> PlanError {
        PlanError::InvalidTransition {
            stage: k,
            state: self.slots[k].state.name(),
            action,
        }
    }

    fn check_attempt(&self, k: usize, event: StageAttempt) -> Result<(), PlanError> {
        let current = self.slots[k].attempt;
        if event == current {
            Ok(())
        } else {
            Err(PlanError::StaleAttempt {
                stage: k,
                event,
                current,
            })
        }
    }

    /// Record the booking the router made for stage `k` and the facts a
    /// later stage's constraints read. `Pin` and `Exclude` are enforced
    /// here. A rejected booking drops and releases itself.
    pub fn book(
        &mut self,
        k: usize,
        booking: Booking,
        facts: WorkerFacts,
    ) -> Result<(), PlanError> {
        self.check_index(k)?;
        if !self.slots[k].state.is_pending() {
            return Err(self.invalid_transition(k, "book"));
        }
        if !self.is_bookable(k) {
            return Err(PlanError::InputsNotReady { stage: k });
        }
        let worker = booking.worker();
        let is_allowed =
            self.slots[k]
                .stage
                .constraints
                .iter()
                .all(|constraint| match constraint {
                    Constraint::Pin(pinned) => *pinned == worker,
                    Constraint::Exclude(excluded) => *excluded != worker,
                    Constraint::TransferCompatible(_) | Constraint::SameDomain { .. } => true,
                });
        if !is_allowed {
            return Err(PlanError::ConstraintViolated { stage: k, worker });
        }
        let slot = &mut self.slots[k];
        slot.last_worker = Some(worker);
        slot.booking = Some(booking);
        slot.facts = Some(facts);
        slot.state = StageState::Booked;
        Ok(())
    }

    pub fn skip(&mut self, k: usize) -> Result<(), PlanError> {
        self.check_index(k)?;
        if !self.slots[k].state.is_pending() {
            return Err(self.invalid_transition(k, "skip"));
        }
        self.slots[k].state = StageState::Skipped;
        Ok(())
    }

    /// The host forwarded stage `k`. Returns the attempt its events must name.
    pub fn dispatch(&mut self, k: usize) -> Result<StageAttempt, PlanError> {
        self.check_index(k)?;
        if self.slots[k].state != StageState::Booked {
            return Err(self.invalid_transition(k, "dispatch"));
        }
        if !self.inputs_settled(k) {
            return Err(PlanError::InputsNotReady { stage: k });
        }
        let attempt = self.slots[k].attempt;
        self.slots[k].state = StageState::Dispatched(attempt);
        Ok(attempt)
    }

    /// Stage `k`'s attempt finished with routing facts. The booking stays
    /// held until `release`. Idempotent for the current attempt.
    pub fn complete(
        &mut self,
        k: usize,
        attempt: StageAttempt,
        outcome: Outcome,
    ) -> Result<(), PlanError> {
        self.check_index(k)?;
        self.check_attempt(k, attempt)?;
        match &self.slots[k].state {
            StageState::Dispatched(_) => {}
            StageState::Completed(_) => return Ok(()),
            StageState::Pending
            | StageState::Booked
            | StageState::Failed(..)
            | StageState::Skipped => {
                return Err(self.invalid_transition(k, "complete"));
            }
        }
        debug_assert_eq!(self.slots[k].last_worker, Some(outcome.worker));
        self.slots[k].state = StageState::Completed(outcome);
        Ok(())
    }

    /// Stage `k`'s attempt failed, or the host declined to dispatch a booked
    /// stage. Its booking is released and every booked stage that reads it,
    /// transitively, falls back to `Pending`. Idempotent for the current
    /// attempt.
    pub fn fail(
        &mut self,
        k: usize,
        attempt: StageAttempt,
        cause: Failure,
    ) -> Result<(), PlanError> {
        self.check_index(k)?;
        self.check_attempt(k, attempt)?;
        match &self.slots[k].state {
            StageState::Booked | StageState::Dispatched(_) => {}
            StageState::Failed(..) => return Ok(()),
            StageState::Pending | StageState::Completed(_) | StageState::Skipped => {
                return Err(self.invalid_transition(k, "fail"));
            }
        }
        drop(self.slots[k].reset(StageState::Failed(attempt, cause)));
        // Dependents of dependents also lose their placement; one forward
        // pass suffices because booking edges point backward and a stage
        // booked against an `inputs`-only edge is re-placed with the rest.
        let mut is_reset = vec![false; self.slots.len()];
        is_reset[k] = true;
        for j in 0..self.slots.len() {
            if self.slots[j].state == StageState::Booked
                && self.slots[j].stage.reads().any(|i| is_reset[i])
            {
                drop(self.slots[j].reset(StageState::Pending));
                is_reset[j] = true;
            }
        }
        Ok(())
    }

    /// Prepare stage `k` to be booked again: exclude the failed worker and
    /// move to the next attempt. The host then passes the plan back to
    /// `schedule`.
    pub fn retry(&mut self, k: usize) -> Result<Retry, PlanError> {
        self.check_index(k)?;
        let (StageState::Failed(_, failure), Some(excluded)) =
            (&self.slots[k].state, self.slots[k].last_worker)
        else {
            return Err(self.invalid_transition(k, "retry"));
        };
        if !failure.is_retryable {
            return Err(self.invalid_transition(k, "retry a non-retryable failure"));
        }
        let slot = &mut self.slots[k];
        slot.stage.constraints.push(Constraint::Exclude(excluded));
        slot.attempt = slot.attempt.next();
        slot.state = StageState::Pending;
        Ok(Retry {
            stage: k,
            attempt: slot.attempt,
            excluded,
        })
    }

    /// The request ended: free every booking and wait for each scheduler to
    /// acknowledge. Dropping the plan instead frees them without waiting.
    pub async fn release(mut self) -> Result<(), SequenceError> {
        for slot in &mut self.slots {
            if let Some(booking) = slot.booking.take() {
                booking.release().await?;
            }
        }
        Ok(())
    }

    /// The request will not continue: free every booking now, without
    /// waiting for acknowledgement.
    pub fn abort(self) {
        drop(self);
    }
}

/// `inputs` may point forward (decode-first), so cycles are possible; reject
/// them with a depth-first walk.
fn check_acyclic(stages: &[Stage]) -> Result<(), PlanError> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        New,
        Open,
        Done,
    }
    fn visit(k: usize, stages: &[Stage], marks: &mut [Mark]) -> Result<(), PlanError> {
        match marks[k] {
            Mark::Done => return Ok(()),
            Mark::Open => return Err(PlanError::DependencyCycle { stage: k }),
            Mark::New => {}
        }
        marks[k] = Mark::Open;
        for &j in &stages[k].inputs {
            visit(j, stages, marks)?;
        }
        marks[k] = Mark::Done;
        Ok(())
    }
    let mut marks = vec![Mark::New; stages.len()];
    (0..stages.len()).try_for_each(|k| visit(k, stages, &mut marks))
}
