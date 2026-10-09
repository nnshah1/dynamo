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
use crate::protocols::{KvTransferEnforcement, WorkerConfigLike, WorkerId, WorkerWithDpRank};
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
    /// Admit on this pass or reject; nothing held across the gap. For stages
    /// booked together.
    Immediate,
    /// A deferred stage may wait this long while earlier work is held.
    Bounded(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DomainMode {
    Required,
    /// Adds `weight` to the matching topology taint's preference.
    Preferred {
        weight: f32,
    },
}

/// A placement rule. Rules that name a stage read that stage's booked
/// worker, so they may only name earlier stages. `Plan::routing_constraints`
/// turns the first two into today's topology taints.
#[derive(Debug, Clone, PartialEq)]
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
    /// The caller's or the class's pin; never relaxed.
    Pin(WorkerWithDpRank),
    /// The router's own pin to the worker it previewed for this stage;
    /// `retry` drops it together with the failed worker.
    Previewed(WorkerWithDpRank),
    /// Every rank of the worker; `retry` appends one per failed attempt.
    Exclude(WorkerId),
}

impl Constraint {
    pub(super) fn reads(&self) -> Option<usize> {
        match self {
            Self::TransferCompatible(stage) | Self::SameDomain { stage, .. } => Some(*stage),
            Self::Pin(_) | Self::Previewed(_) | Self::Exclude(_) => None,
        }
    }
}

/// What a stage's worker does for the request: what its booking is charged
/// for. Derived from the set when a stage does not say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageWork {
    /// Prefill and decode on one worker.
    PrefillAndDecode,
    /// Prefill only; the output is one handoff token.
    PrefillOnly,
    /// Decode from a remote prefill; the prompt is not this worker's load.
    DecodeOnly,
    /// No scheduler accounting (an encoder).
    None,
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
    pub excluded: WorkerId,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StageState {
    Pending,
    Booked,
    Dispatched(StageAttempt),
    /// Dispatched, and it has produced what its dependents need (bootstrap
    /// info, a first output) while it is still running.
    HandedOff(StageAttempt),
    Completed(Outcome),
    Failed(StageAttempt, Failure),
    Skipped,
}

impl StageState {
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Completed, handed off or skipped: a dependent's input is available.
    pub fn is_settled(&self) -> bool {
        matches!(
            self,
            Self::Completed(_) | Self::HandedOff(_) | Self::Skipped
        )
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Booked => "booked",
            Self::Dispatched(_) => "dispatched",
            Self::HandedOff(_) => "handed off",
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
    /// Overrides the accounting derived from the set; see [`Plan::work_of`].
    pub work: Option<StageWork>,
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
            work: None,
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
    kv_hint: Option<KvHint>,
    attempt: StageAttempt,
    /// Attempts the host reported failed and retried; a limit counts these,
    /// not re-placements caused by another stage's failure.
    failures: u32,
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
            kv_hint: None,
            attempt: StageAttempt::FIRST,
            failures: 0,
            last_worker: None,
        }
    }

    fn reset(&mut self, state: StageState) -> Option<Booking> {
        self.state = state;
        self.facts = None;
        self.kv_hint = None;
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
    #[error("stage {stage} cannot be placed against stage {reads}: {reason}")]
    Placement {
        stage: usize,
        reads: usize,
        reason: String,
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

    /// How many of stage `k`'s own attempts the host failed and retried.
    pub fn failures(&self, k: usize) -> Option<u32> {
        self.slots.get(k).map(|slot| slot.failures)
    }

    /// The booking the plan holds for stage `k`: `None` before booking, after
    /// `take_booking`, and for a stage booked untracked.
    pub fn booking(&self, k: usize) -> Option<&Booking> {
        self.slots.get(k)?.booking.as_ref()
    }

    /// The booked worker's placement facts, for a later stage's constraints.
    pub fn facts(&self, k: usize) -> Option<&WorkerFacts> {
        self.slots.get(k)?.facts.as_ref()
    }

    /// The KV hint the router attached to stage `k`'s booking; the host
    /// forwards it with the request.
    pub fn kv_hint(&self, k: usize) -> Option<&KvHint> {
        self.slots.get(k)?.kv_hint.as_ref()
    }

    pub fn outcome(&self, k: usize) -> Option<&Outcome> {
        match &self.slots.get(k)?.state {
            StageState::Completed(outcome) => Some(outcome),
            _ => None,
        }
    }

    /// The worker stage `k` is booked on.
    /// The worker stage `k` is booked on or ran on; `None` before booking
    /// and after a failure.
    pub fn worker(&self, k: usize) -> Option<WorkerWithDpRank> {
        let slot = self.slots.get(k)?;
        match slot.state {
            StageState::Pending | StageState::Failed(..) | StageState::Skipped => None,
            _ => slot.last_worker,
        }
    }

    pub fn state(&self) -> PlanState {
        let mut state = PlanState::Planned;
        for slot in &self.slots {
            match slot.state {
                StageState::Dispatched(_) | StageState::HandedOff(_) | StageState::Completed(_) => {
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

    /// Pending stages whose `when` is satisfied, whether or not their
    /// constraints can be read yet: what a router decides skips for before
    /// any of them is booked.
    pub fn due(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.slots.len()).filter(move |&k| {
            let slot = &self.slots[k];
            slot.state.is_pending()
                && match slot.stage.when {
                    When::Now => true,
                    When::After(j) => self.slots[j].state.is_settled(),
                }
        })
    }

    /// What stage `k`'s booking is charged for: the stage's own `work`, else
    /// by set. A decode stage that reads a prefill stage decodes only, unless
    /// that prefill was skipped, in which case it does the prefill itself.
    pub fn work_of(&self, k: usize) -> Option<StageWork> {
        let slot = self.slots.get(k)?;
        if let Some(work) = slot.stage.work {
            return Some(work);
        }
        Some(match slot.stage.set {
            WorkerType::Aggregated => StageWork::PrefillAndDecode,
            WorkerType::Prefill => StageWork::PrefillOnly,
            WorkerType::Encode => StageWork::None,
            WorkerType::Decode => {
                let remote_prefill = slot.stage.reads().any(|j| {
                    self.slots[j].stage.set == WorkerType::Prefill
                        && self.slots[j].state != StageState::Skipped
                });
                if remote_prefill {
                    StageWork::DecodeOnly
                } else {
                    StageWork::PrefillAndDecode
                }
            }
        })
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

    /// Record the booking the router made for stage `k`, the facts a later
    /// stage's constraints read, and the KV hint the host forwards. `Pin`
    /// and `Exclude` are enforced here. A rejected booking drops and
    /// releases itself.
    pub fn book(
        &mut self,
        k: usize,
        booking: Booking,
        facts: WorkerFacts,
        kv_hint: Option<KvHint>,
    ) -> Result<(), PlanError> {
        self.check_index(k)?;
        if !self.slots[k].state.is_pending() {
            return Err(self.invalid_transition(k, "book"));
        }
        if !self.is_bookable(k) {
            return Err(PlanError::InputsNotReady { stage: k });
        }
        let worker = booking.worker();
        self.check_worker(k, worker)?;
        let slot = &mut self.slots[k];
        slot.last_worker = Some(worker);
        slot.booking = Some(booking);
        slot.facts = Some(facts);
        slot.kv_hint = kv_hint;
        slot.state = StageState::Booked;
        Ok(())
    }

    /// Move the booking out of a booked, dispatched or handed-off stage. The
    /// stage keeps its worker, facts and state; the plan no longer releases
    /// or fails the booking, whose owner is now the caller.
    pub fn take_booking(&mut self, k: usize) -> Result<Booking, PlanError> {
        self.check_index(k)?;
        match self.slots[k].state {
            StageState::Booked | StageState::Dispatched(_) | StageState::HandedOff(_) => {}
            _ => return Err(self.invalid_transition(k, "take the booking of")),
        }
        self.slots[k]
            .booking
            .take()
            .ok_or_else(|| self.invalid_transition(k, "take the booking of an untracked"))
    }

    /// Book a stage that does no scheduler work (an encoder): record the
    /// worker and its facts with no booking behind them.
    pub fn book_untracked(
        &mut self,
        k: usize,
        worker: WorkerWithDpRank,
        facts: WorkerFacts,
    ) -> Result<(), PlanError> {
        self.check_index(k)?;
        if self.work_of(k) != Some(StageWork::None) {
            return Err(self.invalid_transition(k, "book untracked a stage that does work on"));
        }
        if !self.slots[k].state.is_pending() {
            return Err(self.invalid_transition(k, "book"));
        }
        if !self.is_bookable(k) {
            return Err(PlanError::InputsNotReady { stage: k });
        }
        self.check_worker(k, worker)?;
        let slot = &mut self.slots[k];
        slot.last_worker = Some(worker);
        slot.booking = None;
        slot.facts = Some(facts);
        slot.kv_hint = None;
        slot.state = StageState::Booked;
        Ok(())
    }

    /// The stage's own placement rules: a pin names the worker, an exclusion
    /// bars one. The same contract whether or not a scheduler booked it.
    fn check_worker(&self, k: usize, worker: WorkerWithDpRank) -> Result<(), PlanError> {
        let is_allowed =
            self.slots[k]
                .stage
                .constraints
                .iter()
                .all(|constraint| match constraint {
                    Constraint::Pin(pinned) | Constraint::Previewed(pinned) => *pinned == worker,
                    Constraint::Exclude(excluded) => *excluded != worker.worker_id,
                    Constraint::TransferCompatible(_) | Constraint::SameDomain { .. } => true,
                });
        if is_allowed {
            Ok(())
        } else {
            Err(PlanError::ConstraintViolated { stage: k, worker })
        }
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

    /// Stage `k`'s attempt is running and has handed its dependents what
    /// they need: they may be booked and forwarded now. The stage still
    /// completes or fails later. Idempotent for the current attempt.
    pub fn handoff(&mut self, k: usize, attempt: StageAttempt) -> Result<(), PlanError> {
        self.check_index(k)?;
        self.check_attempt(k, attempt)?;
        match &self.slots[k].state {
            StageState::Dispatched(_) => {}
            StageState::HandedOff(_) => return Ok(()),
            _ => return Err(self.invalid_transition(k, "hand off")),
        }
        self.slots[k].state = StageState::HandedOff(attempt);
        Ok(())
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
            StageState::Dispatched(_) | StageState::HandedOff(_) => {}
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
            StageState::Booked | StageState::Dispatched(_) | StageState::HandedOff(_) => {}
            StageState::Failed(..) => return Ok(()),
            StageState::Pending | StageState::Completed(_) | StageState::Skipped => {
                return Err(self.invalid_transition(k, "fail"));
            }
        }
        drop(self.slots[k].reset(StageState::Failed(attempt, cause)));
        // Only booked dependents lose their placement: one already running
        // on a worker is the host's to fail. Dependents of dependents also
        // lose theirs. `inputs` may
        // point either way, so repeat until nothing changes; the graph is
        // acyclic, so this ends. A re-placed dependent is a new attempt:
        // its next booking gets a new id and events for the released one
        // are stale.
        let mut is_reset = vec![false; self.slots.len()];
        is_reset[k] = true;
        loop {
            let mut changed = false;
            for j in 0..self.slots.len() {
                if self.slots[j].state == StageState::Booked
                    && self.slots[j].stage.reads().any(|i| is_reset[i])
                {
                    drop(self.slots[j].reset(StageState::Pending));
                    self.slots[j].attempt = self.slots[j].attempt.next();
                    is_reset[j] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }

    /// Prepare stage `k` to be booked again: exclude the failed worker, drop
    /// the router's own `Previewed` pin to it (a caller's `Pin` stays, and a
    /// pinned stage that failed cannot be retried), count the failure, and
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
        if self.slots[k].stage.constraints.iter().any(
            |constraint| matches!(constraint, Constraint::Pin(pinned) if pinned.worker_id == excluded.worker_id),
        ) {
            return Err(self.invalid_transition(k, "retry a stage pinned to the failed worker"));
        }
        let slot = &mut self.slots[k];
        slot.stage.constraints.retain(|constraint| {
            !matches!(constraint, Constraint::Previewed(pinned) if pinned.worker_id == excluded.worker_id)
        });
        slot.stage
            .constraints
            .push(Constraint::Exclude(excluded.worker_id));
        slot.failures += 1;
        slot.attempt = slot.attempt.next();
        slot.state = StageState::Pending;
        Ok(Retry {
            stage: k,
            attempt: slot.attempt,
            excluded: excluded.worker_id,
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
