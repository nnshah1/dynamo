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

use std::time::Duration;

use crate::WorkerType;
use crate::identity::RoutingPartitionId;
use std::sync::Arc;

use crate::kv_hints::KvHint;
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::scheduling::SchedulingResponse;
use crate::scheduling::config::RouterConfigOverride;
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
/// worker, so they may only name earlier stages.
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
    fn reads(&self) -> Option<usize> {
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

impl StageWork {
    /// What a booking for this work is charged for, in the scheduler's
    /// request-level terms: prefill tracking follows the work, a
    /// prefill-only stage projects one output token, a decode-only stage
    /// assumes no KV reuse. This replaces the request's own override for
    /// those fields, so every host charges a stage the same way the core
    /// does.
    pub fn apply_to(
        self,
        config: &mut RouterConfigOverride,
        expected_output_tokens: &mut Option<u32>,
    ) {
        config.track_prefill_tokens = Some(matches!(
            self,
            StageWork::PrefillAndDecode | StageWork::PrefillOnly
        ));
        match self {
            StageWork::PrefillOnly => *expected_output_tokens = Some(1),
            StageWork::DecodeOnly => config.assume_kv_reuse = Some(false),
            StageWork::PrefillAndDecode | StageWork::None => {}
        }
    }

    /// Whether a booking for this work holds the prompt's KV blocks.
    pub fn tracks_active_blocks(self) -> bool {
        self != StageWork::None
    }
}

/// How a stage's set scores its candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StageScoring {
    /// The cache credit the request or the set configured applies.
    CacheAware,
    /// Load alone: the prompt's KV comes from elsewhere, so a candidate's
    /// cache of it earns no credit.
    LoadOnly,
}

impl StageScoring {
    /// The scorer's cache credit for a stage, in the scheduler's
    /// request-level terms. `LoadOnly` replaces the request's own override
    /// for that field, as [`StageWork::apply_to`] replaces the accounting
    /// fields; `CacheAware` leaves the request's and the set's choice alone.
    pub fn apply_to(self, config: &mut RouterConfigOverride) {
        if self == StageScoring::LoadOnly {
            config.overlap_score_credit = Some(0.0);
        }
    }
}

/// When the router skips a stage instead of booking it.
#[derive(Debug, Clone, PartialEq)]
pub enum SkipRule {
    /// No multimodal input (an encode stage).
    NoMultimodal,
    /// The conditional-disaggregation policy and decode-busy gate.
    ConditionalDisagg,
}

/// The booked worker's configuration as its host knows it, captured at
/// booking so a later stage's placement rules (taints, topology domains,
/// KV-transfer domain) can still read it if the worker has left discovery.
pub type WorkerConfig = Arc<dyn WorkerConfigLike + Send + Sync>;

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
    /// Finished on this worker.
    Completed(WorkerWithDpRank),
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
    /// The worker set this stage books into when more than one set of this
    /// type is routed (a rolling update's new generation, a canary, a second
    /// hardware pool). `None` is the one unnamed set of the type.
    pub pool: Option<String>,
    pub when: When,
    /// Stages whose outcome this one needs before it is forwarded.
    pub inputs: Vec<usize>,
    pub wait: Budget,
    pub constraints: Vec<Constraint>,
    pub skip: Option<SkipRule>,
    /// Overrides the accounting derived from the set; see [`Plan::work_of`].
    ///
    /// Precedence for what a stage says (`work`, `scoring`): the stage list
    /// wins over the request's own `router_config_override`, which wins over
    /// the set's configuration. A stage that says nothing, and the derived
    /// defaults that amount to "as configured", leave the request's and the
    /// set's choice alone.
    pub work: Option<StageWork>,
    /// Overrides the scoring derived from the work; see [`Plan::scoring_of`].
    pub scoring: Option<StageScoring>,
}

impl Stage {
    pub fn new(set: WorkerType) -> Self {
        Self {
            set,
            pool: None,
            when: When::Now,
            inputs: Vec::new(),
            wait: Budget::Full,
            constraints: Vec::new(),
            skip: None,
            work: None,
            scoring: None,
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

struct Slot {
    stage: Stage,
    state: StageState,
    booking: Option<Booking>,
    config: Option<WorkerConfig>,
    /// The routing hint the scheduler attached to the booking; the host
    /// forwards it with the request.
    kv_hint: Option<KvHint>,
    /// The scheduler's answer behind the booking (overlap, cached tokens,
    /// projected load), for the host's tracking and a policy's observations.
    scheduling: Option<SchedulingResponse>,
    attempt: StageAttempt,
    /// Attempts the host reported failed and retried; a limit counts these,
    /// not re-placements caused by another stage's failure.
    failures: u32,
    /// The worker the last attempt was booked on; `retry` excludes it.
    last_worker: Option<WorkerWithDpRank>,
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("stage", &self.stage)
            .field("state", &self.state)
            .field("booking", &self.booking)
            .field("has_config", &self.config.is_some())
            .field("kv_hint", &self.kv_hint)
            .field("scheduling", &self.scheduling)
            .field("attempt", &self.attempt)
            .field("failures", &self.failures)
            .field("last_worker", &self.last_worker)
            .finish()
    }
}

impl Slot {
    fn new(stage: Stage) -> Self {
        Self {
            stage,
            state: StageState::Pending,
            booking: None,
            config: None,
            kv_hint: None,
            scheduling: None,
            attempt: StageAttempt::FIRST,
            failures: 0,
            last_worker: None,
        }
    }

    fn reset(&mut self, state: StageState) -> Option<Booking> {
        self.state = state;
        self.config = None;
        self.kv_hint = None;
        self.scheduling = None;
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

    /// The booked worker's configuration, for a later stage's placement rules.
    pub fn worker_config(&self, k: usize) -> Option<&(dyn WorkerConfigLike + Send + Sync)> {
        self.slots.get(k)?.config.as_deref()
    }

    /// The routing hint the scheduler attached to stage `k`'s booking.
    pub fn kv_hint(&self, k: usize) -> Option<&KvHint> {
        self.slots.get(k)?.kv_hint.as_ref()
    }

    /// The scheduler's answer behind stage `k`'s booking, when a scheduler
    /// made it: what it saw of the worker's cache and load at admission.
    pub fn scheduling(&self, k: usize) -> Option<&SchedulingResponse> {
        self.slots.get(k)?.scheduling.as_ref()
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

    pub fn stages(&self) -> impl Iterator<Item = &Stage> {
        self.slots.iter().map(|slot| &slot.stage)
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

    /// How stage `k`'s set scores its candidates: the stage's own `scoring`,
    /// else by work. A stage that only decodes receives its prompt's KV from
    /// the prefill stage, so it is placed on load alone; every other stage
    /// keeps its set's configured cache credit.
    pub fn scoring_of(&self, k: usize) -> Option<StageScoring> {
        let slot = self.slots.get(k)?;
        if let Some(scoring) = slot.stage.scoring {
            return Some(scoring);
        }
        Some(match self.work_of(k)? {
            StageWork::DecodeOnly => StageScoring::LoadOnly,
            StageWork::PrefillAndDecode | StageWork::PrefillOnly | StageWork::None => {
                StageScoring::CacheAware
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
                    // A placement read wants the peer's worker and facts, which
                    // a stage keeps while it is booked or running whether the
                    // scheduler, the host or nobody holds its booking.
                    self.slots[j].config.is_some() || self.slots[j].state == StageState::Skipped
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

    /// Record the booking the router made for stage `k`, the worker's
    /// configuration a later stage's placement rules read, the routing hint
    /// the host forwards with the request, and the scheduler's answer behind
    /// the booking. `Pin`
    /// and `Exclude` are enforced here. A rejected booking drops and
    /// releases itself.
    pub fn book(
        &mut self,
        k: usize,
        booking: Booking,
        config: WorkerConfig,
        kv_hint: Option<KvHint>,
        scheduling: Option<SchedulingResponse>,
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
        slot.config = Some(config);
        slot.kv_hint = kv_hint;
        slot.scheduling = scheduling;
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
    /// worker and its configuration with no booking behind them.
    pub fn book_untracked(
        &mut self,
        k: usize,
        worker: WorkerWithDpRank,
        config: WorkerConfig,
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
        slot.config = Some(config);
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
    pub fn mark_dispatched(&mut self, k: usize) -> Result<StageAttempt, PlanError> {
        self.check_index(k)?;
        if self.slots[k].state != StageState::Booked {
            return Err(self.invalid_transition(k, "mark dispatched"));
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
    pub fn mark_handed_off(&mut self, k: usize, attempt: StageAttempt) -> Result<(), PlanError> {
        self.check_index(k)?;
        self.check_attempt(k, attempt)?;
        match &self.slots[k].state {
            StageState::Dispatched(_) => {}
            StageState::HandedOff(_) => return Ok(()),
            _ => return Err(self.invalid_transition(k, "mark handed off")),
        }
        self.slots[k].state = StageState::HandedOff(attempt);
        Ok(())
    }

    /// Stage `k`'s attempt finished on `worker`. The booking stays held
    /// until `release`. Idempotent for the current attempt.
    pub fn mark_completed(
        &mut self,
        k: usize,
        attempt: StageAttempt,
        worker: WorkerWithDpRank,
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
                return Err(self.invalid_transition(k, "mark completed"));
            }
        }
        debug_assert_eq!(self.slots[k].last_worker, Some(worker));
        self.slots[k].state = StageState::Completed(worker);
        Ok(())
    }

    /// Stage `k`'s attempt failed, or the host declined to dispatch a booked
    /// stage. Its booking is released and every booked stage that reads it,
    /// transitively, falls back to `Pending`. Idempotent for the current
    /// attempt.
    pub fn mark_failed(
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
                return Err(self.invalid_transition(k, "mark failed"));
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
