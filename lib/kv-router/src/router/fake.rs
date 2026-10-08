// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An in-memory one-set [`Router`] for multistage tests here and in dependent
//! crates (behind the `testing` feature). It selects deterministically
//! (lowest projected decode load, then lowest worker id), records every
//! preview, booking and release, and lets a test inject failures or hold
//! admissions open to exercise cancellation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::watch;

use crate::WorkerType;
use crate::identity::RoutingPartitionId;
use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::scheduling::KvSchedulerError;
use crate::services::overlap::MooncakeOverlapSummary;
use crate::services::selection::{
    SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionError, SelectionWorkerLoad,
};

use super::booking::Booking;
use super::plan::{Budget, Constraint, Plan, PlanId, Stage, WorkerFacts};
use super::router_trait::Router;

/// Projected load and cache state a fake worker reports.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FakeSignals {
    pub cached_tokens: usize,
    pub potential_decode_blocks: u64,
    pub total_kv_blocks: Option<u64>,
    pub active_prefill_tokens: usize,
    pub prefill_token_capacity: usize,
}

#[derive(Debug, Clone)]
pub struct FakeWorker {
    pub worker: WorkerWithDpRank,
    pub facts: WorkerFacts,
    pub signals: FakeSignals,
}

impl FakeWorker {
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            worker: WorkerWithDpRank::new(worker_id, 0),
            facts: WorkerFacts::default(),
            signals: FakeSignals {
                prefill_token_capacity: 1000,
                ..FakeSignals::default()
            },
        }
    }

    pub fn with_facts(mut self, facts: WorkerFacts) -> Self {
        self.facts = facts;
        self
    }

    pub fn with_signals(mut self, signals: FakeSignals) -> Self {
        self.signals = signals;
        self
    }
}

/// What the fake observed, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FakeEvent {
    Preview {
        worker: WorkerWithDpRank,
    },
    Book {
        stage: usize,
        booking_id: String,
        worker: WorkerWithDpRank,
    },
}

/// Busy thresholds the fake evaluates like a core does from its config.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeThresholds {
    pub prefill_busy: Option<f64>,
    pub decode_busy: Option<f64>,
}

pub struct FakeRouter {
    set: WorkerType,
    block_size: u32,
    thresholds: FakeThresholds,
    workers: Mutex<Vec<FakeWorker>>,
    events: Mutex<Vec<FakeEvent>>,
    fail_next: Mutex<VecDeque<SelectionError>>,
    admission_open: watch::Sender<bool>,
    releases: Mutex<HashMap<String, Arc<AtomicUsize>>>,
    /// A worker that leaves right after it is previewed: the preview is
    /// stale by the time the booking is made.
    leaves_after_preview: Mutex<Option<WorkerId>>,
    can_select: bool,
}

impl FakeRouter {
    pub fn new(set: WorkerType) -> Arc<Self> {
        let (admission_open, _) = watch::channel(true);
        Arc::new(Self {
            set,
            block_size: 1,
            thresholds: FakeThresholds::default(),
            workers: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            fail_next: Mutex::new(VecDeque::new()),
            admission_open,
            releases: Mutex::new(HashMap::new()),
            leaves_after_preview: Mutex::new(None),
            can_select: true,
        })
    }

    /// A router over `set` with one single-rank worker per id.
    pub fn with_workers(
        set: WorkerType,
        worker_ids: impl IntoIterator<Item = WorkerId>,
    ) -> Arc<Self> {
        let router = Self::new(set);
        for worker_id in worker_ids {
            router.add_worker(FakeWorker::new(worker_id));
        }
        router
    }

    /// A set whose router cannot select advisorily (an encoder pool chosen
    /// round-robin, say).
    pub fn without_select(
        set: WorkerType,
        worker_ids: impl IntoIterator<Item = WorkerId>,
    ) -> Arc<Self> {
        let mut router = Arc::into_inner(Self::with_workers(set, worker_ids))
            .expect("fresh router has one owner");
        router.can_select = false;
        Arc::new(router)
    }

    pub fn with_thresholds(self: Arc<Self>, thresholds: FakeThresholds) -> Arc<Self> {
        let mut router = Arc::into_inner(self).expect("router has one owner");
        router.thresholds = thresholds;
        Arc::new(router)
    }

    pub fn set(&self) -> WorkerType {
        self.set
    }

    pub fn add_worker(&self, worker: FakeWorker) {
        self.workers.lock().push(worker);
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.workers
            .lock()
            .retain(|worker| worker.worker.worker_id != worker_id);
    }

    pub fn set_signals(&self, worker_id: WorkerId, signals: FakeSignals) {
        for worker in self.workers.lock().iter_mut() {
            if worker.worker.worker_id == worker_id {
                worker.signals = signals;
            }
        }
    }

    /// The next booking fails with `error`.
    pub fn fail_next(&self, error: SelectionError) {
        self.fail_next.lock().push_back(error);
    }

    /// While closed, bookings wait on their budget; dropping a waiting
    /// `schedule` future is a cancellation before any booking exists.
    pub fn set_admission_open(&self, open: bool) {
        self.admission_open.send_replace(open);
    }

    pub fn leaves_after_preview(&self, worker_id: WorkerId) {
        *self.leaves_after_preview.lock() = Some(worker_id);
    }

    pub fn events(&self) -> Vec<FakeEvent> {
        self.events.lock().clone()
    }

    pub fn clear_events(&self) {
        self.events.lock().clear();
    }

    /// Booking ids made and not yet released.
    pub fn outstanding(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .releases
            .lock()
            .iter()
            .filter(|(_, releases)| releases.load(Ordering::SeqCst) == 0)
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        ids
    }

    pub fn release_count(&self, booking_id: &str) -> usize {
        self.releases
            .lock()
            .get(booking_id)
            .map_or(0, |releases| releases.load(Ordering::SeqCst))
    }

    /// Filters on pins, the allow set and required taints; preferred taints
    /// are not scored here.
    fn choose(
        &self,
        pinned: Option<WorkerWithDpRank>,
        allowed: Option<&HashSet<WorkerId>>,
        required_taints: &HashSet<String>,
    ) -> Result<FakeWorker, SelectionError> {
        let workers = self.workers.lock();
        let mut candidates: Vec<&FakeWorker> = workers
            .iter()
            .filter(|candidate| pinned.is_none_or(|pinned| candidate.worker == pinned))
            .filter(|candidate| {
                allowed.is_none_or(|allowed| allowed.contains(&candidate.worker.worker_id))
            })
            .filter(|candidate| required_taints.is_subset(&candidate.facts.taints))
            .collect();
        if candidates.is_empty() {
            return Err(SelectionError::Scheduler(
                KvSchedulerError::AllEligibleWorkersFiltered,
            ));
        }
        candidates.sort_by_key(|candidate| {
            (
                candidate.signals.potential_decode_blocks,
                candidate.worker.worker_id,
                candidate.worker.dp_rank,
            )
        });
        Ok(candidates[0].clone())
    }

    fn response(
        &self,
        req_model: &str,
        req_group: &str,
        prompt_tokens: usize,
        worker: &FakeWorker,
    ) -> SelectResponse {
        let signals = worker.signals;
        let decode_busy = self
            .thresholds
            .decode_busy
            .zip(signals.total_kv_blocks)
            .map(|(threshold, total)| {
                signals.potential_decode_blocks as f64 > threshold * total as f64
            });
        let prefill_busy = self.thresholds.prefill_busy.map(|threshold| {
            signals.active_prefill_tokens as f64 > threshold * signals.prefill_token_capacity as f64
        });
        SelectResponse {
            selection_id: None,
            sequence_hashes: None,
            isl_tokens: None,
            track_prefill_tokens: None,
            model_name: req_model.to_string(),
            routing_group: req_group.to_string(),
            worker_id: worker.worker.worker_id,
            dp_rank: worker.worker.dp_rank,
            endpoint: format!("fake://{}", worker.worker.worker_id),
            block_size: self.block_size,
            overlap: MooncakeOverlapSummary {
                longest_matched: signals.cached_tokens as u32 / self.block_size,
                ..MooncakeOverlapSummary::default()
            },
            effective_prefill_tokens: prompt_tokens.saturating_sub(signals.cached_tokens),
            potential_decode_blocks: signals.potential_decode_blocks,
            decode_busy,
            worker_load: Some(SelectionWorkerLoad {
                active_prefill_tokens: signals.active_prefill_tokens,
                prefill_token_capacity: signals.prefill_token_capacity,
                total_kv_blocks: signals.total_kv_blocks,
                prefill_busy,
            }),
            kv_hint: None,
        }
    }

    async fn wait_for_admission(&self, budget: Budget) -> Result<(), SelectionError> {
        let mut open = self.admission_open.subscribe();
        if *open.borrow_and_update() {
            return Ok(());
        }
        let deadline_exceeded = || SelectionError::Scheduler(KvSchedulerError::DeadlineExceeded);
        match budget {
            Budget::Immediate => Err(deadline_exceeded()),
            Budget::Full => {
                while !*open.borrow_and_update() {
                    open.changed()
                        .await
                        .map_err(|_| SelectionError::Internal("fake router dropped".to_string()))?;
                }
                Ok(())
            }
            Budget::Bounded(budget) => tokio::time::timeout(budget, async {
                while !*open.borrow_and_update() {
                    open.changed()
                        .await
                        .map_err(|_| SelectionError::Internal("fake router dropped".to_string()))?;
                }
                Ok(())
            })
            .await
            .unwrap_or_else(|_| Err(deadline_exceeded())),
        }
    }
}

#[async_trait]
impl Router for FakeRouter {
    async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        if !self.can_select {
            return Err(SelectionError::BadRequest(format!(
                "the {} set does not support advisory selection",
                self.set
            )));
        }
        let worker = self.choose(
            req.pinned_worker,
            req.allowed_worker_ids.as_ref(),
            &req.routing_constraints.required_taints,
        )?;
        self.events.lock().push(FakeEvent::Preview {
            worker: worker.worker,
        });
        if self.leaves_after_preview.lock().take() == Some(worker.worker.worker_id) {
            self.remove_worker(worker.worker.worker_id);
        }
        let prompt_tokens = req
            .prompt
            .isl_tokens
            .or_else(|| req.prompt.token_ids.as_ref().map(Vec::len))
            .unwrap_or(0);
        Ok(self.response(&req.model_name, &req.routing_group, prompt_tokens, &worker))
    }

    fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError> {
        let id = req
            .selection_id
            .clone()
            .unwrap_or_else(|| "fake".to_string());
        Plan::new(
            PlanId::from(id),
            RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone()),
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
            let stage = plan.stage(k).cloned().expect("schedulable stage exists");
            self.wait_for_admission(stage.wait).await?;
            if let Some(error) = self.fail_next.lock().pop_front() {
                return Err(error);
            }
            let mut pinned = req.pinned_worker;
            let mut allowed = req.allowed_worker_ids.clone();
            for constraint in &stage.constraints {
                match constraint {
                    Constraint::Pin(worker) | Constraint::Previewed(worker) => {
                        pinned = Some(*worker)
                    }
                    Constraint::Exclude(worker_id) => {
                        let universe: HashSet<WorkerId> = allowed.take().unwrap_or_else(|| {
                            self.workers
                                .lock()
                                .iter()
                                .map(|w| w.worker.worker_id)
                                .collect()
                        });
                        allowed = Some(universe.into_iter().filter(|id| id != worker_id).collect());
                    }
                    Constraint::TransferCompatible(_) | Constraint::SameDomain { .. } => {}
                }
            }
            let mut required = req.routing_constraints.required_taints.clone();
            required.extend(
                plan.routing_constraints(k)
                    .map_err(|error| SelectionError::Conflict(error.to_string()))?
                    .required_taints,
            );
            let worker = self.choose(pinned, allowed.as_ref(), &required)?;
            plan.check_placement(k, &worker.facts)
                .map_err(|error| SelectionError::Conflict(error.to_string()))?;
            let attempt = plan.attempt(k).expect("stage exists");
            // The same id scheme as `SelectionCore`'s implementation.
            let booking_id =
                if plan.stage_count() == 1 && attempt == super::plan::StageAttempt::FIRST {
                    plan.id().to_string()
                } else {
                    format!("{}/{k}/{attempt}", plan.id())
                };
            let releases = Arc::new(AtomicUsize::new(0));
            self.releases
                .lock()
                .insert(booking_id.clone(), Arc::clone(&releases));
            self.events.lock().push(FakeEvent::Book {
                stage: k,
                booking_id: booking_id.clone(),
                worker: worker.worker,
            });
            plan.book(
                k,
                Booking::scripted(booking_id, worker.worker, releases),
                worker.facts.clone(),
                None,
            )
            .map_err(|error| SelectionError::Internal(error.to_string()))?;
        }
        Ok(())
    }
}
