// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router: the [`Router`] interface and the value it returns, a [`Plan`]
//! of [`Stage`]s, each holding at most one [`Booking`].
//!
//! A one-stage plan is today's `select_and_reserve`. A multistage plan is the
//! same value with more stages, each admitted through its own worker set's
//! queue with its own wait budget. The host owns the plan: it calls `schedule`
//! once per stage that is ready, forwards each booked stage, and records the
//! outcome. The router keeps no session.

mod booking;
mod plan;

#[cfg(test)]
mod plan_tests;

pub use booking::Booking;
pub use plan::{
    Budget, Constraint, DomainMode, Failure, Outcome, Plan, PlanError, PlanId, PlanState, Retry,
    SkipRule, Stage, StageAttempt, StageState, StageWork, When, WorkerFacts,
};

#[cfg(feature = "standalone-selection")]
pub use interface::Router;

#[cfg(feature = "standalone-selection")]
mod interface {
    use async_trait::async_trait;

    use super::Plan;
    use crate::services::selection::{
        SelectAndReserveRequest, SelectRequest, SelectResponse, SelectionError,
    };

    /// The router interface. `SelectionCore` implements it for one worker
    /// set; a multistage router implements it for one model by composing
    /// cores.
    ///
    /// The host owns the plan. `plan` builds it, `schedule` books into it,
    /// and an error leaves it with the host, booked stages intact, so the
    /// host decides between `abort`, a retry, or a fallback.
    #[async_trait]
    pub trait Router: Send + Sync {
        /// An advisory pick: no booking, no KV hint.
        async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError>;

        /// The plan for `req`, every stage pending. A one-set router's plan
        /// is one stage, today's `select_and_reserve`.
        fn plan(&self, req: &SelectAndReserveRequest) -> Result<Plan, SelectionError>;

        /// Book every stage of the plan that can be booked now: pending
        /// stages whose `when` is satisfied and whose constraints have a
        /// worker to read, including stages re-opened by `retry`. Returns
        /// `Ok` only when no stage of this router's set is still bookable;
        /// a stage it cannot book is an error, left pending, with stages
        /// already booked untouched.
        async fn schedule(
            &self,
            req: &SelectAndReserveRequest,
            plan: &mut Plan,
        ) -> Result<(), SelectionError>;
    }
}
