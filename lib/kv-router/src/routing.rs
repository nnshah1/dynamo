// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The value a Router returns: a [`Plan`] of [`Stage`]s, each holding at most
//! one [`Booking`].
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
    SkipRule, Stage, StageAttempt, StageState, When, WorkerFacts,
};
