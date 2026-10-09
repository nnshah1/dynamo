// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Booking`: capacity held on one worker for one stage of one request.
//!
//! A `Lease` admission hands the host an armed [`BookingHandle`] whose drop
//! frees the booking. A `Book` admission, or a host that called
//! [`BookingHandle::commit`], leaves a [`SchedulerBookingDescriptor`] whose
//! release belongs to the core's index row or the wire owner. The id is the
//! wire's `selection_id` in both forms.

use std::fmt;
#[cfg(any(test, feature = "testing"))]
use std::sync::Arc;
#[cfg(any(test, feature = "testing"))]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::protocols::WorkerWithDpRank;
use crate::scheduling::queue::{BookingHandle, SchedulerBookingDescriptor};
use crate::sequences::SequenceError;

#[must_use = "dropping an owned booking releases it; call `release` to wait for the scheduler"]
pub enum Booking {
    /// The plan owns the release: dropping it frees the booking.
    Owned(BookingHandle),
    /// Someone else owns the release: the core's reservation index, or the
    /// host that committed the handle.
    Committed(SchedulerBookingDescriptor),
    /// No scheduler behind it; releases are counted.
    #[cfg(any(test, feature = "testing"))]
    Scripted(ScriptedBooking),
}

impl Booking {
    #[cfg(any(test, feature = "testing"))]
    pub fn scripted(
        id: impl Into<String>,
        worker: WorkerWithDpRank,
        releases: Arc<AtomicUsize>,
    ) -> Self {
        Self::Scripted(ScriptedBooking {
            id: id.into(),
            worker,
            releases,
            is_released: false,
        })
    }

    /// The wire's `selection_id`.
    pub fn id(&self) -> &str {
        match self {
            Self::Owned(handle) => &handle.descriptor().request_id,
            Self::Committed(descriptor) => &descriptor.request_id,
            #[cfg(any(test, feature = "testing"))]
            Self::Scripted(booking) => &booking.id,
        }
    }

    pub fn worker(&self) -> WorkerWithDpRank {
        match self {
            Self::Owned(handle) => handle.descriptor().worker,
            Self::Committed(descriptor) => descriptor.worker,
            #[cfg(any(test, feature = "testing"))]
            Self::Scripted(booking) => booking.worker,
        }
    }

    /// Whether dropping this booking frees it.
    pub fn is_owned(&self) -> bool {
        match self {
            Self::Owned(_) => true,
            Self::Committed(_) => false,
            #[cfg(any(test, feature = "testing"))]
            Self::Scripted(_) => true,
        }
    }

    /// Release now and wait for the scheduler to acknowledge. A committed
    /// booking has nothing to release here; its owner frees it.
    pub async fn release(self) -> Result<(), SequenceError> {
        match self {
            Self::Owned(handle) => handle.release().await,
            Self::Committed(_) => Ok(()),
            #[cfg(any(test, feature = "testing"))]
            Self::Scripted(mut booking) => {
                booking.release_once();
                Ok(())
            }
        }
    }
}

impl fmt::Debug for Booking {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Booking")
            .field("id", &self.id())
            .field("worker", &self.worker())
            .field("is_owned", &self.is_owned())
            .finish()
    }
}

#[cfg(any(test, feature = "testing"))]
pub struct ScriptedBooking {
    id: String,
    worker: WorkerWithDpRank,
    releases: Arc<AtomicUsize>,
    is_released: bool,
}

#[cfg(any(test, feature = "testing"))]
impl ScriptedBooking {
    fn release_once(&mut self) {
        if !self.is_released {
            self.is_released = true;
            self.releases.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[cfg(any(test, feature = "testing"))]
impl Drop for ScriptedBooking {
    fn drop(&mut self) {
        self.release_once();
    }
}
