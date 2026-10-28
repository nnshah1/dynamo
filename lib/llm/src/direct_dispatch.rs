// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process final hops for the frontend's token router.
//!
//! A process that hosts engines itself (frontend-direct serving) installs a
//! [`StreamingDispatch`] for the endpoint those engines register under. When
//! the frontend builds the router for that endpoint it takes this dispatch as
//! the final hop instead of the request plane, so requests reach the hosted
//! engines in-process while selection, fault detection and migration stay in
//! `PushRouter`.
//!
//! Keyed by endpoint id: the same key discovery uses for the workers, so one
//! process can host engines for several endpoints.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use dynamo_runtime::pipeline::StreamingDispatch;
use dynamo_runtime::protocols::EndpointId;

use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use crate::types::Annotated;

/// The router's final hop type for token-in/token-out workers.
pub type PreprocessedDispatch =
    Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>;

static DISPATCHES: LazyLock<RwLock<HashMap<EndpointId, PreprocessedDispatch>>> =
    LazyLock::new(Default::default);

/// Serve `endpoint`'s workers through `dispatch` in this process. Replaces an
/// earlier installation for the same endpoint.
pub fn install(endpoint: EndpointId, dispatch: PreprocessedDispatch) {
    DISPATCHES
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(endpoint, dispatch);
}

/// Stop serving `endpoint` in-process; routers built afterwards use the request plane.
pub fn uninstall(endpoint: &EndpointId) -> Option<PreprocessedDispatch> {
    DISPATCHES
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(endpoint)
}

/// The in-process final hop for `endpoint`, if one is installed.
pub fn lookup(endpoint: &EndpointId) -> Option<PreprocessedDispatch> {
    DISPATCHES
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(endpoint)
        .cloned()
}
