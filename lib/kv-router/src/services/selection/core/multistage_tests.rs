// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `MultiStageRouter` over real prefill and decode cores: what each stage's
//! booking is charged for, on the remote-prefill and the bypass paths.

use std::sync::Arc;

use super::tests::{core_with, reserve_request, test_config, wait_until, worker};
use super::*;
use crate::router::{ClassTable, MultiStageRouter, Router, StageList, StageState};

fn typed_core(worker_type: WorkerType) -> Arc<SelectionCore> {
    Arc::new(core_with(
        test_config(false),
        SelectionHost::default(),
        None,
        worker_type,
        None,
    ))
}

fn prefill_tokens(core: &SelectionCore) -> Option<usize> {
    core.loads(Some("model"), Some("default"))
        .first()
        .and_then(|model| {
            model
                .loads
                .first()
                .map(|load| load.potential_prefill_tokens)
        })
}

async fn ready(core: &SelectionCore) {
    wait_until("slot tracker sees the worker", || {
        prefill_tokens(core).is_some()
    })
    .await;
}

#[tokio::test]
async fn remote_decode_is_not_charged_for_the_prompt_but_a_bypass_decode_is() {
    let prefill = typed_core(WorkerType::Prefill);
    let decode = typed_core(WorkerType::Decode);
    prefill.upsert_worker(worker(11)).await.unwrap();
    decode.upsert_worker(worker(21)).await.unwrap();
    ready(&prefill).await;
    ready(&decode).await;
    let router = MultiStageRouter::builder()
        .set(WorkerType::Prefill, prefill.clone())
        .set(WorkerType::Decode, decode.clone())
        .classes(ClassTable::new(StageList::prefill_decode()))
        .build()
        .unwrap();

    // Remote prefill: the prompt is prefill's load, not decode's.
    let req = reserve_request("pd");
    let mut plan = router.plan(&req).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(state(&plan, 0), &StageState::Booked);
    assert_eq!(state(&plan, 1), &StageState::Booked);
    assert_eq!(prefill_tokens(&prefill), Some(4));
    assert_eq!(
        prefill_tokens(&decode),
        Some(0),
        "a remote-prefill decode holds no prompt load"
    );
    plan.abort();
    wait_until("release", || prefill_tokens(&prefill) == Some(0)).await;

    // Prefill skipped (a bypass): decode does the prefill and is charged.
    let req = reserve_request("bypass");
    let mut plan = router.plan(&req).unwrap();
    plan.skip(0).unwrap();
    router.schedule(&req, &mut plan).await.unwrap();
    assert_eq!(
        prefill_tokens(&decode),
        Some(4),
        "a local-prefill decode is charged for the prompt"
    );
    assert_eq!(prefill_tokens(&prefill), Some(0));
    plan.abort();
}

fn state(plan: &crate::router::Plan, k: usize) -> &StageState {
    plan.state_of(k).expect("stage")
}
