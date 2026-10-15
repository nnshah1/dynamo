// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's edge of the Plan-returning [`Router`]: the request it
//! schedules with, and the error it hands back. The frontend schedules a plan
//! on its worker sets' selection cores and dispatches from the plan; see
//! `prefill_router::planned`.

use anyhow::Error;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::scheduling::KvSchedulerError;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionError, SelectionSessionContext,
};
use dynamo_runtime::pipeline::SingleIn;

use crate::protocols::common::llm_backend::PreprocessedRequest;

/// The plan-level request for a frontend request: the prompt the selector
/// hashes, the caller's constraints, and the class the scheduler queues under.
pub fn routing_request(
    request: &SingleIn<PreprocessedRequest>,
    partition: &RoutingPartitionId,
    policy_class: Option<String>,
    is_eagle: bool,
) -> SelectAndReserveRequest {
    let body = request.content();
    let (token_ids, block_mm_infos) = body.block_mm_routing_info();
    let routing = body.routing.as_ref();
    SelectAndReserveRequest {
        model_name: partition.model_name.clone(),
        routing_group: partition.routing_group.clone(),
        selection_id: Some(request.id().to_string()),
        prompt: PromptRequest {
            token_ids: Some(token_ids.to_vec()),
            block_mm_infos: block_mm_infos.map(<[_]>::to_vec),
            lora_name: routing.and_then(|hints| hints.lora_name.clone()),
            cache_namespace: routing.and_then(|hints| hints.cache_namespace.clone()),
            is_eagle: Some(is_eagle),
            ..PromptRequest::default()
        },
        router_config_override: body.router_config_override.clone(),
        expected_output_tokens: routing.and_then(|hints| hints.expected_output_tokens),
        priority_jump: routing.and_then(|hints| hints.priority_jump),
        strict_priority: routing.and_then(|hints| hints.strict_priority),
        session_id: None,
        // The agent's session, parent, finality and trigger, as the host's
        // own admission hands them to the worker-selection policy.
        session_context: body
            .agent_context
            .as_ref()
            .map(crate::kv_router::to_worker_selection_session_context)
            .map(SelectionSessionContext::from),
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: routing.and_then(|hints| hints.allowed_worker_ids.clone()),
        routing_constraints: routing
            .and_then(|hints| hints.routing_constraints.clone())
            .unwrap_or_default(),
        policy_class,
        all_now: false,
        export_bookings: false,
    }
}

/// A Router error at the frontend's exit: a scheduler answer goes back through
/// the host's one classification (type, queue-deadline reason, overload
/// cause), so the HTTP status and metrics are those of a direct admission.
pub(crate) fn frontend_error(error: SelectionError) -> Error {
    use dynamo_runtime::error::{DynamoError, ErrorType};
    let classified = |error_type: ErrorType, message: String| -> Error {
        DynamoError::builder()
            .error_type(error_type)
            .message(message)
            .build()
            .into()
    };
    match error {
        // The raw payload again, as a direct admission reports it.
        SelectionError::Scheduler(KvSchedulerError::QueueRejected(rejection)) => rejection.into(),
        SelectionError::Scheduler(scheduler) => crate::kv_router::map_scheduler_error(scheduler),
        SelectionError::Cancelled(message) => classified(ErrorType::Cancelled, message),
        SelectionError::BadRequest(message) => classified(ErrorType::InvalidArgument, message),
        other => other.into(),
    }
}
