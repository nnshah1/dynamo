// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-direct hosting of stock engines.
//!
//! [`DirectEngines`] hosts [`LLMEngine`] implementations inside the frontend
//! process and serves them to `PushRouter` through a [`StreamingDispatch`], so
//! the router's final hop is the engine's own protocol (for example the vLLM
//! sidecar's gRPC client) instead of the Dynamo request plane. Each engine
//! registers in discovery as an ordinary worker under its own instance id, so
//! the model watcher, model manager and KV router treat it like any other
//! worker. Selection, occupancy, fault detection and migration stay in
//! `PushRouter` above this seam.

pub mod pods;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use dynamo_llm::model_type::ModelInput;
use dynamo_runtime::component::{Endpoint, Instance, TransportType};
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoverySpec, EndpointInstanceId};
use dynamo_runtime::pipeline::{
    AddressedRequest, AsyncEngine, AsyncEngineContextProvider, Error, ManyIn, ManyOut,
    ResponseStream, SingleIn, StreamingDispatch,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;
use parking_lot::RwLock;

use crate::adapter::EngineAdapter;
use crate::disagg::DisaggregationMode;
use crate::engine::{EngineConfig, LLMEngine, LLMEngineOutput, PreprocessedRequest};
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::worker::{
    WorkerConfig, build_local_model, err, resolve_model_type, resolve_worker_type_and_needs,
};

/// Engines hosted in this process and served on one endpoint.
pub struct DirectEngines {
    endpoint: Endpoint,
    mode: DisaggregationMode,
    dispatch: Arc<DirectDispatch>,
    next_index: AtomicU64,
    hosted: parking_lot::Mutex<HashMap<u64, Hosted>>,
}

/// What [`DirectEngines::remove`] undoes for one engine.
struct Hosted {
    engine: Arc<dyn LLMEngine>,
    /// Discovery records in registration order; withdrawn in reverse.
    registrations: Vec<DiscoveryInstance>,
}

impl DirectEngines {
    /// Hosts engines as workers of `endpoint`, all in the same disaggregation role.
    pub fn new(endpoint: Endpoint, mode: DisaggregationMode) -> Self {
        let dispatch = Arc::new(DirectDispatch::default());
        // The frontend's router builder takes this as the final hop for the
        // endpoint instead of the request plane.
        dynamo_llm::direct_dispatch::install(endpoint.id(), dispatch.clone());
        Self {
            endpoint,
            mode,
            dispatch,
            next_index: AtomicU64::new(0),
            hosted: Default::default(),
        }
    }

    /// The final hop for `PushRouter::from_client_with_dispatch`.
    pub fn dispatch(
        &self,
    ) -> Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>> {
        self.dispatch.clone()
    }

    /// Starts `engine`, registers it in discovery as a worker of this endpoint,
    /// and serves it through [`Self::dispatch`]. Returns its worker instance id.
    pub async fn add(&self, engine: Arc<dyn LLMEngine>) -> Result<u64, DynamoError> {
        self.add_as(engine, self.allocate_instance_id()).await
    }

    /// [`Self::add`] under a caller-chosen worker instance id, for engines whose
    /// identity comes from outside this process (a pod name, say).
    pub async fn add_as(
        &self,
        engine: Arc<dyn LLMEngine>,
        instance_id: u64,
    ) -> Result<u64, DynamoError> {
        if self.hosted.lock().contains_key(&instance_id) {
            return Err(err(
                ErrorType::Backend(BackendError::InvalidArgument),
                format!("direct engine {instance_id:x} is already hosted"),
            ));
        }
        let engine_config = match engine.start(instance_id).await {
            Ok(config) => config,
            Err(error) => {
                // `cleanup` runs after a failed `start` too: engines release
                // partial state there.
                if let Err(cleanup) = engine.cleanup().await {
                    tracing::warn!(%cleanup, "direct engine cleanup after failed start");
                }
                return Err(error);
            }
        };
        self.hosted.lock().insert(
            instance_id,
            Hosted {
                engine: engine.clone(),
                registrations: Vec::new(),
            },
        );
        match self.publish(&engine, &engine_config, instance_id).await {
            Ok(()) => {
                tracing::info!(
                    instance_id = format_args!("{instance_id:x}"),
                    model = %engine_config.model,
                    "direct engine registered"
                );
                Ok(instance_id)
            }
            Err(error) => {
                self.remove(instance_id).await;
                Err(error)
            }
        }
    }

    /// Withdraws a hosted engine: it leaves discovery and the dispatch, then
    /// is cleaned up. Returns whether it was hosted.
    pub async fn remove(&self, instance_id: u64) -> bool {
        let Some(hosted) = self.hosted.lock().remove(&instance_id) else {
            return false;
        };
        let discovery = self.endpoint.drt().discovery();
        for registration in hosted.registrations.into_iter().rev() {
            if let Err(error) = discovery.unregister(registration).await {
                tracing::warn!(
                    instance_id = format_args!("{instance_id:x}"),
                    %error,
                    "direct engine discovery unregister"
                );
            }
        }
        self.dispatch.engines.write().remove(&instance_id);
        if let Err(error) = hosted.engine.cleanup().await {
            tracing::warn!(
                instance_id = format_args!("{instance_id:x}"),
                %error,
                "direct engine cleanup"
            );
        }
        tracing::info!(
            instance_id = format_args!("{instance_id:x}"),
            "direct engine withdrawn"
        );
        true
    }

    /// Ids of the engines hosted now.
    pub fn instance_ids(&self) -> Vec<u64> {
        self.hosted.lock().keys().copied().collect()
    }

    /// Builds the worker's model card and registers the model and endpoint,
    /// recording each registration so [`Self::remove`] can withdraw it.
    async fn publish(
        &self,
        engine: &Arc<dyn LLMEngine>,
        engine_config: &EngineConfig,
        instance_id: u64,
    ) -> Result<(), DynamoError> {
        let endpoint_id = self.endpoint.id();
        let config = WorkerConfig {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
            model_name: engine_config.model.clone(),
            disaggregation_mode: self.mode,
            ..WorkerConfig::default()
        };
        let mut card = build_local_model(&config, engine_config, false)
            .await?
            .into_card();
        card.model_type = resolve_model_type(&config)?;
        card.model_input = ModelInput::Tokens;
        let (worker_type, needs) = resolve_worker_type_and_needs(&config);
        card.worker_type = Some(worker_type);
        card.needs = needs;
        let card_json: serde_json::Value = card
            .to_json()
            .and_then(|json| Ok(serde_json::from_str(&json)?))
            .map_err(|error| {
                err(
                    ErrorType::Backend(BackendError::Unknown),
                    format!("direct engine model card: {error}"),
                )
            })?;

        self.dispatch.engines.write().insert(
            instance_id,
            Arc::new(EngineAdapter::new(engine.clone(), self.mode)),
        );

        let specs = [
            DiscoverySpec::Model {
                namespace: endpoint_id.namespace.clone(),
                component: endpoint_id.component.clone(),
                endpoint: endpoint_id.name.clone(),
                card_json,
                model_suffix: None,
            },
            DiscoverySpec::Endpoint {
                namespace: endpoint_id.namespace.clone(),
                component: endpoint_id.component.clone(),
                endpoint: endpoint_id.name.clone(),
                // Routing never dials this address: the dispatch holds the engine.
                transport: TransportType::Tcp(format!("direct/{instance_id:x}")),
                device_type: None,
                request_plane_codec: None,
            },
        ];
        let discovery = self.endpoint.drt().discovery();
        for spec in specs {
            let registration = discovery
                .register_as(spec, instance_id)
                .await
                .map_err(|error| {
                    err(
                        ErrorType::Backend(BackendError::CannotConnect),
                        format!("direct engine discovery registration: {error:#}"),
                    )
                })?;
            let orphan = match self.hosted.lock().get_mut(&instance_id) {
                Some(hosted) => {
                    hosted.registrations.push(registration);
                    None
                }
                None => Some(registration),
            };
            // Removed while registering: withdraw what just landed.
            if let Some(registration) = orphan {
                if let Err(error) = discovery.unregister(registration).await {
                    tracing::warn!(%error, "direct engine discovery unregister");
                }
                return Err(err(
                    ErrorType::Backend(BackendError::Cancelled),
                    format!("direct engine {instance_id:x} was removed while registering"),
                ));
            }
        }
        Ok(())
    }

    /// Distinct from the process connection id and from every other engine in
    /// this process, so N engines on one frontend never collide in discovery.
    fn allocate_instance_id(&self) -> u64 {
        let index = self.next_index.fetch_add(1, Ordering::Relaxed) + 1;
        splitmix64(self.endpoint.drt().connection_id().wrapping_add(index))
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The engine behind each hosted worker instance id.
#[derive(Default)]
struct DirectDispatch {
    engines: RwLock<HashMap<u64, Arc<EngineAdapter>>>,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for DirectDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (request, address, instance) = addressed.into_parts();
        let instance_id = instance.map(|instance| instance.id()).ok_or_else(|| {
            unavailable(format!(
                "direct dispatch to {address} carries no selected instance"
            ))
        })?;
        let adapter = self
            .engines
            .read()
            .get(&instance_id)
            .cloned()
            .ok_or_else(|| {
                unavailable(format!("direct engine {instance_id:x} is not hosted here"))
            })?;

        let request_ctx = context.context();
        let stream = adapter
            .generate(context.map(|_| request))
            .await
            .map_err(|error| lift_transport_fault(error, request_ctx.is_stopped()))?;
        let engine_ctx = stream.context();
        let lifted = stream.map(move |item| {
            match item
                .err()
                .and_then(|error| lifted_error(error, request_ctx.is_stopped()))
            {
                Some(fault) => Annotated::from_err(fault),
                None => item,
            }
        });
        Ok(ResponseStream::new(Box::pin(lifted), engine_ctx))
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        Err(anyhow::anyhow!("direct engines serve unary requests only"))
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        self.engines.write().remove(&id.instance_id);
    }
}

fn unavailable(message: String) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::WorkerUnavailable)
        .message(message)
        .build()
        .into()
}

/// The engine client reports transport loss as a backend error, as every
/// engine does. On this seam that loss is the worker's transport failing, which
/// `PushRouter` recognises only as a top-level transport `ErrorType`.
///
/// A cancellation the caller never asked for is the same thing: the engine's
/// gRPC stream was cancelled underneath the request because its connection went
/// away, so it is reported as `Disconnected`. A cancellation the caller did ask
/// for stays what it is.
fn lifted_type(error_type: ErrorType, caller_cancelled: bool) -> Option<ErrorType> {
    match error_type {
        ErrorType::Backend(BackendError::CannotConnect) => Some(ErrorType::CannotConnect),
        ErrorType::Backend(BackendError::Disconnected) => Some(ErrorType::Disconnected),
        ErrorType::Backend(BackendError::ConnectionTimeout) => Some(ErrorType::ConnectionTimeout),
        ErrorType::Backend(BackendError::ResponseTimeout) => Some(ErrorType::ResponseTimeout),
        ErrorType::Backend(BackendError::Cancelled) if !caller_cancelled => {
            Some(ErrorType::Disconnected)
        }
        _ => None,
    }
}

fn lifted_error(error: DynamoError, caller_cancelled: bool) -> Option<DynamoError> {
    lifted_type(error.error_type(), caller_cancelled).map(|error_type| {
        DynamoError::builder()
            .error_type(error_type)
            .message(format!("direct engine transport fault: {error}"))
            .cause(error)
            .build()
    })
}

fn lift_transport_fault(error: Error, caller_cancelled: bool) -> Error {
    match error
        .downcast_ref::<DynamoError>()
        .cloned()
        .and_then(|error| lifted_error(error, caller_cancelled))
    {
        Some(lifted) => lifted.into(),
        None => error,
    }
}
