// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-stage placement: the routing constraints a stage derives from the
//! workers booked before it, as the topology taints the selector already
//! filters and scores on.

use crate::protocols::{KvTransferEnforcement, RoutingConstraints, WorkerConfigLike};

use super::plan::{Constraint, DomainMode, Plan, PlanError, WorkerConfig};

/// Prefix of the taint a worker publishes for each topology domain; the same
/// one the frontend's KV-transfer constraints use.
pub const TOPOLOGY_TAINT_PREFIX: &str = "dynamo.topology/";

pub fn topology_taint(domain: &str, value: &str) -> String {
    format!("{TOPOLOGY_TAINT_PREFIX}{domain}={value}")
}

enum Derived {
    Required(String),
    Preferred(String, f32),
    None,
}

impl Plan {
    /// The constraints stage `k` inherits from the stages its rules read.
    /// Required taints union; preferred weights add. A rule that reads a
    /// skipped stage derives nothing.
    pub fn routing_constraints(&self, k: usize) -> Result<RoutingConstraints, PlanError> {
        let mut constraints = RoutingConstraints::default();
        let Some(stage) = self.stage(k) else {
            return Err(PlanError::NoSuchStage { stage: k });
        };
        for constraint in &stage.constraints {
            let derived = match constraint {
                Constraint::TransferCompatible(j) => match self.worker_config(*j) {
                    Some(config) => transfer_constraint(config, k, *j)?,
                    None => Derived::None,
                },
                Constraint::SameDomain {
                    stage: j,
                    key,
                    mode,
                } => match self.worker_config(*j) {
                    Some(config) => same_domain_constraint(config, key, *mode, k, *j)?,
                    None => Derived::None,
                },
                Constraint::Pin(_) | Constraint::Previewed(_) | Constraint::Exclude(_) => {
                    Derived::None
                }
            };
            match derived {
                Derived::Required(taint) => {
                    constraints.required_taints.insert(taint);
                }
                Derived::Preferred(taint, weight) => {
                    *constraints.preferred_taints.entry(taint).or_insert(0.0) += weight;
                }
                Derived::None => {}
            }
        }
        Ok(constraints)
    }

    /// A request's constraints with stage `k`'s derived ones folded in: the
    /// one form both admission and a preview of `k` filter on.
    pub fn placement_constraints(
        &self,
        k: usize,
        base: &RoutingConstraints,
    ) -> Result<RoutingConstraints, PlanError> {
        let derived = self.routing_constraints(k)?;
        let mut constraints = base.clone();
        constraints.required_taints.extend(derived.required_taints);
        for (taint, weight) in derived.preferred_taints {
            *constraints.preferred_taints.entry(taint).or_insert(0.0) += weight;
        }
        Ok(constraints)
    }

    /// The stages `k`'s placement rules read whose worker is not yet known:
    /// neither booked nor skipped. While any remains, `k`'s eligibility
    /// cannot be established.
    pub fn unresolved_reads(&self, k: usize) -> impl Iterator<Item = usize> + '_ {
        self.stage(k)
            .into_iter()
            .flat_map(|stage| stage.constraints.iter().filter_map(Constraint::reads))
            .filter(move |&j| {
                self.worker_config(j).is_none()
                    && self.state_of(j) != Some(&super::plan::StageState::Skipped)
            })
    }
}

impl Plan {
    /// The check in the other direction: the worker chosen for stage `k`
    /// may itself require KV-transfer peers in its own domain, which the
    /// earlier stages' workers must satisfy. Run before booking `k`.
    pub fn check_placement(&self, k: usize, config: &WorkerConfig) -> Result<(), PlanError> {
        let Some(stage) = self.stage(k) else {
            return Err(PlanError::NoSuchStage { stage: k });
        };
        for constraint in &stage.constraints {
            let Constraint::TransferCompatible(j) = constraint else {
                continue;
            };
            let Some(peer) = self.worker_config(*j) else {
                continue;
            };
            if let Derived::Required(taint) = transfer_constraint(config.as_ref(), *j, k)?
                && !peer.taints().contains(&taint)
            {
                return Err(PlanError::Placement {
                    stage: k,
                    reads: *j,
                    reason: format!(
                        "its worker requires peers tainted {taint:?}; stage {j}'s worker is not"
                    ),
                });
            }
        }
        Ok(())
    }
}

fn same_domain_constraint(
    config: &dyn WorkerConfigLike,
    domain: &str,
    mode: DomainMode,
    stage: usize,
    reads: usize,
) -> Result<Derived, PlanError> {
    let Some(value) = topology_value(config, domain) else {
        return match mode {
            DomainMode::Required => Err(PlanError::Placement {
                stage,
                reads,
                reason: format!("its worker publishes no topology domain {domain:?}"),
            }),
            DomainMode::Preferred { .. } => Ok(Derived::None),
        };
    };
    let taint = topology_taint(domain, value);
    Ok(match mode {
        DomainMode::Required => Derived::Required(taint),
        DomainMode::Preferred { weight } => Derived::Preferred(taint, weight),
    })
}

fn topology_value<'a>(config: &'a dyn WorkerConfigLike, domain: &str) -> Option<&'a str> {
    config.topology_domains()?.get(domain).map(String::as_str)
}

/// The KV-transfer constraint a booked worker imposes on its peers.
fn transfer_constraint(
    config: &dyn WorkerConfigLike,
    stage: usize,
    reads: usize,
) -> Result<Derived, PlanError> {
    let Some(domain) = config.kv_transfer_domain() else {
        return Ok(Derived::None);
    };
    let placement = |reason: String| PlanError::Placement {
        stage,
        reads,
        reason,
    };
    let value = topology_value(config, domain).ok_or_else(|| {
        placement(format!(
            "kv_transfer_domain={domain:?} but topology_domains has no such domain"
        ))
    })?;
    let taint = topology_taint(domain, value);
    match config.kv_transfer_enforcement() {
        Some(KvTransferEnforcement::Required) => Ok(Derived::Required(taint)),
        Some(KvTransferEnforcement::Preferred) => {
            let weight = config.kv_transfer_preferred_weight().ok_or_else(|| {
                placement(format!(
                    "preferred KV transfer for {domain:?} but kv_transfer_preferred_weight is missing"
                ))
            })?;
            Ok(Derived::Preferred(taint, weight))
        }
        None => Err(placement(format!(
            "kv_transfer_domain={domain:?} but kv_transfer_enforcement is missing"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use super::super::booking::Booking;
    use super::super::plan::{
        Constraint, DomainMode, Plan, PlanError, PlanId, Stage, WorkerConfig,
    };
    use super::*;
    use crate::WorkerType;
    use crate::identity::RoutingPartitionId;
    use crate::protocols::WorkerWithDpRank;
    use crate::test_utils::SimpleWorkerConfig;

    fn facts(zone: &str, enforcement: Option<KvTransferEnforcement>) -> WorkerConfig {
        Arc::new(SimpleWorkerConfig {
            taints: HashSet::from([topology_taint("zone", zone)]),
            topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: enforcement,
            kv_transfer_preferred_weight: Some(0.5),
            ..SimpleWorkerConfig::default()
        })
    }

    fn booked_prefill(constraint: Constraint, prefill: WorkerConfig) -> Plan {
        let mut plan = Plan::new(
            PlanId::from("p"),
            RoutingPartitionId::new("model", "default"),
            vec![
                Stage::new(WorkerType::Prefill),
                Stage {
                    constraints: vec![constraint],
                    ..Stage::new(WorkerType::Decode)
                },
            ],
        )
        .unwrap();
        let releases = Arc::new(AtomicUsize::new(0));
        plan.book(
            0,
            Booking::scripted("p", WorkerWithDpRank::new(11, 0), releases),
            prefill,
            None,
        )
        .unwrap();
        plan
    }

    #[test]
    fn transfer_compatible_derives_the_prefill_workers_transfer_taint() {
        let plan = booked_prefill(
            Constraint::TransferCompatible(0),
            facts("b", Some(KvTransferEnforcement::Required)),
        );
        let derived = plan.routing_constraints(1).unwrap();
        assert_eq!(
            derived.required_taints,
            HashSet::from([topology_taint("zone", "b")])
        );
        assert!(derived.preferred_taints.is_empty());

        let plan = booked_prefill(
            Constraint::TransferCompatible(0),
            facts("b", Some(KvTransferEnforcement::Preferred)),
        );
        let derived = plan.routing_constraints(1).unwrap();
        assert_eq!(
            derived.preferred_taints,
            HashMap::from([(topology_taint("zone", "b"), 0.5)])
        );

        let plan = booked_prefill(Constraint::TransferCompatible(0), facts("b", None));
        assert!(matches!(
            plan.routing_constraints(1),
            Err(PlanError::Placement {
                stage: 1,
                reads: 0,
                ..
            })
        ));
    }

    #[test]
    fn a_later_worker_with_a_required_transfer_domain_must_match_the_earlier_one() {
        // Decode first in zone b with no transfer policy; the prefill chosen
        // against it requires zone-a peers.
        let plan = booked_prefill(
            Constraint::TransferCompatible(0),
            Arc::new(SimpleWorkerConfig {
                taints: HashSet::from([topology_taint("zone", "b")]),
                topology_domains: HashMap::from([("zone".to_string(), "b".to_string())]),
                ..SimpleWorkerConfig::default()
            }),
        );
        let strict = facts("a", Some(KvTransferEnforcement::Required));
        assert!(matches!(
            plan.check_placement(1, &strict),
            Err(PlanError::Placement {
                stage: 1,
                reads: 0,
                ..
            })
        ));
        let preferred = facts("a", Some(KvTransferEnforcement::Preferred));
        plan.check_placement(1, &preferred).unwrap();
        let same_zone = facts("b", Some(KvTransferEnforcement::Required));
        plan.check_placement(1, &same_zone).unwrap();
    }

    #[test]
    fn same_domain_requires_the_domain_only_when_required() {
        let rule = |mode| Constraint::SameDomain {
            stage: 0,
            key: "rack".to_string(),
            mode,
        };
        let plan = booked_prefill(
            rule(DomainMode::Required),
            Arc::new(SimpleWorkerConfig::default()) as WorkerConfig,
        );
        assert!(matches!(
            plan.routing_constraints(1),
            Err(PlanError::Placement { .. })
        ));
        let plan = booked_prefill(
            rule(DomainMode::Preferred { weight: 1.0 }),
            Arc::new(SimpleWorkerConfig::default()) as WorkerConfig,
        );
        assert!(plan.routing_constraints(1).unwrap().is_empty());
    }
}
