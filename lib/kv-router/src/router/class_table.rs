// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-class stage lists: the routing policy's output, as data. A class in
//! the policy YAML names its list (`stages: prefill_decode`) or spells it
//! out; a class without one is aggregated. Built-in lists are plain `Stage`
//! values, so a custom list is the same thing written down.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;

use crate::WorkerType;
use crate::scheduling::policy_config::{PolicyClassConfig, PolicyProfile};

use super::plan::{Budget, Constraint, SkipRule, Stage, When};

/// What a class does when its first stage is rejected for capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fallback {
    /// Book the decode set as one stage that does its own prefill.
    Aggregated,
}

/// A class's stage list. YAML: a built-in name, a list of stages, or
/// `{ stages: [...], fallback: aggregated }`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "StageListSpec")]
pub struct StageList {
    pub stages: Vec<Stage>,
    pub fallback: Option<Fallback>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StageListSpec {
    Builtin(String),
    Stages(Vec<Stage>),
    Full {
        stages: Vec<Stage>,
        #[serde(default)]
        fallback: Option<Fallback>,
    },
}

impl TryFrom<StageListSpec> for StageList {
    type Error = String;

    fn try_from(spec: StageListSpec) -> Result<Self, String> {
        Ok(match spec {
            StageListSpec::Builtin(name) => StageList::builtin(&name).ok_or_else(|| {
                format!(
                    "unknown built-in stage list {name:?}; expected one of {}",
                    StageList::BUILTINS.join(", ")
                )
            })?,
            StageListSpec::Stages(stages) => StageList {
                stages,
                fallback: None,
            },
            StageListSpec::Full { stages, fallback } => StageList { stages, fallback },
        })
    }
}

/// How long a deferred decode may wait while its prefill's KV is held.
pub const DEFAULT_DEFERRED_WAIT: Duration = Duration::from_secs(2);

impl StageList {
    pub fn new(stages: Vec<Stage>) -> Self {
        Self {
            stages,
            fallback: None,
        }
    }

    pub fn with_fallback(mut self, fallback: Fallback) -> Self {
        self.fallback = Some(fallback);
        self
    }

    pub const BUILTINS: [&'static str; 6] = [
        "aggregated",
        "prefill_decode",
        "prefill_decode_deferred",
        "decode_first",
        "conditional_prefill_decode",
        "encode_prefill_decode",
    ];

    pub fn builtin(name: &str) -> Option<Self> {
        Some(match name {
            "aggregated" => Self::aggregated(),
            "prefill_decode" => Self::prefill_decode(),
            "prefill_decode_deferred" => Self::prefill_decode_deferred(DEFAULT_DEFERRED_WAIT),
            "decode_first" => Self::decode_first(),
            "conditional_prefill_decode" => Self::conditional_prefill_decode(),
            "encode_prefill_decode" => Self::encode_prefill_decode(),
            _ => return None,
        })
    }

    /// One stage on the aggregated set.
    pub fn aggregated() -> Self {
        Self::new(vec![Stage::new(WorkerType::Aggregated)])
    }

    /// Prefill and decode booked together; decode is forwarded after prefill.
    pub fn prefill_decode() -> Self {
        Self::new(vec![
            Stage::new(WorkerType::Prefill),
            Stage {
                inputs: vec![0],
                wait: Budget::Immediate,
                constraints: vec![Constraint::TransferCompatible(0)],
                ..Stage::new(WorkerType::Decode)
            },
        ])
    }

    /// Prefill now; decode booked once prefill completes, against load as it
    /// is then, waiting at most `wait` while the prefill KV is held.
    pub fn prefill_decode_deferred(wait: Duration) -> Self {
        Self::new(vec![
            Stage::new(WorkerType::Prefill),
            Stage {
                when: When::After(0),
                wait: Budget::Bounded(wait),
                constraints: vec![Constraint::TransferCompatible(0)],
                ..Stage::new(WorkerType::Decode)
            },
        ])
    }

    /// Decode booked first, prefill placed against it, prefill forwarded first.
    pub fn decode_first() -> Self {
        Self::new(vec![
            Stage {
                inputs: vec![1],
                ..Stage::new(WorkerType::Decode)
            },
            Stage {
                wait: Budget::Immediate,
                constraints: vec![Constraint::TransferCompatible(0)],
                ..Stage::new(WorkerType::Prefill)
            },
        ])
    }

    /// `prefill_decode` whose prefill stage is skipped when the conditional
    /// disaggregation policy says the previewed decode worker should do it.
    pub fn conditional_prefill_decode() -> Self {
        let mut list = Self::prefill_decode();
        list.stages[0].skip = Some(SkipRule::ConditionalDisagg);
        list
    }

    /// Encode, then prefill, then decode, each booked once its input
    /// completes; encode is skipped without multimodal input.
    pub fn encode_prefill_decode() -> Self {
        Self::new(vec![
            Stage {
                skip: Some(SkipRule::NoMultimodal),
                ..Stage::new(WorkerType::Encode)
            },
            Stage {
                when: When::After(0),
                ..Stage::new(WorkerType::Prefill)
            },
            Stage {
                when: When::After(1),
                wait: Budget::Bounded(DEFAULT_DEFERRED_WAIT),
                constraints: vec![Constraint::TransferCompatible(1)],
                ..Stage::new(WorkerType::Decode)
            },
        ])
    }

    /// Every stage booked now with no hold: the EPP's `all_now` mode. A
    /// deferred stage keeps its execution input; only its booking moves.
    pub fn all_now(&self) -> Self {
        Self {
            stages: self
                .stages
                .iter()
                .cloned()
                .map(|mut stage| {
                    if let When::After(input) = stage.when
                        && !stage.inputs.contains(&input)
                    {
                        stage.inputs.push(input);
                    }
                    Stage {
                        when: When::Now,
                        wait: Budget::Immediate,
                        ..stage
                    }
                })
                .collect(),
            fallback: self.fallback,
        }
    }

    /// The worker sets this list books into.
    pub fn sets(&self) -> impl Iterator<Item = WorkerType> + '_ {
        self.stages.iter().map(|stage| stage.set)
    }
}

/// Stage lists by class name, with the list a request gets when its class
/// names none.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassTable {
    classes: HashMap<String, StageList>,
    default: StageList,
}

impl Default for ClassTable {
    fn default() -> Self {
        Self::new(StageList::aggregated())
    }
}

/// The name a request carries to reach `class`: its family, or the class
/// name for an explicit class.
fn family_key(class: &PolicyClassConfig) -> &str {
    class.policy_family.as_deref().unwrap_or(&class.name)
}

impl ClassTable {
    pub fn new(default: StageList) -> Self {
        Self {
            classes: HashMap::new(),
            default,
        }
    }

    pub fn with_class(mut self, name: impl Into<String>, list: StageList) -> Self {
        self.classes.insert(name.into(), list);
        self
    }

    /// The stage lists a policy profile's classes declare, keyed by the name
    /// a request carries: the class's routing family, or the class name for
    /// an explicit class. The scheduler resolves family + cached-token bucket
    /// to a physical class for queue settings; routing shape varies by the
    /// family. A family's list is whatever its bucket classes declare: a
    /// bucket that omits `stages` inherits it, and two buckets declaring
    /// different lists make the profile an error.
    ///
    /// A request naming no class (or a name the profile does not know) is
    /// queued under the profile's default family, so it routes by that
    /// family's list through the same mapping, and by `fallback` when no
    /// bucket of the family declares one.
    pub fn from_profile(profile: &PolicyProfile, fallback: StageList) -> Result<Self, String> {
        let mut classes: HashMap<String, StageList> = HashMap::new();
        for class in profile.classes() {
            let Some(list) = &class.stages else {
                continue;
            };
            let key = family_key(class);
            match classes.get(key) {
                Some(existing) if existing != list => {
                    return Err(format!(
                        "policy family {key:?}: class {:?} declares a different stage list than another class of the family",
                        class.name
                    ));
                }
                Some(_) => {}
                None => {
                    classes.insert(key.to_string(), list.clone());
                }
            }
        }
        let default = classes
            .get(family_key(profile.default_class()))
            .cloned()
            .unwrap_or(fallback);
        Ok(Self { classes, default })
    }

    pub fn stages(&self, class: Option<&str>) -> &StageList {
        class
            .and_then(|name| self.classes.get(name))
            .unwrap_or(&self.default)
    }

    pub fn lists(&self) -> impl Iterator<Item = (Option<&str>, &StageList)> {
        std::iter::once((None, &self.default)).chain(
            self.classes
                .iter()
                .map(|(name, list)| (Some(name.as_str()), list)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_lists_parse_from_yaml_shorthand_and_long_form() {
        let list: StageList = serde_yaml::from_str("prefill_decode_deferred").unwrap();
        assert_eq!(
            list,
            StageList::prefill_decode_deferred(DEFAULT_DEFERRED_WAIT)
        );

        let list: StageList = serde_yaml::from_str(
            "[prefill, { set: decode, inputs: [0], wait: immediate, constraints: [{ transfer_compatible: 0 }] }]",
        )
        .unwrap();
        assert_eq!(list, StageList::prefill_decode());

        let list: StageList = serde_yaml::from_str(
            "stages: [prefill, { set: decode, when: { after: 0 }, wait: 500ms }]\nfallback: aggregated",
        )
        .unwrap();
        assert_eq!(list.fallback, Some(Fallback::Aggregated));
        assert_eq!(list.stages[1].when, When::After(0));
        assert_eq!(
            list.stages[1].wait,
            Budget::Bounded(Duration::from_millis(500))
        );

        let list: StageList = serde_yaml::from_str(
            "[{ set: encode, skip: no_multimodal }, { set: prefill, constraints: [{ same_domain: { stage: 0, key: zone, mode: { preferred: { weight: 0.5 } } } }] }]",
        )
        .unwrap();
        assert_eq!(list.stages[0].skip, Some(SkipRule::NoMultimodal));
        assert!(serde_yaml::from_str::<StageList>("[{ set: decode, wait: soon }]").is_err());
        let error = serde_yaml::from_str::<StageList>("prefill_then_decode").unwrap_err();
        assert!(error.to_string().contains("unknown built-in"), "{error}");
    }

    #[test]
    fn all_now_books_everything_immediately() {
        let list = StageList::encode_prefill_decode().all_now();
        assert!(list.stages.iter().all(|stage| stage.when == When::Now));
        assert!(
            list.stages
                .iter()
                .all(|stage| stage.wait == Budget::Immediate)
        );
        assert_eq!(list.stages[1].inputs, vec![0]);
        assert_eq!(list.stages[2].inputs, vec![1], "execution order survives");
    }
}
