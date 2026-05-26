use serde::{Deserialize, Serialize};

use crate::{perf::PerfGate, HarnessError, HarnessResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceptanceMode {
    FullProofGraft,
    PrefixCheckpoint,
    Sovereign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureMode {
    Transparent,
    Zk,
    FieldInline,
    ZkFieldInline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParityTarget {
    VerifierAcceptance,
    CoreCommitments,
    CoreStageOutput,
    CoreOpeningClaims,
    CoreProofShape,
    BackendReference,
    Performance,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FrontierSpec {
    pub name: &'static str,
    pub mode: AcceptanceMode,
    pub fixtures: &'static [crate::FixtureKind],
    pub features: &'static [FeatureMode],
    pub parity: &'static [ParityTarget],
    pub perf: Option<PerfGate>,
    pub optimization_ids: &'static [&'static str],
}

impl FrontierSpec {
    pub fn validate(self) -> HarnessResult<()> {
        if self.name.trim().is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier name must not be empty".to_owned(),
            });
        }
        if self.fixtures.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name at least one fixture".to_owned(),
            });
        }
        if has_duplicate(self.fixtures) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier fixture list contains duplicates".to_owned(),
            });
        }
        if self.features.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name at least one feature mode".to_owned(),
            });
        }
        if has_duplicate(self.features) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier feature-mode list contains duplicates".to_owned(),
            });
        }
        if !self.parity.contains(&ParityTarget::VerifierAcceptance) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must include verifier acceptance as a parity target".to_owned(),
            });
        }
        if has_duplicate(self.parity) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier parity-target list contains duplicates".to_owned(),
            });
        }
        if self.optimization_ids.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier must name optimization-inventory IDs or an explicit non-perf ID"
                    .to_owned(),
            });
        }
        if has_duplicate(self.optimization_ids) {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "frontier optimization ID list contains duplicates".to_owned(),
            });
        }
        if self.parity.contains(&ParityTarget::Performance) && self.perf.is_none() {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "performance parity target requires a perf gate".to_owned(),
            });
        }
        if let Some(perf) = self.perf {
            if let Err(reason) = perf.validate() {
                return Err(HarnessError::InvalidManifest {
                    frontier: self.name,
                    reason: reason.to_owned(),
                });
            }
        }
        if matches!(self.mode, AcceptanceMode::Sovereign)
            && self.parity.contains(&ParityTarget::CoreProofShape)
        {
            return Err(HarnessError::InvalidManifest {
                frontier: self.name,
                reason: "sovereign frontiers should not rely on core proof shape".to_owned(),
            });
        }
        Ok(())
    }
}

fn has_duplicate<T: PartialEq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
}

#[derive(Clone, Debug, Default)]
pub struct FrontierManifest {
    frontiers: Vec<FrontierSpec>,
}

impl FrontierManifest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, frontier: FrontierSpec) -> HarnessResult<()> {
        frontier.validate()?;
        if self
            .frontiers
            .iter()
            .any(|existing| existing.name == frontier.name)
        {
            return Err(HarnessError::InvalidManifest {
                frontier: frontier.name,
                reason: "duplicate frontier name".to_owned(),
            });
        }
        self.frontiers.push(frontier);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = &FrontierSpec> {
        self.frontiers.iter()
    }

    pub fn find(&self, name: &str) -> Option<&FrontierSpec> {
        self.frontiers.iter().find(|frontier| frontier.name == name)
    }

    pub fn validate_all(&self) -> HarnessResult<()> {
        for frontier in &self.frontiers {
            frontier.validate()?;
        }
        Ok(())
    }
}
