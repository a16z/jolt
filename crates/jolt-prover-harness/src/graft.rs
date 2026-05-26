use serde::{Deserialize, Serialize};

use crate::{AcceptanceMode, FrontierSpec, HarnessError, HarnessResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraftSurface {
    ProtocolConfig,
    Commitments,
    TrustedAdviceCommitment,
    UntrustedAdviceCommitment,
    Stage(u8),
    OpeningClaims,
    OpeningProof,
    BlindFoldProof,
    FieldInlineCommitments,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraftRecord {
    pub surface: GraftSurface,
    pub label: String,
}

impl GraftRecord {
    pub fn new(surface: GraftSurface, label: impl Into<String>) -> Self {
        Self {
            surface,
            label: label.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraftPlan {
    pub records: Vec<GraftRecord>,
}

impl GraftPlan {
    pub fn new(records: Vec<GraftRecord>) -> Self {
        Self { records }
    }

    pub fn validate_for(self, frontier: &FrontierSpec) -> HarnessResult<Self> {
        if frontier.mode != AcceptanceMode::FullProofGraft {
            return Err(HarnessError::InvalidAcceptanceMode {
                frontier: frontier.name,
                mode: frontier.mode,
            });
        }
        if self.records.is_empty() {
            return Err(HarnessError::InvalidManifest {
                frontier: frontier.name,
                reason: "full-proof graft frontier must name at least one graft surface".to_owned(),
            });
        }
        Ok(self)
    }
}
