//! Verifier-selected protocol configuration.

use serde::{Deserialize, Serialize};

use crate::{proof::JoltProof, VerifierError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

impl ZkConfig {
    pub const fn from_bool(zk: bool) -> Self {
        if zk {
            Self::BlindFold
        } else {
            Self::Transparent
        }
    }
}

pub fn validate_proof_config<PCS>(
    config: ZkConfig,
    proof: &JoltProof<PCS>,
) -> Result<(), VerifierError>
where
    PCS: jolt_openings::CommitmentScheme,
{
    if proof.protocol != config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: config,
            got: proof.protocol,
        });
    }

    Ok(())
}
