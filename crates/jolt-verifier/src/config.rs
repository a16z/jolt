//! Verifier-selected protocol configuration.

use serde::{Deserialize, Serialize};

use crate::{proof::JoltProof, VerifierError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
}

impl JoltProtocolConfig {
    pub const fn for_zk(zk: bool) -> Self {
        Self {
            zk: if zk {
                ZkConfig::BlindFold
            } else {
                ZkConfig::Transparent
            },
        }
    }
}

#[cfg(feature = "zk")]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::BlindFold;

#[cfg(not(feature = "zk"))]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::Transparent;

pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
};

pub fn validate_proof_config<PCS, VC, ZkProof>(
    config: &JoltProtocolConfig,
    proof: &JoltProof<PCS, VC, ZkProof>,
) -> Result<(), VerifierError>
where
    PCS: jolt_openings::CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
{
    if proof.protocol != *config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: *config,
            got: proof.protocol,
        });
    }

    Ok(())
}
