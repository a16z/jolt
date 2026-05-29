//! Verifier-selected protocol configuration.

use jolt_claims::protocols::field_inline::FieldInlineConfig;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
}

impl JoltProtocolConfig {
    pub const fn for_zk(zk: bool) -> Self {
        Self {
            zk: ZkConfig::from_bool(zk),
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
        }
    }
}

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::native_v1();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

#[cfg(feature = "zk")]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::BlindFold;

#[cfg(not(feature = "zk"))]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::Transparent;

pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    field_inline: SELECTED_FIELD_INLINE_CONFIG,
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
