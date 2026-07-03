//! Jolt verifier configuration: the protocol axes ([`JoltProtocolConfig`]:
//! zk × commitment), fixed at compile time. A proof self-describes its axes
//! and [`validate_proof_config`] rejects a mismatch.

use jolt_openings::CommitmentScheme;
use serde::{Deserialize, Serialize};

use crate::{proof::JoltProof, VerifierError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

/// The commitment axis of the protocol: how committed polynomials are
/// discharged at the final opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentConfig {
    /// Per-polynomial commitments, RLC batch opening (requires additive
    /// homomorphism).
    Homomorphic,
    /// One packed witness per commitment lifecycle, reduction-sumcheck batch
    /// opening (no homomorphism; the Akita path).
    Packed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub commitment: CommitmentConfig,
}

/// The protocol config this build verifies, fixed at compile time by the
/// `zk` feature; proofs self-describe theirs and [`validate_proof_config`]
/// rejects a mismatch.
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: if cfg!(feature = "zk") {
        ZkConfig::BlindFold
    } else {
        ZkConfig::Transparent
    },
    commitment: CommitmentConfig::Homomorphic,
};

pub fn validate_proof_config<PCS, VC, ZkProof, JointOpeningProof>(
    config: &JoltProtocolConfig,
    proof: &JoltProof<PCS, VC, ZkProof, JointOpeningProof>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
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
