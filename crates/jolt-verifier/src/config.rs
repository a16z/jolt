//! Verifier-selected protocol configuration.
//!
//! Every protocol axis is fixed at compile time — the `zk` feature selects
//! BlindFold, the `akita` feature selects packed commitments and little-endian
//! scalar challenges, the `field-inline` feature enables the native
//! field-register extension — so one compiled verifier runs exactly one
//! protocol. A proof self-describes its axes and [`validate_proof_config`]
//! rejects a mismatch fail-closed.

pub use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_riscv::JoltInstructionProfile;
#[cfg(not(feature = "field-inline"))]
use jolt_riscv::RV64IMAC_JOLT;
#[cfg(feature = "field-inline")]
use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;
use serde::{Deserialize, Serialize};

use crate::VerifierError;

#[cfg(all(feature = "zk", feature = "akita"))]
compile_error!(
    "the `zk` and `akita` features are mutually exclusive: no zk protocol exists over the \
     packed commitment axis (a lattice-friendly hiding commitment is a future workstream)"
);

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
    /// Packed one-hot trace and dense advice commitments with heterogeneous
    /// Akita opening and verification.
    Packed,
}

/// Byte order used to decode scalar Fiat-Shamir challenges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarChallengeEndianness {
    Big,
    Little,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub commitment: CommitmentConfig,
    pub scalar_challenge_endianness: ScalarChallengeEndianness,
    pub field_inline: FieldInlineConfig,
}

impl JoltProtocolConfig {
    pub const fn for_zk(zk: bool) -> Self {
        Self {
            zk: if zk {
                ZkConfig::BlindFold
            } else {
                ZkConfig::Transparent
            },
            commitment: SELECTED_COMMITMENT_CONFIG,
            scalar_challenge_endianness: SELECTED_SCALAR_CHALLENGE_ENDIANNESS,
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
        }
    }
}

/// The instruction profile this build has constraints for. Input validation
/// rejects a program whose bytecode carries any other Jolt instruction kind:
/// the base rows pin an rd write only through the lookup/load/jump flags, so a
/// row from an extension the verifier was not built with would verify with
/// its write unconstrained.
#[cfg(feature = "field-inline")]
pub const JOLT_VERIFIER_INSTRUCTION_PROFILE: JoltInstructionProfile = RV64IMAC_JOLT_FIELD_INLINE;
#[cfg(not(feature = "field-inline"))]
pub const JOLT_VERIFIER_INSTRUCTION_PROFILE: JoltInstructionProfile = RV64IMAC_JOLT;

#[cfg(feature = "zk")]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::BlindFold;

#[cfg(not(feature = "zk"))]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::Transparent;

#[cfg(feature = "akita")]
pub const SELECTED_COMMITMENT_CONFIG: CommitmentConfig = CommitmentConfig::Packed;

#[cfg(not(feature = "akita"))]
pub const SELECTED_COMMITMENT_CONFIG: CommitmentConfig = CommitmentConfig::Homomorphic;

#[cfg(feature = "akita")]
pub const SELECTED_SCALAR_CHALLENGE_ENDIANNESS: ScalarChallengeEndianness =
    ScalarChallengeEndianness::Little;

#[cfg(not(feature = "akita"))]
pub const SELECTED_SCALAR_CHALLENGE_ENDIANNESS: ScalarChallengeEndianness =
    ScalarChallengeEndianness::Big;

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::enabled();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

/// The one protocol this build verifies.
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    commitment: SELECTED_COMMITMENT_CONFIG,
    scalar_challenge_endianness: SELECTED_SCALAR_CHALLENGE_ENDIANNESS,
    field_inline: SELECTED_FIELD_INLINE_CONFIG,
};

pub fn validate_proof_config(
    config: &JoltProtocolConfig,
    protocol: JoltProtocolConfig,
) -> Result<(), VerifierError> {
    if protocol != *config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: *config,
            got: protocol,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_protocol_config_is_accepted() {
        assert!(validate_proof_config(&JOLT_VERIFIER_CONFIG, JOLT_VERIFIER_CONFIG).is_ok());
    }

    /// A proof declaring the opposite field-inline axis rejects fail-closed, in
    /// both FR-off builds (proof claims enabled) and FR-on builds (proof claims
    /// disabled).
    #[test]
    fn mismatched_field_inline_axis_is_rejected() {
        let mut protocol = JOLT_VERIFIER_CONFIG;
        protocol.field_inline = if JOLT_VERIFIER_CONFIG.field_inline.enabled {
            FieldInlineConfig::disabled()
        } else {
            FieldInlineConfig::enabled()
        };

        assert!(matches!(
            validate_proof_config(&JOLT_VERIFIER_CONFIG, protocol),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }

    /// A proof declaring a different FR register-file size rejects even when the
    /// enabled bit matches: the whole config participates in the equality gate.
    #[test]
    fn mismatched_field_register_log_k_is_rejected() {
        let mut protocol = JOLT_VERIFIER_CONFIG;
        protocol.field_inline.field_register_log_k += 1;

        assert!(matches!(
            validate_proof_config(&JOLT_VERIFIER_CONFIG, protocol),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }

    #[test]
    fn rejects_scalar_challenge_endianness_mismatch() {
        let mut proof_config = JOLT_VERIFIER_CONFIG;
        proof_config.scalar_challenge_endianness = match proof_config.scalar_challenge_endianness {
            ScalarChallengeEndianness::Big => ScalarChallengeEndianness::Little,
            ScalarChallengeEndianness::Little => ScalarChallengeEndianness::Big,
        };

        assert!(validate_proof_config(&JOLT_VERIFIER_CONFIG, proof_config).is_err());
    }
}
