//! Verifier-selected protocol configuration.
//!
//! Every protocol axis is fixed at compile time — the `zk` feature selects
//! BlindFold, the `akita` feature selects packed commitments, the
//! `field-inline` feature enables the native field-register extension — so one
//! compiled verifier runs exactly one protocol. The verifier absorbs its own
//! configuration into the transcript ([`JoltProtocolConfig::transcript_bytes`]);
//! a proof also self-describes its axes so [`validate_proof_config`] reports a
//! mismatch directly.

pub use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_claims::protocols::field_inline::FieldInlineRepresentation;
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

/// The protocol axes a proof is produced under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub commitment: CommitmentConfig,
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
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
        }
    }

    /// The canonical encoding the transcript absorbs: one byte per axis, then
    /// the field-inline parameters (`field_register_log_k` as `u64` little
    /// endian, then the representation byte).
    pub fn transcript_bytes(&self) -> [u8; 12] {
        let zk = match self.zk {
            ZkConfig::Transparent => 0,
            ZkConfig::BlindFold => 1,
        };
        let commitment = match self.commitment {
            CommitmentConfig::Homomorphic => 0,
            CommitmentConfig::Packed => 1,
        };
        let representation = match self.field_inline.representation {
            FieldInlineRepresentation::NativeFieldElement => 0,
        };
        let [k0, k1, k2, k3, k4, k5, k6, k7] =
            crate::num::u64_from_usize(self.field_inline.field_register_log_k).to_le_bytes();
        [
            zk,
            commitment,
            u8::from(self.field_inline.enabled),
            k0,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            representation,
        ]
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

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::enabled();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

/// The one protocol this build verifies.
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    commitment: SELECTED_COMMITMENT_CONFIG,
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
#[expect(clippy::unwrap_used, reason = "wire-format test assertions")]
mod tests {
    use super::*;

    #[test]
    fn protocol_wire_format_is_explicit() {
        let protocol = JoltProtocolConfig {
            zk: ZkConfig::Transparent,
            commitment: CommitmentConfig::Homomorphic,
            field_inline: FieldInlineConfig::disabled(),
        };
        assert_eq!(postcard::to_stdvec(&protocol).unwrap(), [0, 0, 0, 4, 0]);
        assert_eq!(
            postcard::from_bytes::<JoltProtocolConfig>(&[0, 0, 0, 4, 0]).unwrap(),
            protocol
        );
        assert!(postcard::from_bytes::<JoltProtocolConfig>(&[0, 0, 0]).is_err());
        assert_eq!(
            protocol.transcript_bytes(),
            [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    /// A proof declaring a different field-register file size rejects even when the enabled
    /// bit matches: the whole config participates in the equality gate.
    #[test]
    fn mismatched_field_register_log_k_is_rejected() {
        let mut protocol = JOLT_VERIFIER_CONFIG;
        protocol.field_inline.field_register_log_k += 1;

        assert!(matches!(
            validate_proof_config(&JOLT_VERIFIER_CONFIG, protocol),
            Err(VerifierError::ProtocolConfigMismatch { .. })
        ));
    }
}
