//! Verifier-selected protocol configuration.
//!
//! Every protocol axis is fixed at compile time — the `zk` feature selects
//! zero knowledge (BlindFold over Dory, Whiteout over Akita), the `akita`
//! feature selects Akita commitments, the `field-inline` feature enables the
//! native field-register extension — so one compiled verifier runs exactly one
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

#[cfg(all(feature = "zk", feature = "akita", not(feature = "whiteout")))]
compile_error!(
    "`zk` with `akita` selects Whiteout, which needs the explicit `whiteout` feature: \
     Whiteout's constant-time claim is not yet verified on all platforms (an open \
     timing residual in the Σ-protocol's `N·y` product; specs/whiteout-sigma.md)"
);

/// The zero-knowledge axis. Serialized in [`JoltProof`](crate::proof::JoltProof)
/// and absorbed as the first preamble byte, so variants are append-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    /// BlindFold over pads committed with a curve vector commitment (Dory).
    BlindFold,
    /// Whiteout over a lattice pad commitment, with Akita's masked levels
    /// and Σ-protocol (`specs/whiteout-jolt-integration.md`).
    Whiteout,
}

impl ZkConfig {
    /// Whether private messages are masked by committed pads: every mode but
    /// [`Transparent`](Self::Transparent).
    pub const fn is_masked(self) -> bool {
        match self {
            Self::Transparent => false,
            Self::BlindFold | Self::Whiteout => true,
        }
    }
}

/// The commitment axis of the protocol: how committed polynomials are
/// discharged at the final opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentConfig {
    /// Per-polynomial commitments, RLC batch opening (requires additive
    /// homomorphism).
    Homomorphic,
    /// Native one-hot trace groups and dense auxiliary commitments with
    /// heterogeneous Akita opening and verification.
    Akita,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub commitment: CommitmentConfig,
    pub field_inline: FieldInlineConfig,
}

impl JoltProtocolConfig {
    pub const fn for_zk(zk: ZkConfig) -> Self {
        Self {
            zk,
            commitment: SELECTED_COMMITMENT_CONFIG,
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

/// `zk` picks zero knowledge; the commitment axis picks its back end.
pub const SELECTED_ZK_CONFIG: ZkConfig = match (cfg!(feature = "zk"), cfg!(feature = "akita")) {
    (false, _) => ZkConfig::Transparent,
    (true, false) => ZkConfig::BlindFold,
    (true, true) => ZkConfig::Whiteout,
};

#[cfg(feature = "akita")]
pub const SELECTED_COMMITMENT_CONFIG: CommitmentConfig = CommitmentConfig::Akita;

#[cfg(not(feature = "akita"))]
pub const SELECTED_COMMITMENT_CONFIG: CommitmentConfig = CommitmentConfig::Homomorphic;

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::enabled();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

/// The minimum padded trace length — the compiled protocol's PCS floor
/// (legacy's `PCS::MIN_PADDED_TRACE_LENGTH`). Dory needs `T >= K^(1/D)`
/// (256); Akita's folded-only protocol cannot schedule the K=16
/// `OneHotTrace` group below 16 variables, and column arity is
/// `log_k_chunk + log_T`, so the Akita pipeline pads every trace to at
/// least 2^12 cycles. The prover pads to this floor and the verifier rejects
/// shorter traces.
#[cfg(not(feature = "akita"))]
pub const MIN_PADDED_TRACE_LENGTH: usize = 256;
#[cfg(feature = "akita")]
pub const MIN_PADDED_TRACE_LENGTH: usize = 1 << 12;

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
        let akita_protocol = JoltProtocolConfig {
            commitment: CommitmentConfig::Akita,
            ..protocol
        };
        assert_eq!(
            postcard::to_stdvec(&akita_protocol).unwrap(),
            [0, 1, 0, 4, 0]
        );
        assert_eq!(
            postcard::from_bytes::<JoltProtocolConfig>(&[0, 1, 0, 4, 0]).unwrap(),
            akita_protocol
        );
        let whiteout_protocol = JoltProtocolConfig {
            zk: ZkConfig::Whiteout,
            ..akita_protocol
        };
        assert_eq!(
            postcard::to_stdvec(&whiteout_protocol).unwrap(),
            [2, 1, 0, 4, 0]
        );
        assert_eq!(
            postcard::from_bytes::<JoltProtocolConfig>(&[2, 1, 0, 4, 0]).unwrap(),
            whiteout_protocol
        );
        assert_eq!(
            validate_proof_config(&JOLT_VERIFIER_CONFIG, whiteout_protocol).is_ok(),
            JOLT_VERIFIER_CONFIG == whiteout_protocol,
            "only a Whiteout build accepts a Whiteout proof"
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
