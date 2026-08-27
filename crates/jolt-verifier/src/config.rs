//! Verifier-selected protocol configuration.
//!
//! The protocol choices are fixed at compile time: `zk` selects BlindFold,
//! while `akita` selects packed commitments and little-endian scalar
//! challenges. One compiled verifier therefore runs exactly one protocol. A
//! proof self-describes these choices and [`validate_proof_config`] rejects a
//! mismatch fail-closed.

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

/// Which transcript-prior cycle point anchors the booleanity relation's
/// `eq(r_ref_cycle, ·)` factor. The anchor is a pure eq anchor (the
/// address-phase input claim is identically zero; no upstream claim lives at
/// the point; it is derived, never drawn), so any point sampled after the
/// one-hot commitments are transcript-bound is sound — the axis exists so a
/// proof self-describes its choice and mismatched provers/verifiers reject
/// cleanly instead of failing mid-sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BooleanityAnchor {
    /// Legacy: the reversed stage-5 instruction read-RAF cycle point.
    Stage5Instruction,
    /// The reversed stage-1 cycle binding — known four stages earlier, so the
    /// prover overlaps the address-phase pushforward build with stage 5.
    /// Transparent-mode only: BlindFold pins [`Self::Stage5Instruction`]
    /// (fail-closed in [`validate_proof_config`]).
    Stage1CycleV1,
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
    pub booleanity_anchor: BooleanityAnchor,
    pub scalar_challenge_endianness: ScalarChallengeEndianness,
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
            booleanity_anchor: BooleanityAnchor::Stage5Instruction,
            scalar_challenge_endianness: SELECTED_SCALAR_CHALLENGE_ENDIANNESS,
        }
    }
}

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

/// The one protocol this build verifies. The booleanity-anchor member is the
/// baseline the equality axes compare against; [`validate_proof_config`]
/// additionally admits [`BooleanityAnchor::Stage1CycleV1`] on transparent
/// builds (the anchor is prover-selected, not build-pinned).
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    commitment: SELECTED_COMMITMENT_CONFIG,
    booleanity_anchor: BooleanityAnchor::Stage5Instruction,
    scalar_challenge_endianness: SELECTED_SCALAR_CHALLENGE_ENDIANNESS,
};

pub fn validate_proof_config(
    config: &JoltProtocolConfig,
    protocol: JoltProtocolConfig,
) -> Result<(), VerifierError> {
    // The zk and commitment axes are compile-time-pinned: exact equality. The
    // booleanity anchor is prover-selected within the transparent protocol —
    // either anchor verifies on a transparent build, while any BlindFold
    // combination pins the legacy anchor (fail-closed, before any stage work).
    let anchor_allowed = match protocol.booleanity_anchor {
        BooleanityAnchor::Stage5Instruction => true,
        BooleanityAnchor::Stage1CycleV1 => {
            protocol.zk == ZkConfig::Transparent && config.zk == ZkConfig::Transparent
        }
    };
    if protocol.zk != config.zk
        || protocol.commitment != config.commitment
        || protocol.scalar_challenge_endianness != config.scalar_challenge_endianness
        || !anchor_allowed
    {
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
    fn anchor_axis_is_transparent_only() {
        let transparent = JoltProtocolConfig::for_zk(false);
        let transparent_v1 = JoltProtocolConfig {
            booleanity_anchor: BooleanityAnchor::Stage1CycleV1,
            ..transparent
        };
        assert!(validate_proof_config(&transparent, transparent).is_ok());
        assert!(validate_proof_config(&transparent, transparent_v1).is_ok());

        let blindfold = JoltProtocolConfig::for_zk(true);
        let blindfold_v1 = JoltProtocolConfig {
            booleanity_anchor: BooleanityAnchor::Stage1CycleV1,
            ..blindfold
        };
        // A BlindFold proof claiming the V1 anchor is rejected fail-closed,
        // even by a build whose zk axis matches.
        assert!(validate_proof_config(&blindfold, blindfold_v1).is_err());
        assert!(validate_proof_config(&blindfold, blindfold).is_ok());
        // Cross-axis mismatches keep failing regardless of anchor.
        assert!(validate_proof_config(&blindfold, transparent_v1).is_err());
        assert!(validate_proof_config(&transparent, blindfold).is_err());
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
