//! Verifier-selected protocol configuration.
//!
//! The protocol choices are fixed at compile time: `zk` selects BlindFold,
//! while `akita` selects packed commitments and little-endian scalar
//! challenges. One compiled verifier therefore runs exactly one protocol. A
//! proof self-describes these choices and [`validate_proof_config`] rejects a
//! mismatch fail-closed.

use common::constants::{ONEHOT_CHUNK_THRESHOLD_LOG_T, XLEN};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
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

/// The one protocol this build verifies.
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    commitment: SELECTED_COMMITMENT_CONFIG,
    scalar_challenge_endianness: SELECTED_SCALAR_CHALLENGE_ENDIANNESS,
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

/// Smallest accepted padded trace length: one cycle variable, so every
/// cycle-indexed sumcheck has at least one round. The prover pads far higher
/// (its PCS floor); this is the verifier-side floor.
pub const MIN_TRACE_LENGTH: usize = 2;

/// The full instruction lookup key width: two `XLEN`-bit operands.
const LOOKUP_ADDRESS_BITS: usize = 2 * XLEN;

/// The one-hot chunking regime below the trace-length threshold: 4-bit
/// committed chunks and `LOOKUP_ADDRESS_BITS / 8`-bit virtual-RA chunks.
pub const NARROW_ONE_HOT_CONFIG: JoltOneHotConfig = JoltOneHotConfig {
    log_k_chunk: 4,
    lookups_ra_virtual_log_k_chunk: 16,
};

/// The one-hot chunking regime at or above the threshold: 8-bit committed
/// chunks and `LOOKUP_ADDRESS_BITS / 4`-bit virtual-RA chunks.
pub const WIDE_ONE_HOT_CONFIG: JoltOneHotConfig = JoltOneHotConfig {
    log_k_chunk: 8,
    lookups_ra_virtual_log_k_chunk: 32,
};

const _: () = assert!(
    NARROW_ONE_HOT_CONFIG.lookup_virtual_chunk_bits() * 8 == LOOKUP_ADDRESS_BITS
        && WIDE_ONE_HOT_CONFIG.lookup_virtual_chunk_bits() * 4 == LOOKUP_ADDRESS_BITS
);

/// The one-hot chunking policy: [`NARROW_ONE_HOT_CONFIG`] below
/// [`ONEHOT_CHUNK_THRESHOLD_LOG_T`], [`WIDE_ONE_HOT_CONFIG`] at or above it.
/// The prover derives its wire config from this; the verifier admits either
/// regime at any trace length ([`is_admissible_one_hot_config`]), since the
/// wide regime is also exercised below the threshold (the legacy parity
/// fixtures force it, and the packed setup pins `K` independently).
pub const fn one_hot_config_policy(log_t: usize) -> JoltOneHotConfig {
    if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        NARROW_ONE_HOT_CONFIG
    } else {
        WIDE_ONE_HOT_CONFIG
    }
}

/// Whether a proof-carried one-hot config is one of the two policy regimes.
pub fn is_admissible_one_hot_config(config: JoltOneHotConfig) -> bool {
    config == NARROW_ONE_HOT_CONFIG || config == WIDE_ONE_HOT_CONFIG
}

/// The read-write checking phase-split policy: every cycle variable in phase
/// 1 and every address variable in phase 2, for RAM (`ram_log_k` address bits)
/// and registers ([`REGISTER_ADDRESS_BITS`]). The prover derives its wire
/// config from this and the verifier requires an exact match. `None` when a
/// round count does not fit the wire's `u8`.
pub fn read_write_config_policy(log_t: usize, ram_log_k: usize) -> Option<JoltReadWriteConfig> {
    let log_t = u8::try_from(log_t).ok()?;
    Some(JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: log_t,
        ram_rw_phase2_num_rounds: u8::try_from(ram_log_k).ok()?,
        registers_rw_phase1_num_rounds: log_t,
        registers_rw_phase2_num_rounds: u8::try_from(REGISTER_ADDRESS_BITS).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
