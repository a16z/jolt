//! Verifier-selected protocol configuration.
//!
//! Both protocol axes are fixed at compile time — the `zk` feature selects
//! BlindFold, the `akita` feature selects the packed commitment mode — so one
//! compiled verifier runs exactly one protocol. A proof self-describes its
//! axes and [`validate_proof_config`] rejects a mismatch fail-closed.

use serde::{Deserialize, Serialize};

use crate::VerifierError;

#[cfg(all(feature = "zk", feature = "akita"))]
compile_error!(
    "the `zk` and `akita` features are mutually exclusive: no zk protocol exists over the \
     packed commitment axis (a lattice-friendly hiding commitment is a future workstream)"
);

#[cfg(all(feature = "implicit-carry", feature = "akita"))]
compile_error!(
    "the `implicit-carry` and `akita` features are mutually exclusive for now: the packed \
     witness layout has no Carry column (see specs/implicit-carry-handling.md)"
);

#[cfg(all(feature = "implicit-carry", feature = "zk"))]
compile_error!(
    "`implicit-carry` + `zk` is not supported yet: the BlindFold constraints for the carry \
     relations are staged for a follow-up and fail closed until then"
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
    /// Packed one-hot witnesses per commitment object, reduction-sumcheck
    /// batch opening (no homomorphism required).
    Packed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub commitment: CommitmentConfig,
    /// The ISA axis: present only in implicit-carry builds so the feature-off
    /// wire format is unchanged. Cross-build proofs fail closed at
    /// deserialization (the config frames differ).
    #[cfg(feature = "implicit-carry")]
    pub isa: IsaConfig,
}

/// The ISA axis of the protocol (implicit-carry builds only).
#[cfg(feature = "implicit-carry")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsaConfig {
    /// RV64IMAC extended with the ADDC/MULC implicit-carry instructions and
    /// the carry proof machinery.
    Rv64imacImplicitCarry,
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
            #[cfg(feature = "implicit-carry")]
            isa: IsaConfig::Rv64imacImplicitCarry,
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

/// The one protocol this build verifies.
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    commitment: SELECTED_COMMITMENT_CONFIG,
    #[cfg(feature = "implicit-carry")]
    isa: IsaConfig::Rv64imacImplicitCarry,
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
