use serde::{Deserialize, Serialize};

use super::super::{JoltAdviceKind, JoltCommittedPolynomial};

/// Challenge drawn by the [`IncVirtualization`](crate::protocols::jolt::JoltRelationId::IncVirtualization)
/// relation.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncVirtualizationChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IncVirtualizationPublic {
    EqRamReadWrite,
    EqRamValCheck,
    EqRegistersReadWrite,
    EqRegistersValEvaluation,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UnsignedIncChunkReconstructionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UnsignedIncChunkReconstructionPublic {
    /// `eq(r_booleanity_address, r_address)` — reduces the chunk openings
    /// produced at the booleanity address point to this relation's bound
    /// address point.
    EqBooleanityAddress,
    /// The identity MLE `Σ_bit 2^bit · r_address[bit]` at the bound address
    /// point — decodes a one-hot chunk opening into its symbol value.
    IdentityAtAddress,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AdviceBytesValidityChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AdviceBytesValidityPublic {
    /// `eq` over the full `(symbol ‖ limb ‖ word)` cell domain at the bound
    /// point — weights the booleanity leg.
    EqCell,
    /// `eq` over the `(limb ‖ word)` sub-domain at the bound point — weights
    /// the per-byte-position hamming leg (symbol variables are summed, not
    /// eq-bound).
    EqLimbWord,
}

/// The register-selector lanes of a bytecode row, in committed lane order.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeRegisterLane {
    Rs1,
    Rs2,
    Rd,
}

impl BytecodeRegisterLane {
    pub const ALL: [Self; 3] = [Self::Rs1, Self::Rs2, Self::Rd];
}

/// One column of a packed lattice witness — the `Id` fed to
/// `jolt-openings::PrefixPacking`.
///
/// A column with in-protocol openings is a (lattice-mode) committed
/// polynomial and wraps its [`JoltCommittedPolynomial`] id; a precommitted
/// sub-column is only ever reached through a decode view and gets its own
/// variant with no opening-id identity.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LatticeColumn {
    Committed(JoltCommittedPolynomial),
    BytecodeRegisterSelector {
        chunk: usize,
        lane: BytecodeRegisterLane,
    },
    BytecodeCircuitFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeInstructionFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeLookupSelector {
        chunk: usize,
    },
    BytecodeRafFlag {
        chunk: usize,
    },
    BytecodeUnexpandedPcBytes {
        chunk: usize,
    },
    BytecodeImmBytes {
        chunk: usize,
    },
    ProgramImageBytes,
}

impl LatticeColumn {
    pub fn advice_bytes(kind: JoltAdviceKind) -> Self {
        Self::Committed(advice_bytes_polynomial(kind))
    }
}

impl From<JoltCommittedPolynomial> for LatticeColumn {
    fn from(polynomial: JoltCommittedPolynomial) -> Self {
        Self::Committed(polynomial)
    }
}

pub fn advice_bytes_polynomial(kind: JoltAdviceKind) -> JoltCommittedPolynomial {
    match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdviceBytes,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdviceBytes,
    }
}
