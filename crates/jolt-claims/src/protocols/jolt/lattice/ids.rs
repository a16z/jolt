use serde::{Deserialize, Serialize};

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
