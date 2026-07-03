//! The commitment axis of the Jolt PIOP as a claim-type family.
//!
//! Four claim groups change shape between the homomorphic (per-polynomial
//! commitments) and packed lattice (one packed witness) modes. Each mode
//! names its concrete claim structs here; stage outputs select them through
//! one `M: JoltCommitmentMode` generic, so a proof for the wrong mode fails
//! to typecheck or deserialize instead of being runtime-rejected. Everything
//! mode-invariant keeps its concrete struct.

use core::fmt::Debug;

use crate::NoOutputs;

use super::lattice::relations::{
    booleanity::LatticeBooleanityOutputClaims,
    chunk_reconstruction::ChunkReconstructionOutputClaims,
    inc_virtualization::IncVirtualizationOutputClaims,
};
use super::lattice::LATTICE_BYTECODE_VAL_STAGES;
use super::relations::booleanity::BooleanityOutputClaims;
use super::relations::claim_reductions::increments::IncClaimReductionOutputClaims;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

pub trait JoltCommitmentMode:
    Clone + Copy + Debug + Default + PartialEq + Eq + Send + Sync + 'static
{
    /// Bytecode read-raf val stages (the lattice mode adds the store-flag
    /// stage consumed by `IncVirtualization`).
    const BYTECODE_VAL_STAGES: usize;

    /// Stage-6 increment reduction outputs: `RamInc`/`RdInc` claims in the
    /// homomorphic mode, `FusedInc` + store selector under the lattice mode.
    type IncOutputs<C>: Clone + Debug + PartialEq + Eq + Send + Sync
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;

    /// Stage-6 booleanity outputs (the lattice fold extends over the
    /// unsigned-inc chunk/msb columns).
    type BooleanityOutputs<C>: Clone + Debug + PartialEq + Eq + Send + Sync
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;

    /// Stage-7 chunk reconstruction outputs (lattice mode only; empty in the
    /// homomorphic mode).
    type ChunkReconstructionOutputs<C>: Clone + Debug + PartialEq + Eq + Send + Sync
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;
}

/// Per-polynomial commitments, RLC final opening (Dory/HyperKZG).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaseJolt;

impl JoltCommitmentMode for BaseJolt {
    const BYTECODE_VAL_STAGES: usize = NUM_BYTECODE_VAL_STAGES;

    type IncOutputs<C>
        = IncClaimReductionOutputClaims<C>
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;

    type BooleanityOutputs<C>
        = BooleanityOutputClaims<C>
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;

    type ChunkReconstructionOutputs<C>
        = NoOutputs<C>
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;
}

/// One packed one-hot witness, reduction-sumcheck final opening (Akita).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LatticeJolt;

impl JoltCommitmentMode for LatticeJolt {
    const BYTECODE_VAL_STAGES: usize = LATTICE_BYTECODE_VAL_STAGES;

    type IncOutputs<C>
        = IncVirtualizationOutputClaims<C>
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;

    type BooleanityOutputs<C>
        = LatticeBooleanityOutputClaims<C>
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;

    type ChunkReconstructionOutputs<C>
        = ChunkReconstructionOutputClaims<C>
    where
        C: Clone + Debug + PartialEq + Eq + Send + Sync;
}
