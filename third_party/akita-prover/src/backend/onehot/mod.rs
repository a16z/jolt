//! One-hot polynomial: sparse witness with at most one nonzero field
//! element per chunk of size `onehot_k`.
//!
//! [`OneHotPoly`] is a backend for [`AkitaPolyOps`](akita_prover::AkitaPolyOps)
//! that implements the four prover operations (ring evaluation, per-block
//! fold, decompose+fold, and inner-Ajtai commit) by iterating only over
//! the nonzero monomial positions.
//!
//! # Module layout
//!
//! The module is organised as cohesive private submodules — entry types,
//! flat block storage, and the polynomial + its [`AkitaPolyOps`] impl.
//!
//!   - [`OneHotIndex`]: a tiny trait implemented for `u8`/`u16`/`u32`/
//!     `usize` so callers can hand [`OneHotPoly::new`] a `Vec<Option<I>>`
//!     at the narrowest width that fits their hot positions.
//!   - Per-block entry types: [`SingleChunkEntry`] (packed `u32 + u16`,
//!     used when each ring element covers at most one hot element —
//!     i.e. `K >= D && D | K`) and [`MultiChunkEntry`] (`u32 +
//!     Vec<u16>`, used when a ring element can cover zero to many
//!     hot elements — i.e. `K < D` with `K | D`). Coefficient indices fit
//!     in `u16` because the supported ring degrees are small; the
//!     bound is enforced in [`OneHotPoly::build_blocks_inner`].
//!   - [`FlatBlocks<E>`]: a container storing the
//!     variable-length per-block entry lists in one contiguous `Vec<E>`
//!     plus a `Vec<u32>` offsets array.
//!   - [`OneHotBlocks`]: a two-variant enum that wraps the built
//!     `FlatBlocks<E>` so [`OneHotPoly`]'s ops can dispatch to the right
//!     kernel based on the actual layout in use.
//!   - [`OneHotPoly<F, D, I>`]: the caller-facing polynomial.

use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
};
use akita_types::{
    CleartextWitnessProof, FlatDigitBlocks, FlatRingVec, FpExtEncoding, RingMatrixView,
};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use super::sparse_ring::SparseRingCoeff;
use crate::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use crate::backend::tensor_fold::{fill_rotated_tensor_challenge, narrow_tensor_accum_to_i32};
use crate::compute::{
    CommitmentComputeBackend, FlatBlockTable, OneHotCommitBlocks, OneHotCommitRowsPlan,
};
use crate::kernels::linear::decompose_rows_i8_into;
use crate::{
    AkitaPolyOps, CommitInnerWitness, DecomposeFoldWitness, FoldInputPoly, SparseRingPoly,
};

/// Wide accumulators use 16-bit chunks in `i32` limbs, so they can safely
/// absorb at most 32,768 unit-scale additions before overflow.
pub(super) const MAX_WIDE_SHIFT_ACCUMULATIONS: usize = 1 << 15;

mod accumulate;
mod blocks;
mod column_sweep;
mod decompose_fold;
mod entries;
mod fold;
mod inner_ajtai;
mod ops;
mod poly;
#[cfg(test)]
pub(crate) mod test_helpers;
#[cfg(test)]
mod tests;

pub(crate) use blocks::{FlatBlocks, OneHotBlocks};
pub(crate) use column_sweep::column_sweep_ajtai_onehot;
pub(super) use entries::{shift_accumulation_count, OneHotEntry};
pub use entries::{MultiChunkEntry, OneHotIndex, SingleChunkEntry};
#[cfg(test)]
use inner_ajtai::{inner_ajtai_wide_onehot, inner_ajtai_wide_single_chunk_tiled};
pub use poly::OneHotPoly;
