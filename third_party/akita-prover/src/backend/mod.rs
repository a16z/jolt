//! Polynomial backends and prover-only witness state.

mod dense;
mod field_reduction;
mod multilinear_polynomial;
pub(crate) mod onehot;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod poly_helpers;
mod recursive_hint;
mod recursive_witness;
pub(crate) mod sparse_ring;
mod tensor_fold;

pub use dense::DensePoly;
pub use field_reduction::{tensor_pack_recursive_witness, FoldInputPoly};
pub use multilinear_polynomial::MultilinearPolynomial;
pub use onehot::{MultiChunkEntry, OneHotIndex, OneHotPoly, SingleChunkEntry};
pub use recursive_hint::RecursiveCommitmentHintCache;
pub use recursive_witness::{RecursiveWitnessFlat, SuffixWitness};
pub use sparse_ring::{SparseRingBlockEntry, SparseRingPoly};
