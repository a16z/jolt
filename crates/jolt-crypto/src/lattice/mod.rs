//! Generic Ajtai (Module-SIS) commitment pieces.
//!
//! [`derive_public_matrix_prefix`] expands a [`PublicMatrixSeed`] into the
//! flat public stream that Ajtai matrices are cut from, and
//! [`check_public_matrix_prefix`] checks a stored prefix against its seed;
//! [`MatrixView`] cuts one matrix from the stream, [`PreparedMatrix`] applies
//! it in constant time, and [`commit`] is one [`AjtaiLevel`].
//! [`required_randomness_columns`] sizes a level's hiding randomness.

mod ajtai;
mod matrix;
mod randomness;
mod view;

pub use ajtai::{commit, AjtaiLevel, GadgetDigits, PreparedMatrix, MAX_CENTERED_BOUND};
pub use matrix::{
    check_public_matrix_prefix, derive_public_matrix_prefix, derive_public_matrix_range,
    PublicMatrixDerivation, PublicMatrixSeed,
};
#[cfg(feature = "sample")]
pub use randomness::sample_box_digits;
pub use randomness::{required_randomness_columns, HIDING_SECURITY_BITS};
pub use view::{MatrixShape, MatrixView};

/// Ring dimension of Whiteout's challenge ring: its ring-linear maps must be
/// modules over it.
pub const CHALLENGE_RING_DIM: usize = 64;

#[cfg(test)]
mod tests;
