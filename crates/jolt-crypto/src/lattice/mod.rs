//! Generic Ajtai (Module-SIS) commitment pieces.
//!
//! [`derive_public_matrix_prefix`] expands a [`PublicMatrixSeed`] into the
//! flat public stream that Ajtai matrices are cut from; [`MatrixView`] cuts
//! one, [`PreparedMatrix`] applies it in constant time, and [`commit`] is
//! one [`AjtaiLevel`]. [`TwoLevelLayout`] describes Akita's two-level
//! commitment without computing it.

mod ajtai;
mod layout;
mod matrix;
mod randomness;
mod view;

pub use ajtai::{commit, AjtaiLevel, GadgetDigits, PreparedMatrix};
pub use layout::{BlockGeometry, SliceCount, TwoLevelLayout, CHALLENGE_RING_DIM};
pub use matrix::{
    check_public_matrix_prefix, derive_public_matrix_prefix, PublicMatrixDerivation,
    PublicMatrixSeed,
};
pub use randomness::{required_randomness_columns, sample_box_digits, HIDING_SECURITY_BITS};
pub use view::{MatrixShape, MatrixView};

#[cfg(test)]
mod tests;
