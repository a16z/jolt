//! Ajtai (Module-SIS) public matrices.
//!
//! [`derive_public_matrix_prefix`] expands a [`PublicMatrixSeed`] into the
//! flat public stream that Ajtai matrices are cut from, and
//! [`check_public_matrix_prefix`] checks a stored prefix against its seed.

mod matrix;

pub use matrix::{
    check_public_matrix_prefix, derive_public_matrix_prefix, derive_public_matrix_range,
    PublicMatrixDerivation, PublicMatrixSeed,
};
