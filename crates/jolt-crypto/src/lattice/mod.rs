//! Generic Ajtai (Module-SIS) commitment pieces.
//!
//! [`derive_public_matrix_prefix`] expands a [`PublicMatrixSeed`] into the
//! flat public stream that Ajtai matrices are cut from.

mod matrix;
mod modulus;

pub use matrix::{
    check_public_matrix_prefix, derive_public_matrix_prefix, PublicMatrixDerivation,
    PublicMatrixSeed,
};
pub use modulus::field_modulus_be_bytes;
