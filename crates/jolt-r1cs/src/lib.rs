//! Sparse R1CS matrices and constraint construction for Jolt primitives.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

pub mod builder;
pub mod constraint;
#[cfg(feature = "fp128-bn254")]
pub mod fp128_bn254;

pub use builder::{LinearCombination, R1csBuilder, R1csBuilderError, Variable};
pub use constraint::{
    ConstraintMatrices, ConstraintMatrixEvalError, MatrixColumnContributions, SparseRow,
    WeightedMatrixColumns,
};
