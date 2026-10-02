//! HyperKZG multilinear polynomial commitment scheme.
//!
//! HyperKZG reduces multilinear polynomial commitments to univariate KZG using
//! the Gemini transformation (section 2.4.2 of <https://eprint.iacr.org/2022/420.pdf>),
//! operating directly on evaluation-form polynomials (no FFT/interpolation).
//!
//! This crate is generic over `PairingGroup` from `jolt-crypto` and implements
//! the `CommitmentScheme` and `AdditivelyHomomorphic` traits from `jolt-openings`.
//!
//! # Protocol overview
//!
//! 1. **Commit**: MSM of evaluations against SRS G1 powers (treating the
//!    multilinear evaluation table as univariate coefficients).
//! 2. **Open** (Gemini reduction):
//!    - Phase 1: Fold two variables per level, producing intermediate commitments.
//!    - Phase 2: Derive challenge `r` and points `[r, ir, -r, -ir, r^4]`.
//!    - Phase 3: Batch KZG opening of all intermediate polynomials at five points.
//! 3. **Verify**: Check evaluation consistency across the five evaluation vectors,
//!    then batch KZG pairing check.

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

pub mod error;
pub mod kzg;
pub mod multi_open;
pub mod scheme;
pub mod types;

pub use multi_open::{
    open_variable_batch, verify_variable_batch, verify_variable_batch_observed,
    VariableBatchKzgProof,
};
pub use scheme::HyperKZGScheme;
pub use types::{
    HyperKZGCommitment, HyperKZGProof, HyperKZGProverSetup, HyperKZGVerifierSetup,
    NoopVerifierObserver, VerifierObserver,
};
