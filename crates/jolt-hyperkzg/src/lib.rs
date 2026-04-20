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
//!    - Phase 1: Fold the multilinear polynomial `ell - 1` times, producing
//!      intermediate polynomial commitments.
//!    - Phase 2: Derive challenge `r` and evaluation points `[r, -r, r^2]`.
//!    - Phase 3: Batch KZG opening of all intermediate polynomials at three points.
//! 3. **Verify**: Check evaluation consistency across the three evaluation vectors,
//!    then batch KZG pairing check.

pub mod error;
pub mod kzg;
pub mod scheme;
pub mod types;

pub use scheme::HyperKZGScheme;
pub use types::{HyperKZGCommitment, HyperKZGProof, HyperKZGProverSetup, HyperKZGVerifierSetup};
