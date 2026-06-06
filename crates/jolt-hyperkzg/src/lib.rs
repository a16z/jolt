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
//!
//! # SRS handling
//!
//! HyperKZG uses a structured reference string with a KZG trapdoor `beta`.
//! Production runtimes should import ceremony-generated files named
//! `hyperkzg_{k}.srs`, where `k` is the exponent for `2^k` supported
//! evaluations. `setup` and `setup_from_secret` are for tests or trusted setup
//! tooling; live proving/verifying code should use `read_srs_file` or
//! `read_srs_from_dir` so it never observes `beta`.

pub mod error;
pub mod kzg;
pub mod scheme;
mod setup;
pub mod types;

pub use scheme::HyperKZGScheme;
pub use types::{
    HyperKZGCommitment, HyperKZGOpeningHint, HyperKZGProof, HyperKZGProofKind,
    HyperKZGProofPayload, HyperKZGProverSetup, HyperKZGSrsKind, HyperKZGVerifierSetup,
};
