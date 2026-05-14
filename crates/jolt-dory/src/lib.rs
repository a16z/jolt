//! Dory polynomial commitment scheme implementation for the Jolt zkVM.
//!
//! Wraps the [Dory](https://eprint.iacr.org/2020/1274) polynomial commitment
//! scheme for BN254 with transparent setup, logarithmic proof size, and
//! logarithmic verification. Supports source-batch commitment and additive
//! homomorphism for batch opening reduction.
//!
//! Implements [`CommitmentScheme`](jolt_openings::CommitmentScheme),
//! [`AdditivelyHomomorphic`](jolt_openings::AdditivelyHomomorphic),
//! [`ZkOpeningScheme`](jolt_openings::ZkOpeningScheme) from `jolt-openings`.
//!
//! # Public API
//!
//! - [`DoryScheme`] — implements the four PCS traits. Static methods:
//!   `setup_prover` and `setup_verifier`. Use
//!   [`ZkOpeningScheme::commit_zk`](jolt_openings::ZkOpeningScheme::commit_zk)
//!   for hiding commitments. Also implements
//!   [`DeriveSetup<DoryProverSetup>`](jolt_crypto::DeriveSetup) for
//!   [`PedersenSetup<Bn254G1>`](jolt_crypto::PedersenSetup) (use
//!   `PedersenSetup::derive(&prover_setup, capacity)`).
//! - [`DoryCommitment`] — BN254 pairing target element (GT).
//! - [`DoryProof`] — single opening proof.
//! - [`DoryProverSetup`] / [`DoryVerifierSetup`] — prover and verifier SRS.
//! - [`DoryHint`] — row commitments, row width, and commitment blind reusable as opening proof hint.

mod routines;
mod scheme;
mod transcript;
mod types;

pub use scheme::{ArkFr, DoryScheme};
pub use types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};
