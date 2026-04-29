//! Dory polynomial commitment scheme implementation for the Jolt zkVM.
//!
//! Wraps the [Dory](https://eprint.iacr.org/2020/1274) polynomial commitment
//! scheme for BN254 with transparent setup, logarithmic proof size, and
//! logarithmic verification. Supports streaming commitment and additive
//! homomorphism for batch opening reduction.
//!
//! Implements [`CommitmentScheme`](jolt_openings::CommitmentScheme),
//! [`AdditivelyHomomorphic`](jolt_openings::AdditivelyHomomorphic),
//! [`StreamingCommitment`](jolt_openings::StreamingCommitment), and
//! [`ZkOpeningScheme`](jolt_openings::ZkOpeningScheme) from `jolt-openings`.
//!
//! # Public API
//!
//! - [`DoryScheme`] — implements the four PCS traits. Static methods:
//!   `setup_prover`, `setup_verifier`. Also implements
//!   [`DeriveSetup<DoryProverSetup>`](jolt_crypto::DeriveSetup) for
//!   [`PedersenSetup<Bn254G1>`](jolt_crypto::PedersenSetup) (use
//!   `PedersenSetup::derive(&prover_setup, capacity)`).
//! - [`DoryCommitment`] — BN254 pairing target element (GT).
//! - [`DoryProof`] — single opening proof.
//! - [`DoryProverSetup`] / [`DoryVerifierSetup`] — prover and verifier SRS.
//! - [`DoryPartialCommitment`] — intermediate state for streaming commitment.
//! - [`DoryHint`] — row commitments reusable as opening proof hint.

mod scheme;
mod streaming;
mod transcript;
mod types;

pub use scheme::DoryScheme;
pub use types::{
    DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup, DoryVerifierSetup,
};
