//! PCS traits and batched opening verification for the Jolt zkVM.
//!
//! ```text
//!             Commitment              (jolt-crypto: Output type)
//!                 |
//!         CommitmentScheme            (+ Field, Proof, commit/open/verify)
//!                 |
//!     AdditivelyHomomorphic           (+ combine)
//!                 |
//!       StreamingCommitment           (+ begin/feed/finish)
//! ```
//!
//! [`ProverClaim`], [`VerifierClaim`], and the backend-aware [`OpeningClaim`]
//! are stateless data accumulated by the protocol orchestrator.
//! [`OpeningVerification`] is the PCS-owned single-shot trait that proves
//! / verifies an entire batch of opening claims against one
//! [`OpeningVerification::BatchProof`] object: homomorphic schemes
//! (Mock / HyperKZG / Dory) delegate to [`homomorphic_prove_batch`] /
//! [`homomorphic_verify_batch_with_backend`]; non-homomorphic schemes
//! (FRI, Hachi-style lattice batching) wire their own scheme-specific
//! batch routines.

mod backend;
mod claims;
mod error;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod schemes;
mod verification;

pub use backend::{
    BackendError, CommitmentBackend, CommitmentOrigin, FieldBackend, ScalarOrigin,
};
pub use claims::{OpeningClaim, ProverClaim, VerifierClaim};
pub use error::OpeningsError;
pub use schemes::{AdditivelyHomomorphic, CommitmentScheme, StreamingCommitment, ZkOpeningScheme};
pub use verification::{
    homomorphic_prove_batch, homomorphic_verify_batch_with_backend, rlc_combine,
    rlc_combine_scalars, OpeningVerification,
};

pub use jolt_crypto::Commitment;
