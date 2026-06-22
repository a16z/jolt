//! PCS traits and opening reduction for the Jolt zkVM.
//!
//! Abstract interfaces for polynomial commitment schemes (PCS) and a reduction
//! framework for batching opening claims. Protocol code is written generically
//! over the PCS with zero implementation leakage.
//!
//! # Design
//!
//! - **Stateless.** No accumulators. Claims are plain data
//!   ([`ProverOpeningClaim`], [`VerifierOpeningClaim`]) collected by the caller
//!   in `Vec`s.
//! - **Reduction is separate from proving.** [`reduce_prover`] /
//!   [`reduce_verifier`] transform claims (many → fewer) via RLC.
//!   The PCS opens the reduced claims.
//! - **Same-point batch openings are an extension trait.**
//!   [`BatchOpeningScheme`] lets a PCS own its physical batching strategy while
//!   preserving the ordinary single-opening API.
//!
//! # Trait Hierarchy
//!
//! ```text
//!                 Commitment              (jolt-crypto: Output type)
//!                     │
//!             CommitmentScheme            (+ Field, Proof, commit/open/verify)
//!        ╱         │         ╲
//! Additively   BatchOpening   ZkOpeningScheme
//! Homomorphic      Scheme          │
//!       │            │        ZkBatchOpeningScheme
//!   StreamingCommitment
//! ```

mod claims;
mod error;
mod homomorphic_batch;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
#[cfg(any(test, feature = "test-utils"))]
mod packed_combine;
mod packed_linear;
mod reduction;
mod schemes;

pub use claims::{EvaluationClaim, ProverOpeningClaim, VerifierOpeningClaim};
pub use error::OpeningsError;
#[cfg(any(test, feature = "test-utils"))]
pub use packed_combine::PackedCombine;
pub use packed_linear::{
    has_packed_linear_view, prove_packed_linear_reduction, prove_sparse_packed_linear_reduction,
    validate_packed_linear_statement, verify_packed_linear_reduction, PackedLinearAddress,
    PackedLinearBatch, PackedLinearBatchBackend, PackedLinearBatchProof, PackedLinearFamily,
    PackedLinearLayout, PackedLinearProverReduction, PackedLinearReductionProof,
    PackedLinearVerifierReduction, PackedLinearWitnessSource,
};
pub use reduction::{reduce_prover, reduce_verifier, rlc_combine, rlc_combine_scalars};

pub use schemes::{
    AdditivelyHomomorphic, BatchOpeningClaim, BatchOpeningResult, BatchOpeningScheme,
    BatchOpeningStatement, CommitmentLayoutDigest, CommitmentScheme, PackedFamilyRef,
    PackedLinearTerm, PhysicalView, StreamingCommitment, ZkBatchOpeningScheme, ZkOpeningScheme,
};
