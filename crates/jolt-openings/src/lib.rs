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
mod packing;
mod packing_layout;
mod packing_view;
mod reduction;
mod schemes;

pub use claims::{EvaluationClaim, ProverOpeningClaim, VerifierOpeningClaim};
pub use error::OpeningsError;
pub use packing::{
    has_packing_view, prove_packing_reduction, prove_sparse_packing_reduction,
    prove_sparse_packing_reduction_from_entries, validate_packing_statement,
    verify_packing_reduction, PackingAddress, PackingBatch, PackingBatchProof, PackingFamily,
    PackingLayout, PackingProverReduction, PackingProverSetup, PackingReductionProof,
    PackingSetupParams, PackingSource, PackingVerifierReduction, PackingVerifierSetup,
};
pub use packing_layout::{
    packing_witness_source_polynomial, PackingAdviceKind, PackingAlphabet, PackingAlphabetCounts,
    PackingCellAddress, PackingDomainCellCounts, PackingFactDomain, PackingFamilyId,
    PackingFamilySpec, PackingLayoutAudit, PackingLayoutError, PackingLayoutFamily,
    PackingViewKind, PackingWitnessLayout, PackingWitnessSource, SparsePackingWitness,
};
pub use packing_view::{
    PackingViewCatalog, PackingViewDigest, PackingViewEntry, PackingViewError, PackingViewFormula,
    PackingViewTerm, PackingViewValidity,
};
pub use reduction::{reduce_prover, reduce_verifier, rlc_combine, rlc_combine_scalars};

pub use schemes::{
    AdditivelyHomomorphic, BatchOpeningClaim, BatchOpeningResult, BatchOpeningScheme,
    BatchOpeningStatement, CommitmentLayoutDigest, CommitmentScheme, PackingFamilyRef, PackingTerm,
    PhysicalView, StreamingCommitment, ZkBatchOpeningScheme, ZkOpeningScheme,
};
