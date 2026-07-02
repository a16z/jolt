//! PCS traits and batch openings for the Jolt zkVM.
//!
//! Abstract interfaces for polynomial commitment schemes (PCS) and batching
//! adapters. Protocol code is written generically over the PCS with zero
//! implementation leakage.
//!
//! # Design
//!
//! - **Stateless.** No accumulators. A batch opening receives an explicit
//!   statement plus borrowed prover-side source data needed to open it.
//! - **Batch openings are an extension trait.** [`BatchOpeningScheme`] lets a
//!   protocol adapter own its batching strategy while preserving the ordinary
//!   single-opening API for the underlying PCS.
//! - **Statement shape is scheme-specific.** [`HomomorphicBatch`] uses
//!   `Vec<VerifierOpeningClaim<_, _>>`, because each logical polynomial has its
//!   own commitment. [`PackedBatch`] uses [`PrefixPackedStatement`], which
//!   carries one packed commitment and logical `(id, evaluation)` claims.
//!
//! # Trait Hierarchy
//!
//! ```text
//!                 Commitment              (jolt-crypto: Output type)
//!                     │
//!             CommitmentScheme            (+ Field, Proof, commit/open/verify)
//!        ╱          │          ╲
//! Additively   Streaming       ZkOpeningScheme
//! Homomorphic  Commitment            │
//!       │                       ZkStreamingCommitment
//!       │
//! HomomorphicBatch<PCS>        PackedBatch<PCS, Id>
//!   Statement = Vec<...>       Statement = PrefixPackedStatement<...>
//!        ╲                         ╱
//!             BatchOpeningScheme
//!
//! Batching is selected explicitly through [`HomomorphicBatch`] or
//! [`PackedBatch`].
//! ```

mod claims;
mod error;
mod packing;
mod schemes;

pub use claims::{EvaluationClaim, VerifierOpeningClaim, ZkEvaluationClaim};
pub use error::OpeningsError;
pub use packing::{
    PackedBatchProof, PackedPolynomial, PrefixPackedProverSetup, PrefixPackedStatement,
    PrefixPackedVerifierSetup, PrefixPacking, PrefixSlot,
};

pub use schemes::{
    AdditivelyHomomorphic, BatchOpeningScheme, CommitmentScheme, HomomorphicBatch, PackedBatch,
    StreamingCommitment, ZkBatchOpeningScheme, ZkOpeningScheme, ZkStreamingCommitment,
};
