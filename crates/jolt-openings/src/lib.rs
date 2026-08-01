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
//! - **Fixed prefix packing.** [`PrefixPackedLayout`] reduces equal-point
//!   logical claims to one physical opening. Protocol crates own semantic
//!   column order and any zero-prefix embeddings.
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
//! HomomorphicBatch<PCS>: BatchOpeningScheme
//!   Statement = Vec<VerifierOpeningClaim<...>>
//! ```
//!
mod claims;
mod error;
mod prefix;
mod schemes;

pub use claims::{EvaluationClaim, VerifierOpeningClaim, ZkEvaluationClaim};
pub use error::OpeningsError;
pub use prefix::{PrefixPackedClaims, PrefixPackedLayout};

pub use schemes::{
    AdditivelyHomomorphic, BatchOpeningScheme, CommitmentScheme, GroupCommitmentMetadata,
    GroupSetupMetadata, HomomorphicBatch, StreamingCommitment, ZkBatchOpening,
    ZkBatchOpeningScheme, ZkOpeningScheme, ZkStreamingCommitment,
};
