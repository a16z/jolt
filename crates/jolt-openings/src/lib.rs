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
//! - **Two batching protocols.** [`HomomorphicBatch`] settles same-point
//!   claims by RLC-combining per-polynomial commitments (requires
//!   [`AdditivelyHomomorphic`]). [`prove_packed_openings`] /
//!   [`verify_packed_openings`] settle claims at mutually independent points
//!   on prefix-packed commitments through one joint reduction sumcheck plus
//!   one native opening per commitment object — no homomorphism required.
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
//! The packed opening is a pair of free functions rather than a
//! [`BatchOpeningScheme`] impl: its statement spans several commitment
//! objects, each opening against its own borrowed setup, which does not fit
//! the trait's single owned setup.

mod claims;
mod error;
mod packing;
mod schemes;

pub use claims::{EvaluationClaim, VerifierOpeningClaim, ZkEvaluationClaim};
pub use error::OpeningsError;
pub use packing::{
    prove_packed_openings, verify_packed_openings, PackedObjectGroup, PackedOpeningProof,
    PackedPolynomial, PackedProverGroup, PackedProverObject, PackedVerifierObject,
    PrefixPackedStatement, PrefixPacking, PrefixSlot,
};

pub use schemes::{
    AdditivelyHomomorphic, BatchOpeningScheme, CommitmentScheme, GroupCommitmentMetadata,
    GroupSetupMetadata, HomomorphicBatch, StreamingCommitment, ZkBatchOpening,
    ZkBatchOpeningScheme, ZkOpeningScheme, ZkStreamingCommitment,
};
