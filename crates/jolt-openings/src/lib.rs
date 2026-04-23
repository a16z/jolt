//! PCS traits and batched-opening helpers for the Jolt zkVM.
//!
//! ```text
//!                CommitmentSchemeVerifier   (verifier base)
//!                          │
//!         ┌────────────────┼─────────────────┐
//!         │                │                 │
//!         │  AdditivelyHomomorphicVerifier   ZkOpeningSchemeVerifier
//!         │                │                 │
//!     CommitmentScheme     │                 │
//!         │                │                 │
//!         ├────────────────┤                 │
//!         │                │                 │
//!         │  AdditivelyHomomorphic           │
//!         │                │                 │
//!         │   ┌────────────┘                 │
//!         │   │                              │
//!         │   │           ZkOpeningScheme  ──┘
//!         │   │
//!     StreamingCommitment
//! ```
//!
//! Verifier-side traits live above the prover-side extensions so verifier-only
//! crates depend on the minimal surface they actually need. [`OpeningClaim`]
//! and [`ProverClaim`] are the stateless data structs collected by protocol
//! orchestrators; [`homomorphic_prove_batch`] / [`homomorphic_verify_batch`]
//! are recommended bodies for `prove_batch` / `verify_batch` on schemes that
//! follow the standard "group-by-point + RLC" template.

mod claims;
mod error;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod schemes;
mod verification;

pub use claims::{OpeningClaim, ProverClaim};
pub use error::OpeningsError;
pub use schemes::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, CommitmentScheme,
    CommitmentSchemeVerifier, StreamingCommitment, ZkOpeningScheme, ZkOpeningSchemeVerifier,
};
pub use verification::{
    homomorphic_prove_batch, homomorphic_verify_batch, rlc_combine, rlc_combine_scalars,
};
