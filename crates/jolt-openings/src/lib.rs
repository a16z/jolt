//! PCS traits and opening reduction for the Jolt zkVM.
//!
//! Abstract interfaces for polynomial commitment schemes (PCS) and a reduction
//! framework for batching opening claims. Protocol code is written generically
//! over the PCS with zero implementation leakage.
//!
//! # Design
//!
//! - **Stateless.** No accumulators. Claims are plain data ([`ProverClaim`],
//!   [`VerifierClaim`]) collected by the caller in `Vec`s.
//! - **Reduction is separate from proving.** [`reduce_prover`] /
//!   [`reduce_verifier`] transform claims (many → fewer) via RLC.
//!   The PCS opens the reduced claims.
//! - **No batching in PCS traits.** Batching is a reduction concern, not a
//!   PCS property.
//!
//! # Trait Hierarchy
//!
//! ```text
//!                 Commitment              (jolt-crypto: Output type)
//!                     │
//!             CommitmentScheme            (+ Field, Proof, commit/open/verify)
//!                ╱        ╲
//! AdditivelyHomomorphic   ZkOpeningScheme
//!       (+ combine)        (+ open_zk/verify_zk)
//!             │
//!   StreamingCommitment
//!     (+ begin/feed/finish)
//! ```

mod claims;
mod error;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod reduction;
mod schemes;

pub use claims::{ProverClaim, VerifierClaim};
pub use error::OpeningsError;
pub use reduction::{reduce_prover, reduce_verifier, rlc_combine, rlc_combine_scalars};

pub use schemes::{AdditivelyHomomorphic, CommitmentScheme, StreamingCommitment, ZkOpeningScheme};
