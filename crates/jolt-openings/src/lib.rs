//! PCS traits and opening reduction for the Jolt zkVM.
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
//! [`ProverClaim`] and [`VerifierClaim`] are stateless data collected by the
//! protocol orchestrator. [`OpeningReduction`] reduces many claims into fewer;
//! blanket-implemented for homomorphic schemes via RLC.

mod claims;
mod error;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod reduction;
mod schemes;

pub use claims::{ProverClaim, VerifierClaim};
pub use error::OpeningsError;
pub use reduction::{rlc_combine, rlc_combine_scalars, OpeningReduction};

pub use schemes::{
    AdditivelyHomomorphic, CommitmentScheme, StreamingCommitment, ZkOpeningScheme,
};
