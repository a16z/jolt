//! Dory polynomial commitment scheme for the Jolt zkVM.
//!
//! Wraps `dory-pcs` behind the `jolt-openings` trait hierarchy:
//! [`CommitmentScheme`](jolt_openings::CommitmentScheme),
//! [`AdditivelyHomomorphic`](jolt_openings::AdditivelyHomomorphic),
//! [`StreamingCommitment`](jolt_openings::StreamingCommitment),
//! [`ZkOpeningScheme`](jolt_openings::ZkOpeningScheme).

pub mod scheme;
pub mod streaming;
pub mod types;

mod transcript;

pub use scheme::DoryScheme;
pub use types::{
    DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup, DoryVerifierSetup,
};
