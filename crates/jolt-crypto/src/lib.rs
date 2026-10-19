// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

#[cfg(feature = "lattice")]
use jolt_field::AlgebraError;

#[cfg(feature = "lattice")]
pub mod challenge;
pub mod ec;
#[cfg(feature = "lattice")]
pub mod lattice;
pub use ec::{JoltGroup, PairingGroup};
#[cfg(feature = "bn254")]
pub use ec::{Pedersen, PedersenSetup};

mod commitment;
pub use commitment::{
    Commitment, DeriveSetup, HomomorphicCommitment, NoCommitment, NoVectorCommitment,
    VectorCommitment, VectorCommitmentOpening, VectorOpeningError,
};

#[cfg(feature = "bn254")]
pub use ec::bn254::{Bn254, Bn254G1, Bn254G2, Bn254GT};

/// Errors from the lattice primitives: challenge sampling and public-matrix
/// derivation.
#[cfg(feature = "lattice")]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CryptoError {
    /// A per-call argument is invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// An argument has the wrong length.
    #[error("invalid size: expected {expected}, actual {actual}")]
    InvalidSize { expected: usize, actual: usize },
    /// A parameter that only setup supplies (a challenge family, a rejection
    /// policy, a field) is invalid.
    #[error("invalid setup: {0}")]
    InvalidSetup(String),
    /// An invariant this crate maintains failed after its inputs were
    /// admitted; this reports a bug, never a rejected input.
    #[error("internal error: {0}")]
    Internal(String),
    /// An algebra operation rejected its arguments.
    #[error(transparent)]
    Algebra(#[from] AlgebraError),
}
