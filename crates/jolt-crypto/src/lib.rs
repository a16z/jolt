//! Backend-agnostic cryptographic group and commitment primitives for Jolt.
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `ec` | Elliptic curve: `JoltGroup`, `PairingGroup`, `Pedersen` |
//! | `commitment` | `Commitment`, `VectorCommitment`, `HomomorphicCommitment`, `DeriveSetup` |

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

pub mod ec;
pub use ec::{JoltGroup, PairingGroup, Pedersen, PedersenSetup};

mod commitment;
pub use commitment::{
    Commitment, DeriveSetup, HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening,
    VectorOpeningError,
};

#[cfg(feature = "bn254")]
pub use ec::bn254::{Bn254, Bn254G1, Bn254G2, Bn254GT};
