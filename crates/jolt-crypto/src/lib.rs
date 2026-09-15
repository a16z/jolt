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

#[cfg(all(feature = "bn254", feature = "parallel"))]
macro_rules! cfg_iter {
    ($values:expr) => {
        $values.par_iter()
    };
}

#[cfg(all(feature = "bn254", not(feature = "parallel")))]
macro_rules! cfg_iter {
    ($values:expr) => {
        $values.iter()
    };
}

#[cfg(all(feature = "bn254", feature = "parallel"))]
macro_rules! cfg_iter_mut {
    ($values:expr) => {
        $values.par_iter_mut()
    };
}

#[cfg(all(feature = "bn254", not(feature = "parallel")))]
macro_rules! cfg_iter_mut {
    ($values:expr) => {
        $values.iter_mut()
    };
}

pub mod ec;
pub use ec::{JoltGroup, PairingGroup};
#[cfg(feature = "bn254")]
pub use ec::{Pedersen, PedersenSetup};

mod commitment;
pub use commitment::{
    Commitment, DeriveSetup, HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening,
    VectorOpeningError,
};

#[cfg(feature = "bn254")]
pub use ec::bn254::{Bn254, Bn254G1, Bn254G2, Bn254GT};
