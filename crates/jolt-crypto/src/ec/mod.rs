//! Elliptic curve primitives: groups, pairings, and Pedersen commitments.

mod group;
pub use group::JoltGroup;

mod pairing;
pub use pairing::PairingGroup;

#[cfg(feature = "bn254")]
mod pedersen;
#[cfg(feature = "bn254")]
pub use pedersen::{Pedersen, PedersenSetup};

#[cfg(feature = "bn254")]
pub mod bn254;
