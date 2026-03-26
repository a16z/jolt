#![cfg_attr(not(feature = "host"), no_std)]

pub use jolt_platform::{spoil_proof, UnwrapOrSpoilProof};

#[cfg(feature = "elliptic-curve")]
pub mod ec;

#[cfg(feature = "host")]
pub mod host;
