//! Backend-agnostic cryptographic group and commitment primitives for Jolt.
//!
//! This crate provides the core group abstractions (`JoltGroup`, `PairingGroup`)
//! and a vector commitment trait used throughout the Jolt zkVM. The traits are
//! designed to be backend-agnostic: the BN254 implementation wraps arkworks
//! internally, but no arkworks types appear in the public API.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `group` | [`JoltGroup`] trait — additive group with scalar multiplication and MSM |
//! | `pairing` | [`PairingGroup`] trait — pairing-friendly group (extends `JoltGroup`) |
//! | `commitment` | [`JoltCommitment`] trait — backend-agnostic vector commitment |
//! | `arkworks` | Arkworks backend implementations (BN254) |

mod group;
pub use group::JoltGroup;

mod pairing;
pub use pairing::PairingGroup;

mod commitment;
pub use commitment::{Commitment, JoltCommitment};

mod pedersen;
pub use pedersen::{Pedersen, PedersenSetup};

#[cfg(feature = "bn254")]
pub mod arkworks;
#[cfg(feature = "bn254")]
pub use arkworks::bn254::{Bn254, Bn254G1, Bn254G2, Bn254GT};
