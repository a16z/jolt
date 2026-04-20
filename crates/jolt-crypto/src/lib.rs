//! Backend-agnostic cryptographic group and commitment primitives for Jolt.
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `ec` | Elliptic curve: `JoltGroup`, `PairingGroup`, `Pedersen` |
//! | `commitment` | `Commitment`, `VectorCommitment`, `HomomorphicCommitment`, `DeriveSetup` |

pub mod ec;
pub use ec::{JoltGroup, PairingGroup, Pedersen, PedersenSetup};

mod commitment;
pub use commitment::{Commitment, DeriveSetup, HomomorphicCommitment, VectorCommitment};

#[cfg(feature = "bn254")]
pub use ec::bn254::{Bn254, Bn254G1, Bn254G2, Bn254GT};
