//! The Dory (elliptic-curve) prove path: the homomorphic pipeline —
//! streaming per-polynomial witness commitments at stage 0, the stage 1–8
//! recipes over the generated stage drivers, and the RLC-batched joint
//! opening. Compiled exactly when the `akita` feature is off; `crate::akita`
//! is the lattice sibling.

mod prover;
pub mod stages;

pub use prover::prove;
