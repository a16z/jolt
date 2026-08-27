//! The Dory (elliptic-curve) prove path: the homomorphic pipeline —
//! streaming per-polynomial witness commitments at stage 0, the shared
//! stage 1–7 recipes ([`crate::stages`]), and the RLC-batched stage-8 joint
//! opening. Compiled exactly when the `akita` feature is off; `crate::akita`
//! is the lattice sibling.

mod preprocessing;
mod prover;
pub mod stages;

pub use preprocessing::{
    commit_trusted_advice, from_shared, from_shared_parts, preprocess_committed,
    DoryProverPreprocessing, DoryVerifierPreprocessing,
};
pub use prover::prove;
