//! Field abstractions for the Jolt zkVM
//!
//! This crate provides the core field trait (`Field`) and associated types
//! used throughout the Jolt zkVM ecosystem.

// Core field traits
mod field;
pub use field::{
    Field, UnreducedOps, ReductionOps, Challenge, WithChallenge,
    MaybeAllocative, OptimizedMul,
};

// Unreduced arithmetic
mod unreduced;
pub use unreduced::UnreducedField;

// Accumulation patterns
mod accumulation;
pub use accumulation::{FMAdd, BarrettReduce, MontgomeryReduce};

// Challenge types
pub mod challenge;

// Arkworks backend implementations
pub mod arkworks;

// Re-export commonly used types
#[cfg(not(feature = "challenge-254-bit"))]
pub type DefaultChallenge<F> = challenge::MontU128Challenge<F>;
#[cfg(feature = "challenge-254-bit")]
pub type DefaultChallenge<F> = challenge::Mont254BitChallenge<F>;

