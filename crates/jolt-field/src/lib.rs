//! Field abstractions for the Jolt zkVM
//!
//! This crate provides the core field trait (`Field`) and associated types
//! used throughout the Jolt zkVM ecosystem.

mod field;
pub use field::{
    Challenge, Field, MaybeAllocative, OptimizedMul, ReductionOps, UnreducedOps, WithChallenge,
};

mod accumulation;
pub use accumulation::{BarrettReduce, FMAdd, MontgomeryReduce};

pub mod challenge;

pub mod bigint_ext;

pub mod signed;

pub mod arkworks;
pub use arkworks::bn254::Fr;

#[cfg(not(feature = "challenge-254-bit"))]
pub type DefaultChallenge<F> = challenge::MontU128Challenge<F>;
#[cfg(feature = "challenge-254-bit")]
pub type DefaultChallenge<F> = challenge::Mont254BitChallenge<F>;
