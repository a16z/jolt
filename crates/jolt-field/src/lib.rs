//! Field abstractions for the Jolt zkVM
//!
//! This crate provides the core field trait (`Field`) and associated types
//! used throughout the Jolt zkVM ecosystem.

mod field;
pub use field::{Challenge, Field, MaybeAllocative, OptimizedMul, WithChallenge};
pub(crate) use field::{ReductionOps, UnreducedOps};

mod accumulation;
pub(crate) use accumulation::FMAdd;

pub mod limbs;
pub use limbs::Limbs;

pub mod challenge;

pub(crate) mod bigint_ext;

pub mod signed;

pub mod arkworks;
pub use arkworks::bn254::Fr;

#[cfg(not(feature = "challenge-254-bit"))]
pub type DefaultChallenge<F> = challenge::MontU128Challenge<F>;
#[cfg(feature = "challenge-254-bit")]
pub type DefaultChallenge<F> = challenge::Mont254BitChallenge<F>;
