//! Field abstractions for the Jolt zkVM
//!
//! This crate provides the core field trait (`Field`) and associated types
//! used throughout the Jolt zkVM ecosystem.

mod field;
pub use field::{Challenge, Field, MaybeAllocative, OptimizedMul, WithChallenge};
#[cfg(feature = "bn254")]
pub(crate) use field::{ReductionOps, UnreducedOps};

mod accumulation;
#[cfg(feature = "bn254")]
pub(crate) use accumulation::FMAdd;

pub mod limbs;
pub use limbs::Limbs;

#[cfg(feature = "bn254")]
pub mod challenge;

pub(crate) mod bigint_ext;

pub mod signed;

#[cfg(feature = "bn254")]
pub mod arkworks;
#[cfg(feature = "bn254")]
pub use arkworks::bn254::Fr;

#[cfg(all(feature = "bn254", not(feature = "challenge-254-bit")))]
pub type DefaultChallenge<F> = challenge::MontU128Challenge<F>;
#[cfg(all(feature = "bn254", feature = "challenge-254-bit"))]
pub type DefaultChallenge<F> = challenge::Mont254BitChallenge<F>;
