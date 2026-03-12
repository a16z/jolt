//! Field abstractions for the Jolt zkVM
//!
//! This crate provides the core field trait (`Field`) and associated types
//! used throughout the Jolt zkVM ecosystem.

mod field;
pub use field::{Field, MaybeAllocative, OptimizedMul};
#[cfg(feature = "bn254")]
pub(crate) use field::{ReductionOps, UnreducedOps};

mod accumulator;
pub use accumulator::{FieldAccumulator, NaiveAccumulator};

mod accumulation;
#[cfg(feature = "bn254")]
pub(crate) use accumulation::FMAdd;

pub mod limbs;
pub use limbs::Limbs;

pub(crate) mod bigint_ext;

pub mod signed;

#[cfg(feature = "bn254")]
pub mod arkworks;
#[cfg(feature = "bn254")]
pub use arkworks::bn254::Fr;
#[cfg(feature = "bn254")]
pub use arkworks::wide_accumulator::WideAccumulator;

#[cfg(feature = "dory-pcs")]
mod dory_interop;
