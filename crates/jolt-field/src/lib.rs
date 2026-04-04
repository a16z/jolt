//! Field abstractions for the Jolt zkVM.
//!
//! Backend-agnostic interface over prime-order scalar fields, currently
//! implemented for BN254 Fr. Leaf crate with no internal Jolt dependencies.
//!
//! # Core traits
//!
//! - [`Field`] — prime field element (`Copy`, thread-safe, serializable)
//! - [`FieldAccumulator`] — deferred-reduction fused multiply-add
//! - [`OptimizedMul`] — fast-path short-circuits for zero/one
//! - [`MontgomeryConstants`] — Montgomery form constants for GPU backends
//!
//! # BN254 types (feature `bn254`)
//!
//! - [`Fr`] — BN254 scalar field element
//! - [`WideAccumulator`] — 9-limb deferred Montgomery reduction
//!
//! # Multi-precision arithmetic
//!
//! - [`Limbs<N>`] — fixed-width limb array for unreduced arithmetic
//! - [`signed`] module — `S64`, `S128`, `S192`, `S256` and half-limb variants

mod field;
pub use field::{Field, MaybeAllocative, OptimizedMul};
mod accumulator;
pub use accumulator::{FieldAccumulator, NaiveAccumulator};
mod montgomery_constants;
pub use montgomery_constants::MontgomeryConstants;

pub mod limbs;
pub use limbs::Limbs;

pub mod signed;

#[cfg(feature = "bn254")]
pub mod arkworks;
#[cfg(feature = "bn254")]
pub use arkworks::bn254::Fr;
#[cfg(feature = "bn254")]
pub use arkworks::wide_accumulator::WideAccumulator;
