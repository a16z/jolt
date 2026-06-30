//! Field and ring abstractions for the Jolt zkVM.
//!
//! This crate exposes a slim algebraic hierarchy under Jolt's compatibility
//! [`Field`] bundle:
//!
//! ```text
//! AdditiveGroup -> RingCore -> FieldCore
//!                         \-> Invertible
//! ```
//!
//! Serialization, sampling, transcript challenges, primitive-integer embedding,
//! and accumulator support are separate capabilities so non-BN254 fields can
//! opt into only the surface they actually provide.
//!
//! # Core traits
//!
//! - [`Field`] — Jolt compatibility umbrella
//! - [`RingAccumulator`] — deferred-reduction fused multiply-add
//! - [`OptimizedMul`] — fast-path short-circuits for zero/one
//! - [`MontgomeryConstants`] — Montgomery form constants for GPU backends
//!
//! # BN254 types (feature `bn254`)
//!
//! - [`Fr`] — BN254 scalar field element
//! - [`Fq`] — BN254 base field element
//! - [`WideAccumulator`] — 9-limb deferred Montgomery reduction
//!
//! # Multi-precision arithmetic
//!
//! - [`Limbs<N>`] — fixed-width limb array for unreduced arithmetic
//! - [`signed`] module — `S64`, `S128`, `S192`, `S256` and half-limb variants

mod accumulator;
mod additive_group;
#[cfg(feature = "akita")]
mod akita;
mod canonical_bit_length;
mod canonical_bytes;
mod canonical_u64;
mod field;
mod field_core;
mod fixed_byte_size;
mod fixed_bytes;
mod from_primitive_int;
mod invertible;
mod montgomery_constants;
mod mul_pow_2;
mod mul_primitive_int;
mod random_sampling;
mod reducing_bytes;
mod ring_core;
mod signed_product_accumulator;
mod small_scalar_accumulator;
mod transcript_challenge;
mod with_accumulator;

pub use accumulator::{AdditiveAccumulator, NaiveAccumulator, RingAccumulator};
pub use additive_group::AdditiveGroup;
pub use canonical_bit_length::CanonicalBitLength;
pub use canonical_bytes::CanonicalBytes;
pub use canonical_u64::CanonicalU64;
pub use field::{Field, MaybeAllocative, OptimizedMul};
pub use field_core::FieldCore;
pub use fixed_byte_size::FixedByteSize;
pub use fixed_bytes::FixedBytes;
pub use from_primitive_int::FromPrimitiveInt;
pub use invertible::Invertible;
pub use montgomery_constants::MontgomeryConstants;
pub use mul_pow_2::MulPow2;
pub use mul_primitive_int::MulPrimitiveInt;
pub use random_sampling::RandomSampling;
pub use reducing_bytes::ReducingBytes;
pub use ring_core::RingCore;
pub use signed_product_accumulator::{
    NaiveSignedProductAccumulator, SignedProductAccumulator, WithSignedProductAccumulator,
};
pub use small_scalar_accumulator::{
    NaiveSignedScalarAccumulator, SignedScalarAccumulator, WithSmallScalarAccumulator,
};
pub use transcript_challenge::TranscriptChallenge;
pub use with_accumulator::WithAccumulator;

pub mod limbs;
pub use limbs::Limbs;

pub mod signed;

#[cfg(feature = "bn254")]
pub mod arkworks;
#[cfg(feature = "bn254")]
pub use arkworks::bn254::Fr;
#[cfg(feature = "bn254")]
pub use arkworks::bn254_fq::Fq;
#[cfg(feature = "bn254")]
pub use arkworks::signed_product_accumulator::FrSignedProductAccumulator;
#[cfg(feature = "bn254")]
pub use arkworks::small_scalar_accumulator::FrSmallScalarAccumulator;
#[cfg(feature = "bn254")]
pub use arkworks::wide_accumulator::WideAccumulator;
