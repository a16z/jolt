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
//! # Solinas types (feature `solinas`)
//!
//! The Solinas backend provides optimized 32-, 64-, and 128-bit prime fields,
//! extension fields, packed NEON/AVX2/AVX-512 implementations, and unreduced
//! accumulators. Akita adopts these types
//! directly in its cutover to `jolt-field`. Until that cutover lands, the
//! temporary `akita` feature retains the legacy adapter for the pre-cutover
//! `akita-field` types; it is a bootstrap edge, not the target architecture,
//! and is removed in the final migration PR.
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
mod field_error;
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
#[cfg(feature = "solinas")]
mod solinas_traits;
mod transcript_challenge;
mod with_accumulator;

pub use accumulator::{AdditiveAccumulator, NaiveAccumulator, RingAccumulator};
pub use additive_group::AdditiveGroup;
pub use canonical_bit_length::CanonicalBitLength;
pub use canonical_bytes::CanonicalBytes;
pub use canonical_u64::CanonicalU64;
pub use field::{Field, OptimizedMul};
pub use field_core::FieldCore;
pub use field_error::FieldError;
pub use fixed_byte_size::FixedByteSize;
pub use fixed_bytes::FixedBytes;
pub use from_primitive_int::FromPrimitiveInt;
pub use invertible::Invertible;
pub use montgomery_constants::MontgomeryConstants;
pub use mul_pow_2::MulPow2;
pub use mul_primitive_int::MulPrimitiveInt;
pub use num_traits::{One, Zero};
pub use random_sampling::RandomSampling;
pub use reducing_bytes::ReducingBytes;
pub use ring_core::RingCore;
#[cfg(feature = "solinas")]
pub use solinas_traits::{balanced_digit_lut, CanonicalField, HalvingField, PseudoMersenneField};
pub use transcript_challenge::TranscriptChallenge;
pub use with_accumulator::WithAccumulator;

pub mod limbs;
pub use limbs::Limbs;

pub mod signed;

#[cfg(feature = "solinas")]
mod ext;
#[cfg(feature = "solinas")]
pub mod packed;
#[cfg(feature = "solinas")]
pub mod parallel;
#[cfg(feature = "solinas")]
mod prime;
#[cfg(feature = "solinas")]
pub mod unreduced;

#[cfg(feature = "solinas")]
pub use ext::lift::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    ExtField, FrobeniusExtField, LiftBase, MulBase, MulBaseUnreduced,
};
#[cfg(feature = "solinas")]
pub use ext::{
    Ext2, FpExt2, FpExt2Config, FpExt4, FpExt4MulBackend, FpExt8, FpExt8MulBackend, NegOneNr, TwoNr,
};
#[cfg(feature = "solinas")]
pub use prime::{
    is_registered_prime_offset, pseudo_mersenne_modulus, registered_prime_offset_spec, Fp128, Fp32,
    Fp64, Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
    Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195,
    Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};

#[cfg(feature = "bn254")]
pub mod arkworks;
#[cfg(feature = "bn254")]
pub use arkworks::bn254::Fr;
#[cfg(feature = "bn254")]
pub use arkworks::bn254_fq::Fq;
#[cfg(feature = "bn254")]
pub use arkworks::wide_accumulator::WideAccumulator;
