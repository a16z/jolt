//! Field and ring abstractions for the Jolt zkVM.
//!
//! This crate exposes a slim algebraic hierarchy under Jolt's compatibility
//! [`Field`] bundle:
//!
//! ```text
//! AdditiveGroup -> RingCore -> FieldCore
//! ```
//!
//! [`CanonicalRepr`] (the Fiat-Shamir transcript surface), primitive-integer
//! embedding, and accumulator support are separate capabilities so non-BN254
//! fields and rings opt into only the surface they actually provide. Proof
//! and wire serialization use serde + bincode, never the canonical transcript
//! encoding.
//!
//! # Core traits
//!
//! - [`Field`] — Jolt compatibility umbrella
//! - [`Accumulator`] — deferred-reduction fused multiply-add
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
#[cfg(feature = "akita")]
mod akita;
mod algebra;
mod canonical;
mod field;
mod field_error;
mod montgomery_constants;
#[cfg(feature = "solinas")]
mod native_algebra;

pub use accumulator::{Accumulator, NaiveAccumulator, WithAccumulator};
pub use algebra::{AdditiveGroup, FieldCore, FromPrimitiveInt, RingCore};
pub use canonical::CanonicalRepr;
pub use field::Field;
pub use field_error::FieldError;
pub use montgomery_constants::MontgomeryConstants;
pub use num_traits::{One, Zero};

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
    ExtField, MulBaseUnreduced,
};
#[cfg(feature = "solinas")]
pub use ext::{Ext2, ExtMulBackend, FpExt2, FpExt2Config, FpExt4, FpExt8, NegOneNr, TwoNr};
#[cfg(feature = "solinas")]
pub use prime::{
    balanced_digit_lut, is_registered_prime_offset, pseudo_mersenne_modulus,
    registered_prime_offset_spec, CanonicalField, Fp128, Fp32, Fp64, HalvingField,
    Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7, Prime24Offset3,
    Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59,
    Prime56Offset27, Prime64Offset59, PrimeOffsetSpec, PseudoMersenneField,
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
