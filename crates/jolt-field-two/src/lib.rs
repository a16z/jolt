//! Field and ring abstractions for the Jolt zkVM.
//!
//! A slim algebraic ladder — [`AdditiveGroup`] → [`Ring`] → [`Field`] — with
//! orthogonal capabilities: [`CanonicalEncoding`] (the Fiat-Shamir transcript
//! surface) and [`WithAccumulator`] (deferred-reduction fused multiply-add).
//! [`JoltField`] is the blanket-implemented bundle of everything Jolt's
//! protocol stack requires of a scalar field.
//!
//! Proof/wire serialization is serde + bincode over canonical bytes (see
//! [`impl_serde_bytes!`]); transcript bytes use [`CanonicalEncoding`]'s explicit
//! little-endian encoding and never go through a serialization library.
//!
//! # Backends
//!
//! - `bn254` (default): BN254 `Fr`/`Fq` via arkworks, plus a 9-limb wide
//!   accumulator with deferred Montgomery reduction.
//! - `solinas`: 32/64/128-bit pseudo-Mersenne prime fields, extension
//!   towers, packed NEON/AVX2/AVX-512 backends, unreduced accumulators.

mod algebra;
#[cfg(feature = "bn254")]
mod bn254;
mod extension;
mod limbs;
mod ops;
mod schedules;
pub mod signed;
#[cfg(feature = "solinas")]
pub mod solinas;

pub use algebra::{
    Accumulator, AdditiveGroup, CanonicalEncoding, Field, JoltField, NaiveAccumulator,
    PseudoMersenne, Ring, WithAccumulator,
};
#[cfg(feature = "bn254")]
pub use bn254::{Fq, Fr, WideAccumulator};
pub use extension::{Ext2Config, ExtField, NegOneNr, TwoNr};
pub use limbs::Limbs;
pub use num_traits::{One, Zero};
#[cfg(feature = "solinas")]
pub use solinas::{
    balanced_digit_lut, canonical_frobenius_thetas, is_registered_prime_offset,
    pseudo_mersenne_modulus, registered_prime_offset_spec, solve_frobenius_moore,
    validate_canonical_frobenius_thetas, Ext2, Fp128, Fp32, Fp64, FpExt2, FpExt4, FpExt8,
    Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7, Prime24Offset3,
    Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59,
    Prime56Offset27, Prime64Offset59, PrimeOffsetSpec, PRIME_OFFSET_IMPLEMENTED_MAX_BITS,
    PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};

/// Backend-independent input and shape failures.
#[derive(Debug, thiserror::Error)]
pub enum FieldError {
    /// Invalid input parameter or value.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Length mismatch between an expected and provided shape.
    #[error("invalid size: expected {expected}, actual {actual}")]
    InvalidSize { expected: usize, actual: usize },
}
