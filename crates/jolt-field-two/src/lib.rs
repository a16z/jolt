//! Field and ring abstractions for the Jolt zkVM.
//!
//! A slim algebraic ladder — [`AdditiveGroup`] → [`Ring`] → [`Field`] — with
//! orthogonal capabilities: [`CanonicalEncoding`] (the Fiat-Shamir transcript
//! surface) and [`WithAccumulator`] (deferred-reduction fused multiply-add).
//! [`JoltField`] is the blanket-implemented bundle of everything Jolt's
//! protocol stack requires of a scalar field: `Field + CanonicalEncoding +
//! WithAccumulator + Serialize + DeserializeOwned`. Because the impl is a
//! blanket, no field type can forget to opt in.
//!
//! # Architecture: contracts and backends
//!
//! The crate is two layers with a one-way dependency:
//!
//! 1. **Contract layer** (crate root, unconditional): every trait the crate
//!    defines — the spine above plus the capability contracts
//!    [`PseudoMersenne`], [`ExtField`], [`Ext2Config`], [`MulBaseUnreduced`],
//!    [`Unreduced`], [`Fold`], [`Packed`], [`WithPacking`] — together with
//!    the stamping macros ([`impl_ring_ops!`], [`impl_group_ops!`],
//!    [`impl_serde_bytes!`]) and the backend-neutral value types
//!    ([`Limbs`], the [`signed`] bigint families). Contract files contain
//!    trait definitions only; the crate's full capability surface is
//!    readable from the root regardless of enabled features.
//! 2. **Backend layer** (feature-gated modules): implementations of the
//!    contracts. Backends never reference each other and the contract layer
//!    never references a backend, so a backend can be deleted or added
//!    without touching the contracts. A new backend implements the spine
//!    (serde and the [`JoltField`] umbrella come free via the exported
//!    macros and the blanket impl) and opts into whichever capability
//!    contracts it can serve.
//!
//! # Backends
//!
//! - `bn254` (default): BN254 `Fr`/`Fq` wrapping arkworks, plus
//!   `WideAccumulator`, a 9-limb accumulator with deferred Montgomery
//!   reduction (first-party Barrett/Montgomery kernels).
//! - `solinas`: fully first-party pseudo-Mersenne fields `p = 2^k − c` —
//!   `Fp32`/`Fp64` stamped from one fold algebra plus the hand-written
//!   two-limb `Fp128`; cyclotomic extension towers `FpExt2`/`FpExt4`/
//!   `FpExt8` with Frobenius/Moore machinery; unreduced lane accumulators
//!   and fold matrices; packed SIMD backends (NEON, AVX2, AVX-512) for
//!   32/64/128-bit lanes and packed extensions.
//!
//! # Feature flags
//!
//! - `bn254` (default) — the arkworks-backed BN254 backend.
//! - `solinas` — the pseudo-Mersenne backend (scalar, extension, unreduced,
//!   packed, and the conditional-parallelism helpers in
//!   `solinas::parallel`).
//! - `parallel` — activates rayon behind the `cfg_*!` helper macros.
//! - `allocative` — `Allocative` derives on the concrete field types for
//!   memory profiling.
//!
//! # Byte compatibility (hard invariants)
//!
//! Wire and transcript encodings are byte-identical to `jolt-field` at the
//! rebuild baseline, for both backends, so replacing that crate cannot
//! change proof bytes:
//!
//! - Proof/wire serialization is serde + bincode over canonical
//!   little-endian bytes (see [`impl_serde_bytes!`]); deserialization
//!   rejects non-canonical encodings uniformly via
//!   [`CanonicalEncoding::from_bytes_le_checked`].
//! - Fiat-Shamir transcript bytes use [`CanonicalEncoding`]'s explicit
//!   little-endian encoding ([`CanonicalEncoding::to_bytes_le`]) and never
//!   go through a serialization library.
//!
//! Both invariants are enforced by differential tests against `jolt-field`
//! as the oracle (`tests/*_differential.rs`).

mod algebra;
#[cfg(feature = "bn254")]
mod bn254;
mod extension;
mod limbs;
mod ops;
mod packed;
mod schedules;
pub mod signed;
#[cfg(feature = "solinas")]
pub mod solinas;
mod unreduced;

pub use algebra::{
    Accumulator, AdditiveGroup, CanonicalEncoding, Field, JoltField, NaiveAccumulator,
    PseudoMersenne, Ring, WithAccumulator,
};
#[cfg(feature = "bn254")]
pub use bn254::{Fq, Fr, WideAccumulator};
pub use extension::{Ext2Config, ExtField, MulBaseUnreduced, NegOneNr, TwoNr};
pub use limbs::Limbs;
pub use num_traits::{One, Zero};
pub use packed::{NoPacking, Packed, WithPacking};
#[cfg(feature = "solinas")]
pub use solinas::{
    balanced_digit_lut, canonical_frobenius_thetas, is_registered_prime_offset,
    pseudo_mersenne_modulus, registered_prime_offset_spec, solve_frobenius_moore,
    validate_canonical_frobenius_thetas, AccumPair, Ext2, FoldMatrixFp32, FoldMatrixFp64, Fp128,
    Fp128MulU64Accum, Fp128Packing, Fp128ProductAccum, Fp128x8i32, Fp32, Fp32Packing,
    Fp32ProductAccum, Fp32x2i32, Fp64, Fp64Packing, Fp64ProductAccum, Fp64x4i32, FpExt2,
    FpExt2Fp64ProductAccum, FpExt4, FpExt4Fp32ProductAccum, FpExt8, PackedFpExt2, PackedFpExt4,
    PackedFpExt8, Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7,
    Prime24Offset3, Prime30Offset35, Prime31Offset19, Prime32Offset99, Prime40Offset195,
    Prime48Offset59, Prime56Offset27, Prime64Offset59, PrimeOffsetSpec,
    PRIME_OFFSET_IMPLEMENTED_MAX_BITS, PRIME_OFFSET_MAX, PRIME_OFFSET_SPECS,
};
pub use unreduced::{Fold, Unreduced};

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
