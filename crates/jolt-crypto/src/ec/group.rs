use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use jolt_field::{CanonicalBytes, CanonicalDecode, JoltField};
use serde::{Deserialize, Serialize};

/// Cryptographic group suitable for commitments.
///
/// Not necessarily an elliptic curve — the trait is intentionally general
/// enough for lattice-based or other algebraic groups. The group operation
/// uses additive notation (`Add`/`Sub`), but this is purely conventional;
/// the underlying algebra may be multiplicative.
///
/// All elements are `Copy` and thread-safe. Implementors must provide
/// scalar multiplication and multi-scalar multiplication (MSM).
///
/// # Invariant: `Default` is the identity
///
/// `Default::default()` must equal [`JoltGroup::identity()`]. Generic
/// aggregation code (e.g. `combine_commitments` in `jolt_crypto::commitment`)
/// seeds accumulators with `Default::default()`; a non-identity default would
/// silently offset every aggregate.
///
/// Requires the canonical [`CanonicalBytes`]/[`CanonicalDecode`] codec so
/// group elements travel as transcript atoms (e.g., Pedersen commitments in
/// ZK sumcheck). Decoding must accept only valid group elements.
pub trait JoltGroup:
    Clone
    + Copy
    + Debug
    + Default
    + Eq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + Serialize
    + for<'de> Deserialize<'de>
    + CanonicalBytes
    + CanonicalDecode
{
    #[must_use]
    fn identity() -> Self;

    #[must_use]
    fn is_identity(&self) -> bool;

    #[must_use]
    fn double(&self) -> Self;

    #[must_use]
    fn scalar_mul<F: JoltField>(&self, scalar: &F) -> Self;

    /// Multi-scalar multiplication: `Σᵢ scalars[i] * bases[i]`.
    ///
    /// # Panics
    ///
    /// Panics if `bases.len() != scalars.len()` (in all build profiles —
    /// backend MSMs silently truncate to the shorter slice otherwise).
    #[must_use]
    fn msm<F: JoltField>(bases: &[Self], scalars: &[F]) -> Self;
}
