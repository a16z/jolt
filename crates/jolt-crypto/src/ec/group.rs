use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use jolt_field::Field;
use jolt_transcript::AppendToTranscript;
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
/// Requires [`AppendToTranscript`] so group elements can be absorbed into
/// Fiat-Shamir transcripts (e.g., Pedersen commitments in ZK sumcheck).
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
    + AppendToTranscript
{
    /// Scalar field used for scalar multiplication in this group.
    type ScalarField: Field;

    /// Group identity element.
    #[must_use]
    fn identity() -> Self;

    /// Returns `true` if this element is the identity.
    #[must_use]
    fn is_identity(&self) -> bool;

    /// Returns `self + self`.
    #[must_use]
    fn double(&self) -> Self;

    /// Scalar multiplication: `scalar * self`.
    #[must_use]
    fn scalar_mul(&self, scalar: &Self::ScalarField) -> Self;

    /// Multi-scalar multiplication: `Σᵢ scalars[i] * bases[i]`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `bases.len() == scalars.len()`.
    #[must_use]
    fn msm(bases: &[Self], scalars: &[Self::ScalarField]) -> Self;
}
