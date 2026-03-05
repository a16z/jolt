use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Additive cryptographic group suitable for commitments.
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
{
    /// Additive identity.
    fn zero() -> Self;

    /// Returns `true` if this element is the identity.
    fn is_zero(&self) -> bool;

    /// Returns `self + self`.
    fn double(&self) -> Self;

    /// Scalar multiplication: `scalar * self`.
    fn scalar_mul<F: Field>(&self, scalar: &F) -> Self;

    /// Multi-scalar multiplication: `Σᵢ scalars[i] * bases[i]`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `bases.len() == scalars.len()`.
    fn msm<F: Field>(bases: &[Self], scalars: &[F]) -> Self;
}
