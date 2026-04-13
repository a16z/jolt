//! Convenience trait bounds for challenge-field arithmetic in prefix/suffix evaluation.
//!
//! During the sumcheck protocol, prefix MLEs are evaluated using challenge values
//! drawn from the Fiat-Shamir transcript. These traits capture the arithmetic bounds
//! needed for prefix/suffix MLE computation.
//!
//! Since challenges are now just field elements (`C = F`), these traits are trivially
//! satisfied by any `F: Field`. They remain as named bounds for readability at use sites.

use jolt_field::Field;
use std::ops::{Add, Mul, Sub};

/// A challenge value that can do arithmetic with field elements and other challenges.
///
/// The key property is that all arithmetic produces field elements `F`, even
/// operations between two challenges (`C * C -> F`).
pub trait ChallengeOps<F: Field>:
    Copy
    + Send
    + Sync
    + Into<F>
    + Add<F, Output = F>
    + for<'a> Add<&'a F, Output = F>
    + Sub<F, Output = F>
    + for<'a> Sub<&'a F, Output = F>
    + Mul<F, Output = F>
    + for<'a> Mul<&'a F, Output = F>
    + Add<Self, Output = F>
    + Sub<Self, Output = F>
    + Mul<Self, Output = F>
{
}

impl<F: Field, C> ChallengeOps<F> for C where
    C: Copy
        + Send
        + Sync
        + Into<F>
        + Add<F, Output = F>
        + for<'a> Add<&'a F, Output = F>
        + Sub<F, Output = F>
        + for<'a> Sub<&'a F, Output = F>
        + Mul<F, Output = F>
        + for<'a> Mul<&'a F, Output = F>
        + Add<C, Output = F>
        + Sub<C, Output = F>
        + Mul<C, Output = F>
{
}

/// A field element that accepts arithmetic with a challenge type `C`.
///
/// Enables expressions like `F::from_u64(n) * challenge` where the field element
/// is on the left-hand side.
pub trait FieldOps<C>:
    Add<C, Output = Self>
    + for<'a> Add<&'a C, Output = Self>
    + Sub<C, Output = Self>
    + for<'a> Sub<&'a C, Output = Self>
    + Mul<C, Output = Self>
    + for<'a> Mul<&'a C, Output = Self>
{
}

impl<F, C> FieldOps<C> for F where
    F: Add<C, Output = F>
        + for<'a> Add<&'a C, Output = F>
        + Sub<C, Output = F>
        + for<'a> Sub<&'a C, Output = F>
        + Mul<C, Output = F>
        + for<'a> Mul<&'a C, Output = F>
{
}
