//! Unreduced field element operations
//!
//! This module provides traits and types for working with unreduced field elements,
//! which are intermediate representations used in field arithmetic before reduction.

use crate::Field;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Trait for unreduced field element arithmetic
///
/// This trait defines operations on unreduced field elements, which are
/// intermediate representations that may exceed the field modulus.
pub trait UnreducedField<F: Field>:
    Clone
    + Copy
    + Debug
    + Zero
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
{
    /// Perform truncated multiplication with configurable output size
    fn mul_trunc<const M: usize>(&self, other: &Self) -> Self;

    /// Add-assign with mixed-size operands
    fn add_assign_mixed<const M: usize>(&mut self, other: &Self);
}