use crate::AdditiveGroup;
use num_traits::One;
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Mul, MulAssign},
};

/// Core ring arithmetic: additive group plus multiplication and one.
pub trait RingCore:
    AdditiveGroup
    + One
    + PartialEq
    + Eq
    + Default
    + Debug
    + Display
    + Hash
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign<Self>
    + Sum<Self>
    + for<'a> Sum<&'a Self>
    + Product<Self>
    + for<'a> Product<&'a Self>
{
    /// Returns `self * self`.
    #[inline]
    fn square(&self) -> Self {
        *self * *self
    }
}
