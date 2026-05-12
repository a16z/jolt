use crate::{FromPrimitiveInt, RingCore};

/// Multiplication by primitive integer scalars.
pub trait MulPrimitiveInt: RingCore + FromPrimitiveInt {
    /// Multiplies by a `u64`.
    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    /// Multiplies by an `i64`.
    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }

    /// Multiplies by a `u128`.
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    /// Multiplies by an `i128`.
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }
}
