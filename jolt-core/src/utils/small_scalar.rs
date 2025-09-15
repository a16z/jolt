use crate::field::{JoltField};
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// A trait for small scalars ({u/i}{8/16/32/64})
pub trait SmallScalar:
    Copy + Ord + Sync + CanonicalSerialize + CanonicalDeserialize + Allocative
{
    /// Performs a field multiplication. Uses `JoltField::mul_u64` under the hood.
    fn field_mul<F: JoltField>(&self, n: F) -> F;
    /// Converts a small scalar into a (potentially Montgomery form) `JoltField` type
    fn to_field<F: JoltField>(self) -> F;
    /// Computes `|self - other|` as a u64.
    fn abs_diff_u128(self, other: Self) -> u128;
}

impl SmallScalar for bool {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        if *self {
            n
        } else {
            F::zero()
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        if self {
            F::one()
        } else {
            F::zero()
        }
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        (self ^ other) as u128
    }
}

impl SmallScalar for u8 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u8(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        self.abs_diff(other) as u128
    }
}
impl SmallScalar for u16 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u16(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        self.abs_diff(other) as u128
    }
}
impl SmallScalar for u32 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self as u64)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u32(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        self.abs_diff(other) as u128
    }
}
impl SmallScalar for u64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u64(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        self.abs_diff(other) as u128
    }
}
impl SmallScalar for i64 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        if self.is_negative() {
            -n.mul_u64(-self as u64)
        } else {
            n.mul_u64(*self as u64)
        }
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_i64(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        // abs_diff for signed integers returns the corresponding unsigned type (u64 for i64)
        self.abs_diff(other) as u128
    }
}
impl SmallScalar for u128 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_u128(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_u128(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        self.abs_diff(other)
    }
}
impl SmallScalar for i128 {
    #[inline]
    fn field_mul<F: JoltField>(&self, n: F) -> F {
        n.mul_i128(*self)
    }
    #[inline]
    fn to_field<F: JoltField>(self) -> F {
        F::from_i128(self)
    }
    #[inline]
    fn abs_diff_u128(self, other: Self) -> u128 {
        self.abs_diff(other)
    }
}
