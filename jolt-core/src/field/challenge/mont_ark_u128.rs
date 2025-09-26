use crate::field::{IntoField, JoltField};
use allocative::Allocative;
use ark_ff::{BigInt, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
/// Bespoke implementation of Challenge type that is a subset of the JoltField
/// with the property that the 2 least significant digits are 0'd out, and it needs
/// 125 bits to represent.
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    CanonicalSerialize,
    CanonicalDeserialize,
    Allocative,
)]
pub struct MontU128Challenge<F: JoltField> {
    value: [u64; 4],
    _marker: PhantomData<F>,
}

impl<F: JoltField> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MontU128Challenge([{}, {}, {}, {}]",
            self.value[0], self.value[1], self.value[2], self.value[3]
        )
    }
}

impl<F: JoltField> MontU128Challenge<F> {
    pub fn new(value: u128) -> Self {
        // MontU128 can always be represented by 125 bits.
        // This guarantees that the big integer is never greater than the
        // bn254 modulus
        let val_masked = value & (u128::MAX >> 3);
        let low = val_masked as u64;
        let high = (val_masked >> 64) as u64;
        Self {
            value: [0, 0, low, high],
            _marker: PhantomData,
        }
    }

    pub fn value(&self) -> [u64; 4] {
        self.value
    }
}
impl IntoField<ark_bn254::Fr> for MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into_F(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}
macro_rules! impl_field_ops_inline {
    ($t:ty, $f:ty) => {
        /* ----------------------
         * $t ⊗ $t
         * ---------------------- */
        impl Add<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                self.into_F() + rhs.into_F()
            }
        }
        impl<'a> Add<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                self.into_F() + rhs.into_F()
            }
        }
        impl<'a> Add<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                self.into_F() + rhs.into_F()
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                self.into_F() + rhs.into_F()
            }
        }

        impl Sub<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                self.into_F() - rhs.into_F()
            }
        }
        impl<'a> Sub<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                self.into_F() - rhs.into_F()
            }
        }
        impl<'a> Sub<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                self.into_F() - rhs.into_F()
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                self.into_F() - rhs.into_F()
            }
        }

        impl Mul<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self.into_F() * rhs.into_F()
            }
        }
        impl<'a> Mul<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                self.into_F() * rhs.into_F()
            }
        }
        impl<'a> Mul<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self.into_F() * rhs.into_F()
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                self.into_F() * rhs.into_F()
            }
        }

        /* ----------------------
         * $t ⊗ $f
         * ---------------------- */
        impl Add<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                self.into_F() + rhs
            }
        }
        impl<'a> Add<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $f) -> $f {
                self.into_F() + rhs
            }
        }
        impl<'a> Add<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                self.into_F() + rhs
            }
        }
        impl<'a, 'b> Add<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $f) -> $f {
                self.into_F() + rhs
            }
        }

        impl Sub<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                self.into_F() - rhs
            }
        }
        impl<'a> Sub<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $f) -> $f {
                self.into_F() - rhs
            }
        }
        impl<'a> Sub<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                self.into_F() - rhs
            }
        }
        impl<'a, 'b> Sub<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $f) -> $f {
                self.into_F() - rhs
            }
        }

        impl Mul<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }
        impl<'a> Mul<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }
        impl<'a> Mul<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }
        impl<'a, 'b> Mul<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $f) -> $f {
                rhs.mul_hi_bigint_u128(self.value())
            }
        }

        /* ----------------------
         * $f ⊗ $t
         * ---------------------- */
        impl Add<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                self + rhs.into_F()
            }
        }
        impl<'a> Add<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                self + rhs.into_F()
            }
        }
        impl<'a> Add<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                *self + rhs.into_F()
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                *self + rhs.into_F()
            }
        }

        impl Sub<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                self - rhs.into_F()
            }
        }
        impl<'a> Sub<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                self - rhs.into_F()
            }
        }
        impl<'a> Sub<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                *self - rhs.into_F()
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                *self - rhs.into_F()
            }
        }

        impl Mul<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
        impl<'a> Mul<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
        impl<'a> Mul<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                self.mul_hi_bigint_u128(rhs.value())
            }
        }
    };
}
impl_field_ops_inline!(MontU128Challenge<ark_bn254::Fr>, ark_bn254::Fr);

impl From<u128> for MontU128Challenge<ark_bn254::Fr> {
    fn from(value: u128) -> Self {
        Self::new(value)
    }
}
