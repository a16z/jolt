use crate::field::{tracked_ark::TrackedFr, JoltField};
use allocative::Allocative;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::*;
/// Trivial implementation of Challenge type that just wraps the field element
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
pub struct TrivialChallenge<F: JoltField> {
    value: F,
}

impl<F: JoltField> TrivialChallenge<F> {
    pub fn new(value: F) -> Self {
        Self { value }
    }

    pub fn value(&self) -> F {
        self.value
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> UniformRand for TrivialChallenge<F> {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> Display for TrivialChallenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TrivialChallenge({})", self.value)
    }
}

impl<F: JoltField> Neg for TrivialChallenge<F> {
    type Output = F;

    fn neg(self) -> F {
        -self.value
    }
}

impl<F: JoltField> From<u128> for TrivialChallenge<F> {
    fn from(val: u128) -> Self {
        Self::new(F::from_u128(val))
    }
}

impl<F: JoltField> From<F> for TrivialChallenge<F> {
    fn from(value: F) -> Self {
        Self::new(value)
    }
}
impl Into<ark_bn254::Fr> for TrivialChallenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        self.value()
    }
}

impl Into<ark_bn254::Fr> for &TrivialChallenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        self.value()
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
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(self) + Into::<$f>::into(rhs)
            }
        }

        impl Sub<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(self) - Into::<$f>::into(rhs)
            }
        }

        impl Mul<$t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<&'a $t> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<$t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                Into::<$f>::into(self) * Into::<$f>::into(rhs)
            }
        }

        /* ----------------------
         * $t ⊗ $f
         * ---------------------- */
        impl Add<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a> Add<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a> Add<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }
        impl<'a, 'b> Add<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $f) -> $f {
                Into::<$f>::into(self) + rhs
            }
        }

        impl Sub<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a> Sub<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a> Sub<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }
        impl<'a, 'b> Sub<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $f) -> $f {
                Into::<$f>::into(self) - rhs
            }
        }

        impl Mul<$f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                Into::<$f>::into(self) * rhs
            }
        }
        impl<'a> Mul<&'a $f> for $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $f) -> $f {
                *rhs * Into::<$f>::into(self)
            }
        }

        impl<'a> Mul<$f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $f) -> $f {
                rhs * Into::<$f>::into(self)
            }
        }
        impl<'a, 'b> Mul<&'b $f> for &'a $t {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $f) -> $f {
                *rhs * Into::<$f>::into(self)
            }
        }

        /* ----------------------
         * $f ⊗ $t
         * ---------------------- */
        impl Add<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                self + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'a $t) -> $f {
                self + Into::<$f>::into(rhs)
            }
        }
        impl<'a> Add<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: $t) -> $f {
                *self + Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Add<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn add(self, rhs: &'b $t) -> $f {
                *self + Into::<$f>::into(rhs)
            }
        }

        impl Sub<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                self - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'a $t) -> $f {
                self - Into::<$f>::into(rhs)
            }
        }
        impl<'a> Sub<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: $t) -> $f {
                *self - Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Sub<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn sub(self, rhs: &'b $t) -> $f {
                *self - Into::<$f>::into(rhs)
            }
        }

        impl Mul<$t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                self * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<&'a $t> for $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'a $t) -> $f {
                self * Into::<$f>::into(rhs)
            }
        }
        impl<'a> Mul<$t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: $t) -> $f {
                *self * Into::<$f>::into(rhs)
            }
        }
        impl<'a, 'b> Mul<&'b $t> for &'a $f {
            type Output = $f;
            #[inline(always)]
            fn mul(self, rhs: &'b $t) -> $f {
                *self * Into::<$f>::into(rhs)
            }
        }
    };
}
impl_field_ops_inline!(TrivialChallenge<ark_bn254::Fr>, ark_bn254::Fr);

impl Into<TrackedFr> for TrivialChallenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(self.value().0)
    }
}

impl Into<TrackedFr> for &TrivialChallenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(self.value().0)
    }
}

impl_field_ops_inline!(TrivialChallenge<TrackedFr>, TrackedFr);
