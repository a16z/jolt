//! Native `num_traits`/`std` supertrait impls and core-algebra markers for the
//! extension field types (`FpExt2`, `FpExt4`, `FpExt8`).
//!
//! These are the Jolt-free supertrait obligations of the native
//! [`AdditiveGroup`]/[`FieldCore`] hierarchy. The non-trivial `RingCore::square`
//! / `FieldCore::inverse` impls stay co-located with each extension type.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};

use num_traits::{One, Zero};

use super::{ExtMulBackend, FpExt2, FpExt2Config, FpExt4, FpExt8};
use crate::{AdditiveGroup, FieldCore};

// --- FpExt2 -----------------------------------------------------------------

impl<F: FieldCore, C: FpExt2Config<F>> Zero for FpExt2<F, C> {
    #[inline]
    fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero() && self.coeffs[1].is_zero()
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> One for FpExt2<F, C> {
    #[inline]
    fn one() -> Self {
        Self::new(F::one(), F::zero())
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> fmt::Display for FpExt2<F, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.coeffs[0], self.coeffs[1])
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Hash for FpExt2<F, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs[0].hash(state);
        self.coeffs[1].hash(state);
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Sum for FpExt2<F, C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore, C: FpExt2Config<F>> Sum<&'a Self> for FpExt2<F, C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> Product for FpExt2<F, C> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore, C: FpExt2Config<F>> Product<&'a Self> for FpExt2<F, C> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore, C: FpExt2Config<F>> AdditiveGroup for FpExt2<F, C> {}

// --- FpExt4 -----------------------------------------------------

impl<F: FieldCore> Zero for FpExt4<F> {
    #[inline]
    fn zero() -> Self {
        Self::new([F::zero(), F::zero(), F::zero(), F::zero()])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<F: FieldCore + ExtMulBackend> One for FpExt4<F> {
    #[inline]
    fn one() -> Self {
        Self::new([F::one(), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: FieldCore> fmt::Display for FpExt4<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {}, {})",
            self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        )
    }
}

impl<F: FieldCore> Hash for FpExt4<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
}

impl<F: FieldCore> Sum for FpExt4<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore> Sum<&'a Self> for FpExt4<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore + ExtMulBackend> Product for FpExt4<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore + ExtMulBackend> Product<&'a Self> for FpExt4<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore + ExtMulBackend> AdditiveGroup for FpExt4<F> {}

// --- FpExt8 -----------------------------------------------------

impl<F: FieldCore> Zero for FpExt8<F> {
    #[inline]
    fn zero() -> Self {
        Self::new([
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl<F: FieldCore + ExtMulBackend> One for FpExt8<F> {
    #[inline]
    fn one() -> Self {
        Self::new([
            F::one(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        ])
    }
}

impl<F: FieldCore> fmt::Display for FpExt8<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {}, {}, {}, {})",
            self.coeffs[0],
            self.coeffs[1],
            self.coeffs[2],
            self.coeffs[3],
            self.coeffs[4],
            self.coeffs[5],
            self.coeffs[6],
            self.coeffs[7]
        )
    }
}

impl<F: FieldCore> Hash for FpExt8<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coeffs.hash(state);
    }
}

impl<F: FieldCore> Sum for FpExt8<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore> Sum<&'a Self> for FpExt8<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore + ExtMulBackend> Product for FpExt8<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore + ExtMulBackend> Product<&'a Self> for FpExt8<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore + ExtMulBackend> AdditiveGroup for FpExt8<F> {}
