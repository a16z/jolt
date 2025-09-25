use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
//use ark_std::{One, Zero};
use std::fmt::{Debug, Display};
use std::hash::Hash;
//use std::iter::{Product, Sum};
use std::ops::*;

/// Trivial implementation of Challenge type that just wraps the field element
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize,
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
}

impl<F: JoltField> Display for TrivialChallenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TrivialChallenge({})", self.value)
    }
}

//impl<F: JoltField> Zero for TrivialChallenge<F> {
//    fn zero() -> Self {
//        Self { value: F::zero() }
//    }
//
//    fn is_zero(&self) -> bool {
//        self.value.is_zero()
//    }
//}
//
//impl<F: JoltField> One for TrivialChallenge<F> {
//    fn one() -> Self {
//        Self { value: F::one() }
//    }
//
//    fn is_one(&self) -> bool {
//        self.value.is_one()
//    }
//}
//
impl<F: JoltField> Add for TrivialChallenge<F> {
    type Output = F;

    fn add(self, other: Self) -> F {
        self.value + other.value
    }
}

impl<F: JoltField> Sub for TrivialChallenge<F> {
    type Output = F;

    fn sub(self, other: Self) -> F {
        self.value - other.value
    }
}

impl<F: JoltField> Mul for TrivialChallenge<F> {
    type Output = F;

    fn mul(self, other: Self) -> F {
        self.value * other.value
    }
}

impl<F: JoltField> Neg for TrivialChallenge<F> {
    type Output = F;

    fn neg(self) -> F {
        -self.value
    }
}

//impl<F: JoltField> AddAssign for TrivialChallenge<F> {
//    fn add_assign(&mut self, other: Self) {
//        self.value += other.value;
//    }
//}
//
//impl<F: JoltField> SubAssign for TrivialChallenge<F> {
//    fn sub_assign(&mut self, other: Self) {
//        self.value -= other.value;
//    }
//}
//
//impl<F: JoltField> MulAssign for TrivialChallenge<F> {
//    fn mul_assign(&mut self, other: Self) {
//        self.value *= other.value;
//    }
//}
//
// Critical: Challenge * F -> F
impl<F: JoltField> Mul<F> for TrivialChallenge<F> {
    type Output = F;

    fn mul(self, rhs: F) -> F {
        self.value * rhs
    }
}

impl<F: JoltField> Mul<&F> for TrivialChallenge<F> {
    type Output = F;

    fn mul(self, rhs: &F) -> F {
        self.value * rhs
    }
}

impl<'a, F: JoltField> Mul<&'a F> for &'a TrivialChallenge<F> {
    type Output = F;

    fn mul(self, rhs: &'a F) -> F {
        self.value * rhs
    }
}

impl<F: JoltField> Mul<F> for &TrivialChallenge<F> {
    type Output = F;

    fn mul(self, rhs: F) -> F {
        self.value * rhs
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

//impl<F: JoltField> Sum for TrivialChallenge<F> {
//    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//        iter.fold(Self::zero(), |acc, x| acc + x)
//    }
//}
//
//impl<'a, F: JoltField> Sum<&'a TrivialChallenge<F>> for TrivialChallenge<F> {
//    fn sum<I: Iterator<Item = &'a TrivialChallenge<F>>>(iter: I) -> Self {
//        iter.fold(Self::zero(), |acc, x| acc + *x)
//    }
//}
//
//impl<F: JoltField> Product for TrivialChallenge<F> {
//    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
//        iter.fold(Self::one(), |acc, x| acc * x)
//    }
//}
//
//impl<'a, F: JoltField> Product<&'a TrivialChallenge<F>> for TrivialChallenge<F> {
//    fn product<I: Iterator<Item = &'a TrivialChallenge<F>>>(iter: I) -> Self {
//        iter.fold(Self::one(), |acc, x| acc * *x)
//    }
//}
