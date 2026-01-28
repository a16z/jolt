//! Challenge type whose range is the same as the Field type

use crate::{Field, OptimizedMul, Challenge};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_traits::{One, Zero};
use rand::{Rng, RngCore};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::*;

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
)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct Mont254BitChallenge<F: Field> {
    value: F,
}

impl<F: Field> Mont254BitChallenge<F> {
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

impl<F: Field> From<u128> for Mont254BitChallenge<F> {
    fn from(val: u128) -> Self {
        Self::new(F::from_u128(val))
    }
}

impl<F: Field> From<F> for Mont254BitChallenge<F> {
    fn from(value: F) -> Self {
        Self::new(value)
    }
}

impl<F: Field> UniformRand for Mont254BitChallenge<F> {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F> Challenge<F> for Mont254BitChallenge<F>
where
    F: Field,
    Self: From<u128> + Into<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<F, Output = F>,
{
    fn rand<R: RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }
}

impl<F: Field> Display for Mont254BitChallenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mont254BitChallenge({})", self.value)
    }
}

impl<F: Field> Neg for Mont254BitChallenge<F> {
    type Output = F;

    fn neg(self) -> F {
        -self.value
    }
}

impl From<Mont254BitChallenge<Fr>> for Fr {
    #[inline(always)]
    fn from(challenge: Mont254BitChallenge<Fr>) -> Self {
        challenge.value()
    }
}

impl From<&Mont254BitChallenge<Fr>> for Fr {
    #[inline(always)]
    fn from(challenge: &Mont254BitChallenge<Fr>) -> Self {
        challenge.value()
    }
}

impl_field_ops_inline!(Mont254BitChallenge<Fr>, Fr, standard);

impl OptimizedMul<Fr, Fr> for Mont254BitChallenge<Fr> {
    fn mul_0_optimized(self, other: Fr) -> Self::Output {
        if other.is_zero() {
            Fr::zero()
        } else {
            self * other
        }
    }

    fn mul_1_optimized(self, other: Fr) -> Self::Output {
        if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }

    fn mul_01_optimized(self, other: Fr) -> Self::Output {
        if other.is_zero() {
            Fr::zero()
        } else if other.is_one() {
            self.into()
        } else {
            self * other
        }
    }
}