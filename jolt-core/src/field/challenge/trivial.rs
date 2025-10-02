//! This module implements a base Challenge type whose range is exactly that the same as the
//! concrete type being used as JoltField.
//!

use crate::field::{tracked_ark::TrackedFr, JoltField};
use crate::impl_field_ops_inline;
use allocative::Allocative;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
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

impl_field_ops_inline!(TrivialChallenge<ark_bn254::Fr>, ark_bn254::Fr, standard);

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

impl_field_ops_inline!(TrivialChallenge<TrackedFr>, TrackedFr, standard);
