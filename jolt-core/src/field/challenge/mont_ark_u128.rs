//! Optimized Challenge field for faster polynomial operations
//!
//! This module implements a specialized Challenge type that is a 125-bit subset of JoltField
//! with the two least significant bits zeroed out. This constraint enables ~1.6x faster
//! multiplication with ark_bn254::Fr elements, resulting in ~1.3x speedup for polynomial
//! binding operations.
//!
//! For implementation details and benchmarks, see: *TODO: LINK*

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use allocative::Allocative;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{Rng, RngCore};

use crate::{
    field::{tracked_ark::TrackedFr, JoltField},
    impl_field_ops_inline,
};
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

impl<F: JoltField> From<u128> for MontU128Challenge<F> {
    fn from(value: u128) -> Self {
        Self::new(value)
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

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl<F: JoltField> UniformRand for MontU128Challenge<F> {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Self::from(rng.gen::<u128>())
    }
}

impl Into<ark_bn254::Fr> for MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}
impl Into<ark_bn254::Fr> for &MontU128Challenge<ark_bn254::Fr> {
    #[inline(always)]
    fn into(self) -> ark_bn254::Fr {
        ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap()
    }
}

impl_field_ops_inline!(MontU128Challenge<ark_bn254::Fr>, ark_bn254::Fr, optimized);

impl Into<TrackedFr> for MontU128Challenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap())
    }
}

impl Into<TrackedFr> for &MontU128Challenge<TrackedFr> {
    #[inline(always)]
    fn into(self) -> TrackedFr {
        TrackedFr(ark_bn254::Fr::from_bigint_unchecked(BigInt::new(self.value())).unwrap())
    }
}

impl_field_ops_inline!(MontU128Challenge<TrackedFr>, TrackedFr, optimized);
