//! Arkworks-backed field implementations.
//!
//! Provides the BN254 scalar field (`Fr`) and its low-level arithmetic
//! (Montgomery/Barrett reduction, precomputed lookup tables, sparse multiplication).

use crate::Limbs;
use ark_ff::BigInt;

pub mod bn254;
pub(crate) mod bn254_ops;
pub mod montgomery_impl;
pub mod wide_accumulator;

impl<const N: usize> From<Limbs<N>> for BigInt<N> {
    #[inline]
    fn from(limbs: Limbs<N>) -> Self {
        BigInt(limbs.0)
    }
}

impl<const N: usize> From<BigInt<N>> for Limbs<N> {
    #[inline]
    fn from(bigint: BigInt<N>) -> Self {
        Limbs(bigint.0)
    }
}
