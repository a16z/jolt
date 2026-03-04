//! [`Field`] implementation for the BN254 scalar field (`ark_bn254::Fr`).
//!
//! This is the only production backend. All arithmetic delegates to the
//! arkworks `Fp256` implementation with custom Montgomery and Barrett
//! reduction paths from `bn254_ops`.

use crate::bigint_ext::BigIntExt;
#[cfg(feature = "challenge-254-bit")]
use crate::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::challenge::MontU128Challenge;
use crate::{Field, ReductionOps, UnreducedOps, WithChallenge};
use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use rand_core::RngCore;

use super::bn254_ops;

type Fr = ark_bn254::Fr;
type FrConfig = ark_bn254::FrConfig;

impl Field for Fr {
    const NUM_BYTES: usize = 32;

    fn random<R: RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Fr::from_le_bytes_mod_order(bytes)
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = <Self as ark_ff::PrimeField>::into_bigint(*self);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as Field>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    fn num_bits(&self) -> u32 {
        <Self as ark_ff::PrimeField>::into_bigint(*self).num_bits()
    }

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    fn from_bool(val: bool) -> Self {
        if val {
            Self::one()
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        bn254_ops::from_u64(n as u64)
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        bn254_ops::from_u64(n as u64)
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        bn254_ops::from_u64(n as u64)
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        bn254_ops::from_u64(n)
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            -bn254_ops::from_u64(val.unsigned_abs())
        } else {
            bn254_ops::from_u64(val as u64)
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            -bn254_ops::from_u128(val.unsigned_abs())
        } else {
            bn254_ops::from_u128(val as u128)
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        bn254_ops::from_u128(val)
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        bn254_ops::mul_u64(*self, n)
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        bn254_ops::mul_i64(*self, n)
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        bn254_ops::mul_u128(*self, n)
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        bn254_ops::mul_i128(*self, n)
    }
}

impl UnreducedOps for Fr {
    #[inline(always)]
    fn as_unreduced_ref(&self) -> &BigInt<4> {
        &self.0
    }

    #[inline]
    fn mul_unreduced<const L: usize>(self, other: Self) -> BigInt<L> {
        BigIntExt::mul_trunc(&self.0, &other.0)
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> BigInt<5> {
        BigIntExt::mul_trunc(&self.0, &BigInt::new([other]))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> BigInt<6> {
        BigIntExt::mul_trunc(&self.0, &BigInt::new([other as u64, (other >> 64) as u64]))
    }
}

impl ReductionOps for Fr {
    // SAFETY: `Fr` and `BigInt<4>` have identical layout (4 x u64 limbs).
    // `MontConfig::R` is the Montgomery form of 1, which is a valid `Fr` value.
    const MONTGOMERY_R: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R)
    };
    // SAFETY: Same layout guarantee as above. `R2 = R^2 mod p` is a valid field element.
    const MONTGOMERY_R_SQUARE: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R2)
    };

    #[inline]
    fn from_montgomery_reduce<const L: usize>(unreduced: BigInt<L>) -> Self {
        bn254_ops::from_montgomery_reduce(unreduced)
    }

    #[inline]
    fn from_barrett_reduce<const L: usize>(unreduced: BigInt<L>) -> Self {
        bn254_ops::from_barrett_reduce(unreduced)
    }
}

impl WithChallenge for Fr {
    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<Fr>;

    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<Fr>;
}

impl<const N: usize, const M: usize> crate::FMAdd<BigInt<4>, BigInt<M>> for BigInt<N> {
    fn fmadd(&mut self, left: &BigInt<4>, right: &BigInt<M>) {
        for i in 0..4 {
            let mut carry = 0u64;
            for j in 0..M {
                if i + j < N {
                    let product = (left.0[i] as u128) * (right.0[j] as u128)
                        + (self.0[i + j] as u128)
                        + (carry as u128);
                    self.0[i + j] = product as u64;
                    carry = (product >> 64) as u64;
                } else {
                    break;
                }
            }
            if i + M < N {
                self.0[i + M] = self.0[i + M].wrapping_add(carry);
            }
        }
    }
}
