#[cfg(feature = "challenge-254-bit")]
use crate::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::challenge::MontU128Challenge;
use crate::{Field, ReductionOps, UnreducedOps, WithChallenge};
use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use rand_core::RngCore;

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
        <Self as ark_ff::PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        <Self as ark_ff::PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        <Self as ark_ff::PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        if n <= u16::MAX as u64 {
            <Self as Field>::from_u16(n as u16)
        } else if n <= u32::MAX as u64 {
            <Self as Field>::from_u32(n as u32)
        } else {
            <Self as ark_ff::PrimeField>::from_u64::<5>(n).unwrap()
        }
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u64 {
                -<Self as Field>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                -<Self as Field>::from_u32(val as u32)
            } else {
                -<Self as Field>::from_u64(val)
            }
        } else {
            let val = val as u64;
            if val <= u16::MAX as u64 {
                <Self as Field>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                <Self as Field>::from_u32(val as u32)
            } else {
                <Self as Field>::from_u64(val)
            }
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u128 {
                -<Self as Field>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                -<Self as Field>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                -<Self as Field>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                -<Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
            }
        } else {
            let val = val as u128;
            if val <= u16::MAX as u128 {
                <Self as Field>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                <Self as Field>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                <Self as Field>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
            }
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        if val <= u16::MAX as u128 {
            <Self as Field>::from_u16(val as u16)
        } else if val <= u32::MAX as u128 {
            <Self as Field>::from_u32(val as u32)
        } else if val <= u64::MAX as u128 {
            <Self as Field>::from_u64(val as u64)
        } else {
            let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
            <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
        }
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else {
            ark_ff::Fp::mul_u64::<5>(*self, n)
        }
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        ark_ff::Fp::mul_i64::<5>(*self, n)
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        ark_ff::Fp::mul_u128::<5, 6>(*self, n)
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else {
            ark_ff::Fp::mul_i128::<5, 6>(*self, n)
        }
    }
}

impl UnreducedOps for Fr {
    #[inline(always)]
    fn as_unreduced_ref(&self) -> &BigInt<4> {
        &self.0
    }

    #[inline]
    fn mul_unreduced<const L: usize>(self, other: Self) -> BigInt<L> {
        self.0.mul_trunc::<4, L>(&other.0)
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> BigInt<5> {
        self.0.mul_trunc::<1, 5>(&BigInt::new([other]))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> BigInt<6> {
        self.0
            .mul_trunc::<2, 6>(&BigInt::new([other as u64, (other >> 64) as u64]))
    }
}

impl ReductionOps for Fr {
    const MONTGOMERY_R: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R)
    };
    const MONTGOMERY_R_SQUARE: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R2)
    };

    #[inline]
    fn from_montgomery_reduce<const L: usize>(unreduced: BigInt<L>) -> Self {
        Fr::from_montgomery_reduce::<L, 5>(unreduced)
    }

    #[inline]
    fn from_barrett_reduce<const L: usize>(unreduced: BigInt<L>) -> Self {
        Fr::from_barrett_reduce::<L, 5>(unreduced)
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
