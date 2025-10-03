//! JoltField implementations for ark_bn254::Fq field type.
//!
//! This module provides JoltField implementations for field types that are
//! specifically needed for recursive SNARK composition.

use crate::field::{FieldOps, JoltField};
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ff::{BigInt, BigInteger, Field, One, PrimeField, UniformRand, Zero};
use rayon::prelude::*;

impl FieldOps for ark_bn254::Fq {}
impl FieldOps<&ark_bn254::Fq, ark_bn254::Fq> for &ark_bn254::Fq {}
impl FieldOps<&ark_bn254::Fq, ark_bn254::Fq> for ark_bn254::Fq {}

impl JoltField for ark_bn254::Fq {
    const NUM_BYTES: usize = 32;
    const MONTGOMERY_R: Self = ark_ff::Fp::new_unchecked(
        <ark_bn254::FqConfig as ark_ff::fields::models::fp::MontConfig<4>>::R,
    );
    const MONTGOMERY_R_SQUARE: Self = ark_ff::Fp::new_unchecked(
        <ark_bn254::FqConfig as ark_ff::fields::models::fp::MontConfig<4>>::R2,
    );

    type Unreduced<const N: usize> = BigInt<N>;
    type SmallValueLookupTables = [Vec<Self>; 2];

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        // These two lookup tables correspond to the two 16-bit limbs of a u64
        let mut lookup_tables = [
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
        ];

        for i in 0..2 {
            let bitshift = 16 * i;
            let unit = <Self as ark_ff::PrimeField>::from_u64::<5>(1 << bitshift).unwrap();
            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * <Self as ark_ff::PrimeField>::from_u64::<5>(j).unwrap())
                .collect();
        }

        lookup_tables
    }

    #[inline]
    fn from_bool(val: bool) -> Self {
        if val {
            Self::one()
        } else {
            Self::zero()
        }
    }

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
        // The new `from_u64` is faster than doing 4 lookups & adding them together
        // but it's slower than doing <=2 lookups & adding them together (if n fits in u16 or u32)
        if n <= u16::MAX as u64 {
            <Self as JoltField>::from_u16(n as u16)
        } else if n <= u32::MAX as u64 {
            <Self as JoltField>::from_u32(n as u32)
        } else {
            <Self as ark_ff::PrimeField>::from_u64::<5>(n).unwrap()
        }
    }

    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u64 {
                -<Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                -<Self as JoltField>::from_u32(val as u32)
            } else {
                -<Self as JoltField>::from_u64(val)
            }
        } else {
            let val = val as u64;
            if val <= u16::MAX as u64 {
                <Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                <Self as JoltField>::from_u32(val as u32)
            } else {
                <Self as JoltField>::from_u64(val)
            }
        }
    }

    #[inline]
    fn square(&self) -> Self {
        <Self as Field>::square(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        ark_bn254::Fq::from_le_bytes_mod_order(bytes)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as Field>::inverse(self)
    }

    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u128 {
                -<Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                -<Self as JoltField>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                -<Self as JoltField>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                -<Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
            }
        } else {
            let val = val as u128;
            if val <= u16::MAX as u128 {
                <Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                <Self as JoltField>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                <Self as JoltField>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
            }
        }
    }

    fn from_u128(val: u128) -> Self {
        if val <= u16::MAX as u128 {
            <Self as JoltField>::from_u16(val as u16)
        } else if val <= u32::MAX as u128 {
            <Self as JoltField>::from_u32(val as u32)
        } else if val <= u64::MAX as u128 {
            <Self as JoltField>::from_u64(val as u64)
        } else {
            let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
            <Self as ark_ff::PrimeField>::from_bigint(bigint).unwrap()
        }
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = self.into_bigint();
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as JoltField>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    fn num_bits(&self) -> u32 {
        self.into_bigint().num_bits()
    }

    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else if self == &Self::one() {
            <Self as JoltField>::from_u64(n)
        } else {
            ark_ff::Fp::mul_u64::<5>(*self, n)
        }
    }

    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else if self == &Self::one() {
            <Self as JoltField>::from_i128(n)
        } else {
            ark_ff::Fp::mul_i128::<5, 6>(*self, n)
        }
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        ark_ff::Fp::mul_u128::<5, 6>(*self, n)
    }

    fn as_unreduced_ref(&self) -> &Self::Unreduced<4> {
        &self.0
    }

    fn mul_unreduced<const N: usize>(self, other: Self) -> Self::Unreduced<N> {
        // Perform the multiplication without reduction
        let a_limbs = self.0.as_ref();
        let b_limbs = other.0.as_ref();

        // For BN254 Fq, we have 4 limbs, so full product is 8 limbs
        let mut result = BigInt::<N>::zero();

        // Standard long multiplication
        for i in 0..4 {
            for j in 0..4 {
                if i + j < N {
                    let prod = a_limbs[i] as u128 * b_limbs[j] as u128;
                    let lo = prod as u64;
                    let hi = (prod >> 64) as u64;
                    let mut carry = 0u64;

                    // Add lo to result[i+j]
                    let (sum, c) = result.0[i + j].overflowing_add(lo);
                    result.0[i + j] = sum;
                    carry = c as u64;

                    // Propagate carry and add hi
                    if i + j + 1 < N {
                        let (sum, c) = result.0[i + j + 1].overflowing_add(hi + carry);
                        result.0[i + j + 1] = sum;
                        carry = c as u64;

                        // Continue propagating carry if needed
                        let mut k = i + j + 2;
                        while k < N && carry > 0 {
                            let (sum, c) = result.0[k].overflowing_add(carry);
                            result.0[k] = sum;
                            carry = c as u64;
                            k += 1;
                        }
                    }
                }
            }
        }

        result
    }

    fn mul_u64_unreduced(self, other: u64) -> Self::Unreduced<5> {
        // Use BigInteger trait's mul_u64_w_carry
        <BigInt<4> as BigInteger>::mul_u64_w_carry::<5>(&self.0, other)
    }

    fn mul_u128_unreduced(self, other: u128) -> Self::Unreduced<6> {
        // Split u128 into two u64s and do the multiplication
        let lo = other as u64;
        let hi = (other >> 64) as u64;

        // Multiply by low part
        let mut result = self.mul_u64_unreduced(lo);

        // Multiply by high part and shift left by 64 bits (add to limb 1)
        if hi != 0 {
            let hi_prod = self.mul_u64_unreduced(hi);
            // Add hi_prod shifted by one limb
            let mut carry = 0u64;
            for i in 0..5 {
                if i + 1 < 6 {
                    let (sum, c) = result.0[i + 1].overflowing_add(hi_prod.0[i]);
                    let (sum2, c2) = sum.overflowing_add(carry);
                    result.0[i + 1] = sum2;
                    carry = (c as u64) + (c2 as u64);
                }
            }
        }

        // Zero-extend to 6 limbs
        let mut result6 = BigInt::<6>::zero();
        for i in 0..5 {
            result6.0[i] = result.0[i];
        }
        result6
    }

    fn from_montgomery_reduce<const N: usize>(unreduced: Self::Unreduced<N>) -> Self {
        ark_ff::Fp::from_montgomery_reduce::<N, 5>(unreduced)
    }

    fn from_barrett_reduce<const N: usize>(unreduced: Self::Unreduced<N>) -> Self {
        ark_ff::Fp::from_barrett_reduce::<N, 5>(unreduced)
    }
}
