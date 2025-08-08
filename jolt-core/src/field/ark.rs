use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use rayon::prelude::*;

use crate::utils::thread::unsafe_allocate_zero_vec;

use super::{FieldOps, JoltField};

impl FieldOps for ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for &ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for ark_bn254::Fr {}

lazy_static::lazy_static! {
    static ref SMALL_VALUE_LOOKUP_TABLES: [Vec<ark_bn254::Fr>; 2] = ark_bn254::Fr::compute_lookup_tables();
}

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;
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
            let unit = <Self as ark_ff::PrimeField>::from_u64(1 << bitshift).unwrap();
            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * <Self as ark_ff::PrimeField>::from_u64(j).unwrap())
                .collect();
        }

        lookup_tables
    }

    fn initialize_lookup_tables(_init: Self::SmallValueLookupTables) {
        // no-op
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
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
            <Self as ark_ff::PrimeField>::from_u64(n).unwrap()
        }
    }

    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            let val = (-val) as u64;
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

    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            let val = (-val) as u128;
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

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
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
        } else if self.is_one() {
            <Self as JoltField>::from_u64(n)
        } else {
            ark_ff::Fp::mul_u64(*self, n)
        }
    }

    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else if self.is_one() {
            <Self as JoltField>::from_i128(n)
        } else {
            ark_ff::Fp::mul_i128(*self, n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    #[test]
    fn implicit_montgomery_conversion() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let x = rng.next_u64();
            assert_eq!(<Fr as JoltField>::from_u64(x), Fr::one().mul_u64(x));
        }

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = Fr::random(&mut rng);
            assert_eq!(y * <Fr as JoltField>::from_u64(x), y.mul_u64(x));
        }
    }
}
