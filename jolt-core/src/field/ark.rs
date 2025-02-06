use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use rayon::prelude::*;

use crate::utils::thread::unsafe_allocate_zero_vec;

use super::{FieldOps, JoltField};

impl FieldOps for ark_bn254::Fr {}
impl<'a, 'b> FieldOps<&'b ark_bn254::Fr, ark_bn254::Fr> for &'a ark_bn254::Fr {}
impl<'b> FieldOps<&'b ark_bn254::Fr, ark_bn254::Fr> for ark_bn254::Fr {}

static mut SMALL_VALUE_LOOKUP_TABLES: [Vec<ark_bn254::Fr>; 4] = [vec![], vec![], vec![], vec![]];

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;
    type SmallValueLookupTables = [Vec<Self>; 4];

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        // These four lookup tables correspond to the four 16-bit limbs of a u64
        let mut lookup_tables = [
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
        ];

        for i in 0..4 {
            let bitshift = 16 * i;
            let unit = <Self as ark_ff::PrimeField>::from_u64(1 << bitshift).unwrap();
            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * <Self as ark_ff::PrimeField>::from_u64(j).unwrap())
                .collect();
        }

        lookup_tables
    }

    fn initialize_lookup_tables(init: Self::SmallValueLookupTables) {
        unsafe {
            SMALL_VALUE_LOOKUP_TABLES = init;
        }
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        // TODO(moodlezoup): Using the lookup tables seems to break our tests
        #[cfg(test)]
        {
            <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
        }
        #[cfg(not(test))]
        {
            unsafe { SMALL_VALUE_LOOKUP_TABLES[0][n as usize] }
        }
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        // TODO(moodlezoup): Using the lookup tables seems to break our tests
        #[cfg(test)]
        {
            <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
        }
        #[cfg(not(test))]
        {
            unsafe { SMALL_VALUE_LOOKUP_TABLES[0][n as usize] }
        }
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        // TODO(moodlezoup): Using the lookup tables seems to break our tests
        #[cfg(test)]
        {
            <Self as ark_ff::PrimeField>::from_u64(n as u64).unwrap()
        }
        #[cfg(not(test))]
        {
            const BITMASK: u32 = (1 << 16) - 1;
            unsafe {
                SMALL_VALUE_LOOKUP_TABLES[0][(n & BITMASK) as usize]
                    + SMALL_VALUE_LOOKUP_TABLES[1][((n >> 16) & BITMASK) as usize]
            }
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        // TODO(moodlezoup): Using the lookup tables seems to break our tests
        #[cfg(test)]
        {
            <Self as ark_ff::PrimeField>::from_u64(n).unwrap()
        }
        #[cfg(not(test))]
        {
            const BITMASK: u64 = (1 << 16) - 1;
            unsafe {
                SMALL_VALUE_LOOKUP_TABLES[0][(n & BITMASK) as usize]
                    + SMALL_VALUE_LOOKUP_TABLES[1][((n >> 16) & BITMASK) as usize]
                    + SMALL_VALUE_LOOKUP_TABLES[2][((n >> 32) & BITMASK) as usize]
                    + SMALL_VALUE_LOOKUP_TABLES[3][((n >> 48) & BITMASK) as usize]
            }
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

    fn montgomery_r2() -> Option<Self> {
        Some(ark_ff::Fp::new_unchecked(Self::R2))
    }

    #[inline(always)]
    fn mul_u64_unchecked(&self, n: u64) -> Self {
        ark_ff::Fp::mul_u64(*self, n)
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
            assert_eq!(
                <Fr as JoltField>::from_u64(x),
                Fr::montgomery_r2().unwrap().mul_u64_unchecked(x)
            );
        }

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = Fr::random(&mut rng);
            assert_eq!(
                y * <Fr as JoltField>::from_u64(x),
                (y * Fr::montgomery_r2().unwrap()).mul_u64_unchecked(x)
            );
        }
    }
}
