use ark_ff::{prelude::*, AdditiveGroup, BigInt, PrimeField, UniformRand};
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
            let unit = <Self as JoltField>::from_u64(1 << bitshift);
            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * <Self as JoltField>::from_u64(j))
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

    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }

    fn num_bits(&self) -> u32 {
        self.into_bigint().num_bits()
    }

    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        ark_ff::Fp::mul_u64::<5>(*self, n)
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        ark_ff::Fp::mul_i64::<5>(*self, n)
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        ark_ff::Fp::mul_u128::<5, 6>(*self, n)
    }

    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        ark_ff::Fp::mul_i128::<5, 6>(*self, n)
    }

    #[inline]
    fn linear_combination_u64(pairs: &[(Self, u64)], add_terms: &[Self]) -> Self {
        let mut tmp = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&pairs[0].0 .0, pairs[0].1);
        for (a, b) in &pairs[1..] {
            let carry = tmp.add_with_carry(&ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a.0, *b));
            debug_assert!(!carry, "spurious carry in linear_combination_u64");
        }

        // Add the additional terms that don't need multiplication
        let mut result = ark_ff::Fp::from_unchecked_nplus1(tmp);
        for term in add_terms {
            result += *term;
        }
        result
    }

    #[inline]
    fn linear_combination_i64(
        pos: &[(Self, u64)],
        neg: &[(Self, u64)],
        pos_add: &[Self],
        neg_add: &[Self],
    ) -> Self {
        // unreduced linear combination of positive and negative terms
        let mut pos_lc = if !pos.is_empty() {
            let mut tmp = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&pos[0].0 .0, pos[0].1);
            for (a, b) in &pos[1..] {
                let carry =
                    tmp.add_with_carry(&ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a.0, *b));
                debug_assert!(!carry, "spurious carry in linear_combination_i64");
            }
            tmp
        } else {
            ark_ff::BigInt::<5>::zero()
        };

        let mut neg_lc = if !neg.is_empty() {
            let mut tmp = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&neg[0].0 .0, neg[0].1);
            for (a, b) in &neg[1..] {
                let carry =
                    tmp.add_with_carry(&ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a.0, *b));
                debug_assert!(!carry, "spurious carry in linear_combination_i64");
            }
            tmp
        } else {
            ark_ff::BigInt::<5>::zero()
        };

        // Compute the difference of the linear combinations
        let diff = match pos_lc.cmp(&neg_lc) {
            std::cmp::Ordering::Greater => {
                let borrow = pos_lc.sub_with_borrow(&neg_lc);
                debug_assert!(!borrow, "spurious borrow in linear_combination_i64");
                ark_ff::Fp::from_unchecked_nplus1(pos_lc)
            }
            std::cmp::Ordering::Less => {
                let borrow = neg_lc.sub_with_borrow(&pos_lc);
                debug_assert!(!borrow, "spurious borrow in linear_combination_i64");
                *ark_ff::Fp::from_unchecked_nplus1(neg_lc).neg_in_place()
            }
            std::cmp::Ordering::Equal => ark_ff::Fp::zero(),
        };

        // Add the positive and negative add terms
        let mut result = diff;
        for term in pos_add {
            result += *term;
        }
        for term in neg_add {
            result -= *term;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::One;
    use rand_chacha::rand_core::RngCore;

    #[test]
    fn implicit_montgomery_conversion() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let x = rng.next_u64();
            assert_eq!(
                <Fr as JoltField>::from_u64(x),
                JoltField::mul_u64(&Fr::one(), x)
            );
        }

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = Fr::random(&mut rng);
            assert_eq!(
                y * <Fr as JoltField>::from_u64(x),
                JoltField::mul_u64(&y, x)
            );
        }
    }
}
