use super::{FieldOps, JoltField, UnreducedInteger};
#[cfg(feature = "challenge-254-bit")]
use crate::field::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::field::challenge::MontU128Challenge;
use crate::field::folded_accum::{
    Folded256MulU128, Folded256MulU128Accum, Folded256MulU64, Folded256Product,
    Folded256ProductAccum,
};
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ff::{prelude::*, BigInt, BigInteger, PrimeField, UniformRand};
use rayon::prelude::*;

impl FieldOps for ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for &ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for ark_bn254::Fr {}

impl<const N: usize> UnreducedInteger for BigInt<N> {}

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;
    const NUM_LIMBS: usize = 4;

    // SAFETY: Transmuting from the Montgomery R constants from arkworks,
    // which are guaranteed to be valid field elements in Montgomery form.
    const MONTGOMERY_R: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<ark_bn254::FrConfig as MontConfig<4>>::R)
    };
    const MONTGOMERY_R_SQUARE: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<ark_bn254::FrConfig as MontConfig<4>>::R2)
    };

    type UnreducedElem = BigInt<4>;
    type UnreducedMulU64 = Folded256MulU64;
    type UnreducedMulU128 = Folded256MulU128;
    type UnreducedMulU128Accum = Folded256MulU128Accum;
    type UnreducedProduct = Folded256Product;
    type UnreducedProductAccum = Folded256ProductAccum;

    type SmallValueLookupTables = [Vec<Self>; 2];

    // Default to the optimized 125-bit challenge path; `challenge-254-bit`
    // remains an explicit opt-in for the wider representation.
    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<ark_bn254::Fr>;
    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<ark_bn254::Fr>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
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
        <Self as PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        <Self as PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        <Self as PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        if n <= u16::MAX as u64 {
            <Self as JoltField>::from_u16(n as u16)
        } else if n <= u32::MAX as u64 {
            <Self as JoltField>::from_u32(n as u32)
        } else {
            <Self as PrimeField>::from_u64::<5>(n).unwrap()
        }
    }

    #[inline]
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
                -<Self as PrimeField>::from_bigint(bigint).unwrap()
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
                <Self as PrimeField>::from_bigint(bigint).unwrap()
            }
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        if val <= u16::MAX as u128 {
            <Self as JoltField>::from_u16(val as u16)
        } else if val <= u32::MAX as u128 {
            <Self as JoltField>::from_u32(val as u32)
        } else if val <= u64::MAX as u128 {
            <Self as JoltField>::from_u64(val as u64)
        } else {
            let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
            <Self as PrimeField>::from_bigint(bigint).unwrap()
        }
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        let bigint = <Self as PrimeField>::into_bigint(*self);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as JoltField>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    #[inline]
    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <Self as PrimeField>::into_bigint(*self).num_bits()
    }

    #[inline(always)]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        self.0
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

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Folded256MulU64 {
        Folded256MulU64::from_bigint(self.0.mul_trunc::<1, 5>(&BigInt::new([other])))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Folded256MulU128 {
        Folded256MulU128::from_bigint(
            self.0
                .mul_trunc::<2, 6>(&BigInt::new([other as u64, (other >> 64) as u64])),
        )
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Folded256Product {
        Folded256Product::from_mul(self.0, other.0)
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Folded256ProductAccum {
        Folded256ProductAccum::from_mul(self.0, other.0)
    }

    #[inline]
    fn unreduced_mul_u64(a: &BigInt<4>, b: u64) -> Folded256MulU64 {
        Folded256MulU64::from_bigint(a.mul_u64_w_carry::<5>(b))
    }

    #[inline]
    fn unreduced_mul_to_product_accum(a: &BigInt<4>, b: &BigInt<4>) -> Folded256ProductAccum {
        Folded256ProductAccum::from_mul(*a, *b)
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded256MulU128Accum {
        Folded256MulU128Accum::from_bigint(self.0.mul_trunc::<M, 7>(mag))
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded256Product {
        Folded256Product::from_bigint(self.0.mul_trunc::<M, 8>(mag))
    }

    #[inline]
    fn reduce_mul_u64(x: Folded256MulU64) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<5, 5>(x.normalize())
    }

    #[inline]
    fn reduce_mul_u128(x: Folded256MulU128) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<6, 5>(x.normalize())
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Folded256MulU128Accum) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<7, 5>(x.normalize())
    }

    #[inline]
    fn reduce_product(x: Folded256Product) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<8, 5>(x.normalize())
    }

    #[inline]
    fn reduce_product_accum(x: Folded256ProductAccum) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<9, 5>(x.normalize())
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
