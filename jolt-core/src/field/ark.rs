use super::{BarrettReduce, FMAdd, FieldOps, JoltField, MontgomeryReduce, UnreducedInteger};
#[cfg(feature = "challenge-254-bit")]
use crate::field::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::field::challenge::MontU128Challenge;
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ff::biginteger::{S128, S160, S192, S256, S64};
use ark_ff::{prelude::*, BigInt, BigInteger, PrimeField, UniformRand};
use ark_std::ops::Add;
use num_traits::Zero;
use rayon::prelude::*;
use std::fmt;

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
    type UnreducedMulU64 = BigInt<5>;
    type UnreducedMulU128 = BigInt<6>;
    type UnreducedMulU128Accum = BigInt<7>;
    type UnreducedProduct = BigInt<8>;
    type UnreducedProductAccum = BigInt<9>;

    type WideAccumS = WideAccumSBn254;
    type FullAccumS = FullAccumSBn254;

    type SmallValueLookupTables = [Vec<Self>; 2];

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
    fn mul_u64_unreduced(self, other: u64) -> BigInt<5> {
        self.0.mul_trunc::<1, 5>(&BigInt::new([other]))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> BigInt<6> {
        self.0
            .mul_trunc::<2, 6>(&BigInt::new([other as u64, (other >> 64) as u64]))
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> BigInt<8> {
        self.0.mul_trunc::<4, 8>(&other.0)
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> BigInt<9> {
        self.0.mul_trunc::<4, 9>(&other.0)
    }

    #[inline]
    fn unreduced_mul_u64(a: &BigInt<4>, b: u64) -> BigInt<5> {
        a.mul_u64_w_carry(b)
    }

    #[inline]
    fn unreduced_mul_to_product_accum(a: &BigInt<4>, b: &BigInt<4>) -> BigInt<9> {
        a.mul_trunc::<4, 9>(b)
    }

    #[inline]
    fn reduce_mul_u64(x: BigInt<5>) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<5, 5>(x)
    }

    #[inline]
    fn reduce_mul_u128(x: BigInt<6>) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<6, 5>(x)
    }

    #[inline]
    fn reduce_mul_u128_accum(x: BigInt<7>) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<7, 5>(x)
    }

    #[inline]
    fn reduce_product(x: BigInt<8>) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<8, 5>(x)
    }

    #[inline]
    fn reduce_product_accum(x: BigInt<9>) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<9, 5>(x)
    }
}

// ---- BN254-specific signed accumulators ----
//
// These use BigInt<7>/BigInt<8> storage and mul_trunc internally for
// efficient unreduced accumulation of field × large-scalar products.

type Fr = ark_bn254::Fr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideAccumSBn254 {
    pos: BigInt<7>,
    neg: BigInt<7>,
}

impl Default for WideAccumSBn254 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for WideAccumSBn254 {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            pos: BigInt::new([0u64; 7]),
            neg: BigInt::new([0u64; 7]),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.pos == BigInt::new([0u64; 7]) && self.neg == BigInt::new([0u64; 7])
    }
}

impl Add for WideAccumSBn254 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.pos += rhs.pos;
        out.neg += rhs.neg;
        out
    }
}

impl fmt::Display for WideAccumSBn254 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WideAccumS(pos={}, neg={})", self.pos, self.neg)
    }
}

impl FMAdd<Fr, i128> for WideAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &i128) {
        let v = *other;
        if v == 0 {
            return;
        }
        let abs = v.unsigned_abs();
        if v > 0 {
            self.pos += field.mul_u128_unreduced(abs);
        } else {
            self.neg += field.mul_u128_unreduced(abs);
        }
    }
}

impl FMAdd<Fr, S128> for WideAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S128) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u128();
        let result = field.mul_u128_unreduced(limbs);
        if other.is_positive {
            self.pos += result;
        } else {
            self.neg += result;
        }
    }
}

impl FMAdd<Fr, S160> for WideAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S160) {
        if other.is_zero() {
            return;
        }
        let mag = other.magnitude_as_bigint_nplus1();
        let field_bigint = &field.0;
        if other.is_positive() {
            self.pos += field_bigint.mul_trunc::<3, 7>(&mag);
        } else {
            self.neg += field_bigint.mul_trunc::<3, 7>(&mag);
        }
    }
}

impl FMAdd<Fr, S192> for WideAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let mag = other.magnitude;
        let field_bigint = &field.0;
        if other.sign() {
            self.pos += field_bigint.mul_trunc::<3, 7>(&mag);
        } else {
            self.neg += field_bigint.mul_trunc::<3, 7>(&mag);
        }
    }
}

impl FMAdd<Fr, S64> for WideAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S64) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u64();
        let result = field.mul_u64_unreduced(limbs);
        if other.is_positive {
            self.pos += result;
        } else {
            self.neg += result;
        }
    }
}

impl BarrettReduce<Fr> for WideAccumSBn254 {
    #[inline(always)]
    fn barrett_reduce(&self) -> Fr {
        let result = if self.pos >= self.neg {
            Fr::from_barrett_reduce::<7, 5>(self.pos - self.neg)
        } else {
            -Fr::from_barrett_reduce::<7, 5>(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = Fr::from_barrett_reduce::<7, 5>(self.pos);
            let neg = Fr::from_barrett_reduce::<7, 5>(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullAccumSBn254 {
    pos: BigInt<8>,
    neg: BigInt<8>,
}

impl Default for FullAccumSBn254 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for FullAccumSBn254 {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            pos: BigInt::new([0u64; 8]),
            neg: BigInt::new([0u64; 8]),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.pos == BigInt::new([0u64; 8]) && self.neg == BigInt::new([0u64; 8])
    }
}

impl Add for FullAccumSBn254 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.pos += rhs.pos;
        out.neg += rhs.neg;
        out
    }
}

impl fmt::Display for FullAccumSBn254 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FullAccumS(pos={}, neg={})", self.pos, self.neg)
    }
}

impl FMAdd<Fr, S128> for FullAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S128) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u128();
        let term = field.mul_u128_unreduced(limbs);
        if other.is_positive {
            self.pos += term;
        } else {
            self.neg += term;
        }
    }
}

impl FMAdd<Fr, S192> for FullAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let mag = other.magnitude;
        let field_bigint = &field.0;
        if other.sign() {
            self.pos += field_bigint.mul_trunc::<3, 8>(&mag);
        } else {
            self.neg += field_bigint.mul_trunc::<3, 8>(&mag);
        }
    }
}

impl FMAdd<Fr, S256> for FullAccumSBn254 {
    #[inline(always)]
    fn fmadd(&mut self, field: &Fr, other: &S256) {
        if other.magnitude_limbs() == [0u64; 4] {
            return;
        }
        let mag = other.magnitude;
        let field_bigint = &field.0;
        if other.sign() {
            self.pos += field_bigint.mul_trunc::<4, 8>(&mag);
        } else {
            self.neg += field_bigint.mul_trunc::<4, 8>(&mag);
        }
    }
}

impl MontgomeryReduce<Fr> for FullAccumSBn254 {
    #[inline(always)]
    fn montgomery_reduce(&self) -> Fr {
        let result = if self.pos >= self.neg {
            Fr::from_montgomery_reduce::<8, 5>(self.pos - self.neg)
        } else {
            -Fr::from_montgomery_reduce::<8, 5>(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = Fr::from_montgomery_reduce::<8, 5>(self.pos);
            let neg = Fr::from_montgomery_reduce::<8, 5>(self.neg);
            debug_assert_eq!(result, pos - neg);
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
