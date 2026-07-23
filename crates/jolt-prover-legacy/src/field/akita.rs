//! `JoltField` implementation for the Akita fp128 field
//! (`p = 2^128 - 2^32 + 22537`, a pseudo-Mersenne prime).
//!
//! `AkitaFp128` is a newtype rather than a direct impl on the upstream type:
//! `JoltField`'s supertraits include foreign traits the upstream type does not
//! implement (`CanonicalSerialize`/`CanonicalDeserialize`, `Div`,
//! `UniformRand`, `Allocative`), and orphan rules prevent adding them here.
//!
//! fp128 is not a Montgomery field: elements are stored canonically, so
//! `MONTGOMERY_R = 1` and the "Montgomery-reduced" ladder levels use the same
//! Solinas fold as the Barrett-reduced ones.

use super::{FieldOps, JoltField};
use crate::field::folded_accum::{Folded128MulU64, Folded128Product};
use akita_field::{
    CanonicalBitLength, CanonicalField, CanonicalU64, FromPrimitiveInt, Invertible, RandomSampling,
    ReducingBytes,
};
use allocative::Allocative;
use ark_ff::{BigInt, UniformRand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::rand::Rng;
use num_traits::{One, Zero};
use std::fmt;
use std::io::{Read, Write};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub type AkitaField = akita_config::proof_optimized::fp128::Field;

/// The Akita fp128 base field, wrapped so it can implement `JoltField`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct AkitaFp128(pub AkitaField);

impl Allocative for AkitaFp128 {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl fmt::Display for AkitaFp128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[inline(always)]
fn div_inner(a: &AkitaField, b: &AkitaField) -> AkitaField {
    *a * Invertible::inverse(b).expect("division by zero")
}

impl Add for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a AkitaFp128> for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &'a AkitaFp128) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<'b> Add<&'b AkitaFp128> for &AkitaFp128 {
    type Output = AkitaFp128;
    #[inline(always)]
    fn add(self, rhs: &'b AkitaFp128) -> AkitaFp128 {
        AkitaFp128(self.0 + rhs.0)
    }
}

impl Sub for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a AkitaFp128> for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &'a AkitaFp128) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<'b> Sub<&'b AkitaFp128> for &AkitaFp128 {
    type Output = AkitaFp128;
    #[inline(always)]
    fn sub(self, rhs: &'b AkitaFp128) -> AkitaFp128 {
        AkitaFp128(self.0 - rhs.0)
    }
}

impl Mul for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl<'a> Mul<&'a AkitaFp128> for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &'a AkitaFp128) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Mul<AkitaFp128> for &AkitaFp128 {
    type Output = AkitaFp128;
    #[inline(always)]
    fn mul(self, rhs: AkitaFp128) -> AkitaFp128 {
        AkitaFp128(self.0 * rhs.0)
    }
}

impl<'b> Mul<&'b AkitaFp128> for &AkitaFp128 {
    type Output = AkitaFp128;
    #[inline(always)]
    fn mul(self, rhs: &'b AkitaFp128) -> AkitaFp128 {
        AkitaFp128(self.0 * rhs.0)
    }
}

impl Div for AkitaFp128 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self(div_inner(&self.0, &rhs.0))
    }
}

impl<'a> Div<&'a AkitaFp128> for AkitaFp128 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: &'a AkitaFp128) -> Self {
        Self(div_inner(&self.0, &rhs.0))
    }
}

impl Div<AkitaFp128> for &AkitaFp128 {
    type Output = AkitaFp128;
    #[inline]
    fn div(self, rhs: AkitaFp128) -> AkitaFp128 {
        AkitaFp128(div_inner(&self.0, &rhs.0))
    }
}

impl<'b> Div<&'b AkitaFp128> for &AkitaFp128 {
    type Output = AkitaFp128;
    #[inline]
    fn div(self, rhs: &'b AkitaFp128) -> AkitaFp128 {
        AkitaFp128(div_inner(&self.0, &rhs.0))
    }
}

impl Neg for AkitaFp128 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for AkitaFp128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for AkitaFp128 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for AkitaFp128 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Zero for AkitaFp128 {
    #[inline(always)]
    fn zero() -> Self {
        Self(AkitaField::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for AkitaFp128 {
    #[inline(always)]
    fn one() -> Self {
        Self(AkitaField::one())
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0 == AkitaField::one()
    }
}

impl Sum for AkitaFp128 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for AkitaFp128 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl Product for AkitaFp128 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

impl<'a> Product<&'a Self> for AkitaFp128 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

impl From<u128> for AkitaFp128 {
    #[inline(always)]
    fn from(value: u128) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_u128(value))
    }
}

impl UniformRand for AkitaFp128 {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        // Rejection sampling over u128, identical in distribution to the
        // native `RandomSampling` impl (which requires a sized RngCore).
        loop {
            let x: u128 = rng.gen();
            if let Some(f) = AkitaField::from_canonical_u128_checked(x) {
                return Self(f);
            }
        }
    }
}

impl Valid for AkitaFp128 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for AkitaFp128 {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0
            .to_canonical_u128()
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u128.serialized_size(compress)
    }
}

impl CanonicalDeserialize for AkitaFp128 {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u128::deserialize_with_mode(reader, compress, validate)?;
        match validate {
            Validate::Yes => AkitaField::from_canonical_u128_checked(value)
                .map(Self)
                .ok_or(SerializationError::InvalidData),
            // Mirrors the upstream Akita deserializer: unvalidated input is
            // reduced (p > 2^127, so any u128 is < 2p and one conditional
            // subtract suffices).
            Validate::No => Ok(Self(AkitaField::from_canonical_u128_reduced(value))),
        }
    }
}

impl FieldOps for AkitaFp128 {}
impl FieldOps<&AkitaFp128, AkitaFp128> for AkitaFp128 {}
impl FieldOps<&AkitaFp128, AkitaFp128> for &AkitaFp128 {}

/// Reinterpret a (canonical) unreduced element as a field element.
///
/// `from_canonical_u128` only debug-asserts canonicity; values produced by
/// `to_unreduced` are always canonical, and the subsequent widening multiply
/// plus Solinas reduce are correct for any 128-bit integer regardless.
#[inline(always)]
fn elem_to_field(a: &BigInt<2>) -> AkitaField {
    AkitaField::from_canonical_u128(a.0[0] as u128 | (a.0[1] as u128) << 64)
}

/// field × M-limb magnitude, eagerly reduced to a canonical element.
///
/// The trait allows smaller fields to eagerly reduce here (BN254 uses a
/// truncated multiply instead): the full 2+M-limb product would consume the
/// folded accumulator's slot headroom, so we reduce and embed the canonical
/// value, which is equivalent modulo p by linearity of the final reduction.
#[inline(always)]
fn mul_mag_reduced<const M: usize>(a: AkitaField, mag: &BigInt<M>) -> AkitaField {
    // M is const, so this match is resolved at monomorphization time. The
    // M ≤ 3 and M = 4 arms hit `mul_wide_limbs`' specialized paths and are
    // exact (2 + M output limbs). Callers currently pass M ∈ {3, 4}
    // (S160/S192/S256 magnitudes); the fallback covers M ≤ 10.
    match M {
        0..=3 => AkitaField::solinas_reduce(&a.mul_wide_limbs::<M, 5>(mag.0)),
        4 => AkitaField::solinas_reduce(&a.mul_wide_limbs::<M, 6>(mag.0)),
        _ => a * AkitaField::solinas_reduce(&mag.0),
    }
}

impl JoltField for AkitaFp128 {
    const NUM_BYTES: usize = 16;
    const NUM_LIMBS: usize = 2;

    // fp128 stores elements canonically (no Montgomery scaling), so R = 1.
    const MONTGOMERY_R: Self = AkitaFp128(AkitaField::from_i64_const(1));
    const MONTGOMERY_R_SQUARE: Self = AkitaFp128(AkitaField::from_i64_const(1));

    type UnreducedElem = BigInt<2>;
    type UnreducedMulU64 = Folded128MulU64;
    // The three product-tier levels share one folded type: field × u128 and
    // field × field products are both 4 limbs for fp128, and the folded slots
    // already carry the accumulation headroom that BN254 encodes as extra
    // limbs in its wider accumulator levels.
    type UnreducedMulU128 = Folded128Product;
    type UnreducedMulU128Accum = Folded128Product;
    type UnreducedProduct = Folded128Product;
    type UnreducedProductAccum = Folded128Product;

    type SmallValueLookupTables = [Vec<Self>; 2];

    // Full field elements: fp128 challenges are already 128 bits, so there is
    // no narrower representation to gain from (unlike BN254's 125-bit
    // MontU128Challenge), and challenge × field is the native field multiply.
    type Challenge = AkitaFp128;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(<AkitaField as RandomSampling>::random(rng))
    }

    #[inline]
    fn from_bool(val: bool) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_bool(val))
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_u8(n))
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_u16(n))
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_u32(n))
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_u64(n))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_i64(val))
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_i128(val))
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        Self(<AkitaField as FromPrimitiveInt>::from_u128(val))
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        <AkitaField as CanonicalU64>::to_canonical_u64_checked(&self.0)
    }

    #[inline]
    fn square(&self) -> Self {
        Self(self.0 * self.0)
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        Invertible::inverse(&self.0).map(Self)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        Self(<AkitaField as ReducingBytes>::from_le_bytes_mod_order(
            bytes,
        ))
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <AkitaField as CanonicalBitLength>::num_bits(&self.0)
    }

    #[inline(always)]
    fn to_unreduced(&self) -> BigInt<2> {
        BigInt::new(self.0.to_limbs())
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Folded128MulU64 {
        Folded128MulU64::from_bigint(BigInt::new(self.0.mul_wide_u64(other)))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Folded128Product {
        Folded128Product::from_bigint(BigInt::new(self.0.mul_wide_u128(other)))
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Folded128Product {
        Folded128Product::from_bigint(BigInt::new(self.0.mul_wide(other.0)))
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Folded128Product {
        Folded128Product::from_bigint(BigInt::new(self.0.mul_wide(other.0)))
    }

    #[inline]
    fn unreduced_mul_u64(a: &BigInt<2>, b: u64) -> Folded128MulU64 {
        Folded128MulU64::from_bigint(BigInt::new(elem_to_field(a).mul_wide_u64(b)))
    }

    #[inline]
    fn unreduced_mul_to_product_accum(a: &BigInt<2>, b: &BigInt<2>) -> Folded128Product {
        Folded128Product::from_bigint(BigInt::new(elem_to_field(a).mul_wide(elem_to_field(b))))
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded128Product {
        Folded128Product::from_bigint(BigInt::new(mul_mag_reduced(self.0, mag).to_limbs()))
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded128Product {
        Folded128Product::from_bigint(BigInt::new(mul_mag_reduced(self.0, mag).to_limbs()))
    }

    #[inline]
    fn reduce_mul_u64(x: Folded128MulU64) -> Self {
        Self(AkitaField::solinas_reduce(&x.normalize().0))
    }

    #[inline]
    fn reduce_mul_u128(x: Folded128Product) -> Self {
        Self(AkitaField::solinas_reduce(&x.normalize().0))
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Folded128Product) -> Self {
        Self(AkitaField::solinas_reduce(&x.normalize().0))
    }

    #[inline]
    fn reduce_product(x: Folded128Product) -> Self {
        Self(AkitaField::solinas_reduce(&x.normalize().0))
    }

    #[inline]
    fn reduce_product_accum(x: Folded128Product) -> Self {
        Self(AkitaField::solinas_reduce(&x.normalize().0))
    }
}

#[cfg(test)]
mod tests {
    // The reference-operand ops, `Add<UnreducedElem>`, and `Into<Self>` are
    // exercised deliberately: they are `JoltField`/`Challenge` trait surface.
    #![expect(clippy::op_ref, clippy::assign_op_pattern, clippy::useless_conversion)]
    use super::*;
    use crate::field::{BarrettReduce, FMAdd, MontgomeryReduce};
    use crate::transcripts::{Blake2bTranscript, Transcript};
    use crate::utils::accumulation::{FullAccumS, MedAccumS, SmallAccumS, WideAccumS};
    use ark_ff::biginteger::{S128, S64};
    use ark_std::test_rng;

    type F = AkitaFp128;

    /// 2^128 mod p, i.e. the pseudo-Mersenne offset c.
    fn two_pow_128() -> F {
        F::from_u128(AkitaField::C)
    }

    fn modulus() -> u128 {
        0u128.wrapping_sub(AkitaField::C)
    }

    #[test]
    fn arithmetic_matches_u128_semantics() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let x: u64 = rng.gen();
            let y: u64 = rng.gen();
            let fx = F::from_u64(x);
            let fy = F::from_u64(y);
            assert_eq!(fx + fy, F::from_u128(x as u128 + y as u128));
            assert_eq!(fx * fy, F::from_u128(x as u128 * y as u128));
            assert_eq!(fx - fy, F::from_i128(x as i128 - y as i128));
            assert_eq!(fx.square(), fx * fx);
            assert_eq!(-fx + fx, F::zero());

            let a = F::rand(&mut rng);
            let b = F::rand(&mut rng);
            let c = F::rand(&mut rng);
            assert_eq!((a + b) * c, a * c + b * c);
            assert_eq!(a * b, b * a);
            assert_eq!(a + &b, a + b);
            assert_eq!(&a * &b, a * b);
            assert_eq!(&a - &b, a - b);
        }

        assert_eq!(F::from_u128(modulus()), F::zero());
        assert_eq!(F::from_u128(modulus() - 1), -F::one());
        assert_eq!(F::from_u128(u128::MAX), F::from_u128(AkitaField::C - 1));

        let vals = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        assert_eq!(vals.iter().sum::<F>(), F::from_u64(10));
        assert_eq!(vals.iter().product::<F>(), F::from_u64(30));
        assert_eq!(vals.into_iter().sum::<F>(), F::from_u64(10));
        assert_eq!(vals.into_iter().product::<F>(), F::from_u64(30));
    }

    #[test]
    fn inverse_and_division() {
        let mut rng = test_rng();
        assert!(JoltField::inverse(&F::zero()).is_none());
        for _ in 0..64 {
            let a = F::rand(&mut rng);
            let b = F::rand(&mut rng);
            if a.is_zero() || b.is_zero() {
                continue;
            }
            let inv = JoltField::inverse(&a).unwrap();
            assert_eq!(a * inv, F::one());
            assert_eq!(a / b, a * JoltField::inverse(&b).unwrap());
            assert_eq!(&a / &b, a / b);
            assert_eq!(&a / b, a / b);
            assert_eq!((a / b) * b, a);
        }
    }

    #[test]
    fn integer_conversions() {
        let mut rng = test_rng();
        assert_eq!(F::from_bool(true), F::one());
        assert_eq!(F::from_bool(false), F::zero());
        assert_eq!(F::from_i64(-1), -F::one());
        assert_eq!(F::from_i128(-1), -F::one());
        assert_eq!(
            F::from_i128(i128::MIN),
            -F::from_u128(1u128 << 127) // |i128::MIN| = 2^127
        );
        for _ in 0..256 {
            let x: u64 = rng.gen();
            assert_eq!(F::from_u8(x as u8), F::from_u64(x as u8 as u64));
            assert_eq!(F::from_u16(x as u16), F::from_u64(x as u16 as u64));
            assert_eq!(F::from_u32(x as u32), F::from_u64(x as u32 as u64));
            assert_eq!(
                F::from_i64(-(x as i64).abs()),
                -F::from_u64((x as i64).unsigned_abs())
            );
            let n: i128 = rng.gen();
            assert_eq!(
                F::from_i128(n),
                if n >= 0 {
                    F::from_u128(n as u128)
                } else {
                    -F::from_u128(n.unsigned_abs())
                }
            );

            let fx = F::from_u64(x);
            assert_eq!(fx.to_u64(), Some(x));
            assert_eq!(JoltField::num_bits(&fx), 64 - x.leading_zeros());
        }
        assert_eq!((F::from_u128(u64::MAX as u128 + 1)).to_u64(), None);
        assert_eq!(F::from_u64(3).mul_pow_2(4), F::from_u64(48));
    }

    #[test]
    fn implicit_conversion() {
        // Mirrors ark.rs's implicit_montgomery_conversion test.
        let mut rng = test_rng();
        for _ in 0..256 {
            let x = rng.gen::<u64>();
            assert_eq!(F::from_u64(x), JoltField::mul_u64(&F::one(), x));
        }
        for _ in 0..256 {
            let x = rng.gen::<u64>();
            let y = F::random(&mut rng);
            assert_eq!(y * F::from_u64(x), JoltField::mul_u64(&y, x));
            assert_eq!(
                y * F::from_i64(-(x as i64)),
                JoltField::mul_i64(&y, -(x as i64))
            );
            let n = rng.gen::<u128>();
            assert_eq!(y * F::from_u128(n), JoltField::mul_u128(&y, n));
            let m = rng.gen::<i128>();
            assert_eq!(y * F::from_i128(m), JoltField::mul_i128(&y, m));
        }
    }

    #[test]
    fn from_bytes_reduces_mod_order() {
        let mut rng = test_rng();
        for _ in 0..64 {
            let a = F::rand(&mut rng);
            let bytes = a.0.to_canonical_u128().to_le_bytes();
            assert_eq!(F::from_bytes(&bytes), a);

            // 32-byte input: lo + hi * 2^128 (mod p)
            let lo: u128 = rng.gen();
            let hi: u128 = rng.gen();
            let mut wide = [0u8; 32];
            wide[..16].copy_from_slice(&lo.to_le_bytes());
            wide[16..].copy_from_slice(&hi.to_le_bytes());
            let expected = F::from_u128(lo) + F::from_u128(hi) * two_pow_128();
            assert_eq!(F::from_bytes(&wide), expected);
        }
        assert_eq!(
            F::from_bytes(&u128::MAX.to_le_bytes()),
            F::from_u128(u128::MAX)
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let mut rng = test_rng();
        for _ in 0..64 {
            let a = F::rand(&mut rng);
            let mut compressed = Vec::new();
            a.serialize_compressed(&mut compressed).unwrap();
            assert_eq!(compressed.len(), 16);
            assert_eq!(a.compressed_size(), 16);
            assert_eq!(F::deserialize_compressed(&compressed[..]).unwrap(), a);

            let mut uncompressed = Vec::new();
            a.serialize_uncompressed(&mut uncompressed).unwrap();
            assert_eq!(F::deserialize_uncompressed(&uncompressed[..]).unwrap(), a);
        }

        // Non-canonical bytes are rejected when validating, reduced otherwise.
        let bytes = u128::MAX.to_le_bytes();
        assert!(F::deserialize_compressed(&bytes[..]).is_err());
        assert_eq!(
            F::deserialize_compressed_unchecked(&bytes[..]).unwrap(),
            F::from_u128(u128::MAX)
        );
    }

    #[test]
    fn montgomery_constants_are_identity() {
        assert_eq!(F::MONTGOMERY_R, F::one());
        assert_eq!(F::MONTGOMERY_R_SQUARE, F::one());
        let mut rng = test_rng();
        for _ in 0..64 {
            let a = F::rand(&mut rng);
            let b = F::rand(&mut rng);
            // With R = 1, the "Montgomery-reduced" product levels must return
            // the plain field product.
            assert_eq!(F::reduce_product(a.mul_to_product(b)), a * b);
            assert_eq!(F::reduce_product_accum(a.mul_to_product_accum(b)), a * b);
        }
    }

    #[test]
    fn unreduced_ladder_matches_field_arithmetic() {
        let mut rng = test_rng();
        for _ in 0..32 {
            // Level 1: field × u64 products plus embedded field elements.
            let mut acc_u64 = Folded128MulU64::zero();
            let mut expected = F::zero();
            for _ in 0..100 {
                let a = F::rand(&mut rng);
                let n: u64 = rng.gen();
                acc_u64 += a.mul_u64_unreduced(n);
                expected += a * F::from_u64(n);

                let e = F::rand(&mut rng);
                acc_u64 = acc_u64 + e.to_unreduced();
                expected += e;

                let b = F::rand(&mut rng);
                acc_u64 += F::unreduced_mul_u64(&b.to_unreduced(), n);
                expected += b * F::from_u64(n);
            }
            assert_eq!(F::reduce_mul_u64(acc_u64), expected);

            // Levels 2-5: u128 / field / magnitude products, all feeding the
            // shared product-tier accumulator.
            let mut acc = Folded128Product::zero();
            let mut expected = F::zero();
            for _ in 0..100 {
                let a = F::rand(&mut rng);
                let b = F::rand(&mut rng);
                let n: u128 = rng.gen();
                acc += a.mul_u128_unreduced(n);
                expected += a * F::from_u128(n);

                acc += a.mul_to_product(b);
                expected += a * b;

                acc += F::unreduced_mul_to_product_accum(&a.to_unreduced(), &b.to_unreduced());
                expected += a * b;

                acc += a.to_unreduced();
                expected += a;

                acc += b.mul_u64_unreduced(n as u64);
                expected += b * F::from_u64(n as u64);
            }
            assert_eq!(F::reduce_product_accum(acc), expected);
            assert_eq!(F::reduce_product(acc), expected);
            assert_eq!(F::reduce_mul_u128(acc), expected);
            assert_eq!(F::reduce_mul_u128_accum(acc), expected);
        }
    }

    #[test]
    fn magnitude_multiplies_match_field_arithmetic() {
        let mut rng = test_rng();
        let base = two_pow_128();
        for _ in 0..256 {
            let a = F::rand(&mut rng);

            let m3 = BigInt::<3>::new([rng.gen(), rng.gen(), rng.gen()]);
            let m3_field = F::from_u128(m3.0[0] as u128 | (m3.0[1] as u128) << 64)
                + F::from_u64(m3.0[2]) * base;
            assert_eq!(
                F::reduce_mul_u128_accum(a.mul_to_accum_mag(&m3)),
                a * m3_field
            );
            assert_eq!(F::reduce_product(a.mul_to_product_mag(&m3)), a * m3_field);

            let m4 = BigInt::<4>::new([rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
            let m4_field = F::from_u128(m4.0[0] as u128 | (m4.0[1] as u128) << 64)
                + F::from_u128(m4.0[2] as u128 | (m4.0[3] as u128) << 64) * base;
            assert_eq!(F::reduce_product(a.mul_to_product_mag(&m4)), a * m4_field);
        }
        // All-ones magnitudes stress the widest intermediate values.
        let a = F::rand(&mut rng);
        let m3 = BigInt::<3>::new([u64::MAX; 3]);
        let m3_field = F::from_u128(u128::MAX) + F::from_u64(u64::MAX) * base;
        assert_eq!(
            F::reduce_mul_u128_accum(a.mul_to_accum_mag(&m3)),
            a * m3_field
        );
    }

    #[test]
    fn signed_accumulators_match_naive_sums() {
        let mut rng = test_rng();
        for _ in 0..32 {
            let mut small = SmallAccumS::<F>::zero();
            let mut med = MedAccumS::<F>::zero();
            let mut wide = WideAccumS::<F>::zero();
            let mut full = FullAccumS::<F>::zero();
            let mut expected_small = F::zero();
            let mut expected_med = F::zero();
            let mut expected_wide = F::zero();
            let mut expected_full = F::zero();
            for _ in 0..100 {
                let f = F::rand(&mut rng);
                let i: i64 = rng.gen();
                small.fmadd(&f, &i);
                expected_small += f * F::from_i64(i);

                let n: i128 = rng.gen();
                med.fmadd(&f, &n);
                expected_med += f * F::from_i128(n);

                wide.fmadd(&f, &n);
                expected_wide += f * F::from_i128(n);
                let s: i64 = rng.gen();
                wide.fmadd(&f, &S64::from(s));
                expected_wide += f * F::from_i64(s);

                full.fmadd(&f, &S128::from(n));
                expected_full += f * F::from_i128(n);
            }
            assert_eq!(small.barrett_reduce(), expected_small);
            assert_eq!(med.barrett_reduce(), expected_med);
            assert_eq!(wide.barrett_reduce(), expected_wide);
            assert_eq!(full.montgomery_reduce(), expected_full);
        }
    }

    #[test]
    fn challenge_via_transcript() {
        let mut t1 = Blake2bTranscript::new(b"akita_fp128");
        let mut t2 = Blake2bTranscript::new(b"akita_fp128");
        for t in [&mut t1, &mut t2] {
            t.append_u64(b"n", 42);
            t.append_scalar(b"s", &F::from_u64(7));
        }

        let c1: <F as JoltField>::Challenge = t1.challenge_scalar_optimized::<F>();
        let c2 = t2.challenge_scalar_optimized::<F>();
        assert_eq!(c1, c2);

        // Challenge = Self: conversion into the field is the identity and
        // challenge arithmetic is field arithmetic.
        let f: F = c1.into();
        assert_eq!(f, c1);
        let x = F::from_u64(3);
        assert_eq!(c1 * x, f * x);
        assert_eq!(x - c1, x - f);

        let s1: F = t1.challenge_scalar();
        let s2: F = t2.challenge_scalar();
        assert_eq!(s1, s2);
        assert!(s1.0.to_canonical_u128() < modulus());

        // From<u128> reduces mod p.
        assert_eq!(
            <F as JoltField>::Challenge::from(u128::MAX),
            F::from_u128(AkitaField::C - 1)
        );
    }
}
