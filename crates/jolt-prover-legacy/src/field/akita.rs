use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use allocative::Allocative;
use ark_ff::BigInt;
use ark_ff::UniformRand;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_akita::AkitaField;
use jolt_field::{
    CanonicalBitLength, CanonicalBytes, CanonicalU64, FixedByteSize, FromPrimitiveInt,
    MulPrimitiveInt, RandomSampling, ReducingBytes,
};
use num_traits::{One, Zero};
use rand::Rng;
use rand_core::RngCore;

use crate::field::{FieldOps, JoltField, UnreducedInteger};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JoltAkitaField(pub AkitaField);

impl JoltAkitaField {
    #[inline]
    pub fn into_akita(self) -> AkitaField {
        self.0
    }

    #[inline]
    fn canonical_bytes(self) -> [u8; <AkitaField as FixedByteSize>::NUM_BYTES] {
        let mut bytes = [0u8; <AkitaField as FixedByteSize>::NUM_BYTES];
        self.0.to_bytes_le(&mut bytes);
        bytes
    }

    #[inline]
    fn canonical_u128(self) -> u128 {
        u128::from_le_bytes(self.canonical_bytes())
    }
}

impl From<AkitaField> for JoltAkitaField {
    #[inline]
    fn from(value: AkitaField) -> Self {
        Self(value)
    }
}

impl From<JoltAkitaField> for AkitaField {
    #[inline]
    fn from(value: JoltAkitaField) -> Self {
        value.0
    }
}

impl From<u128> for JoltAkitaField {
    #[inline]
    fn from(value: u128) -> Self {
        Self(AkitaField::from_u128(value))
    }
}

impl fmt::Display for JoltAkitaField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Hash for JoltAkitaField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_bytes().hash(state);
    }
}

impl Allocative for JoltAkitaField {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl Ord for JoltAkitaField {
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonical_u128().cmp(&other.canonical_u128())
    }
}

impl PartialOrd for JoltAkitaField {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Zero for JoltAkitaField {
    #[inline]
    fn zero() -> Self {
        Self(AkitaField::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for JoltAkitaField {
    #[inline]
    fn one() -> Self {
        Self(AkitaField::one())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl Add for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Add<&JoltAkitaField> for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &JoltAkitaField) -> Self::Output {
        self + *rhs
    }
}

impl Sub for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sub<&JoltAkitaField> for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &JoltAkitaField) -> Self::Output {
        self - *rhs
    }
}

impl Mul for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Mul<&JoltAkitaField> for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &JoltAkitaField) -> Self::Output {
        self * *rhs
    }
}

#[expect(
    clippy::suspicious_arithmetic_impl,
    reason = "field division is multiplication by the inverse"
)]
#[expect(
    clippy::expect_used,
    reason = "field division by zero is a caller error matching existing field semantics"
)]
impl Div for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0.inverse().expect("division by zero"))
    }
}

impl Div<&JoltAkitaField> for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &JoltAkitaField) -> Self::Output {
        self / *rhs
    }
}

impl Neg for JoltAkitaField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl AddAssign for JoltAkitaField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl AddAssign<&JoltAkitaField> for JoltAkitaField {
    #[inline]
    fn add_assign(&mut self, rhs: &JoltAkitaField) {
        *self += *rhs;
    }
}

impl SubAssign for JoltAkitaField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl SubAssign<&JoltAkitaField> for JoltAkitaField {
    #[inline]
    fn sub_assign(&mut self, rhs: &JoltAkitaField) {
        *self -= *rhs;
    }
}

impl MulAssign for JoltAkitaField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Sum for JoltAkitaField {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, value| acc + value)
    }
}

impl<'a> Sum<&'a JoltAkitaField> for JoltAkitaField {
    fn sum<I: Iterator<Item = &'a JoltAkitaField>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl Product for JoltAkitaField {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, value| acc * value)
    }
}

impl<'a> Product<&'a JoltAkitaField> for JoltAkitaField {
    fn product<I: Iterator<Item = &'a JoltAkitaField>>(iter: I) -> Self {
        iter.copied().product()
    }
}

impl FieldOps for JoltAkitaField {}
impl FieldOps<&JoltAkitaField, JoltAkitaField> for JoltAkitaField {}
impl UnreducedInteger for JoltAkitaField {}

impl Valid for JoltAkitaField {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for JoltAkitaField {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer
            .write_all(&self.canonical_bytes())
            .map_err(SerializationError::IoError)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        <AkitaField as FixedByteSize>::NUM_BYTES
    }
}

impl CanonicalDeserialize for JoltAkitaField {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut bytes = [0u8; <AkitaField as FixedByteSize>::NUM_BYTES];
        reader
            .read_exact(&mut bytes)
            .map_err(SerializationError::IoError)?;
        let value = Self(AkitaField::from_le_bytes_mod_order(&bytes));
        if validate == Validate::Yes && value.canonical_bytes() != bytes {
            return Err(SerializationError::InvalidData);
        }
        Ok(value)
    }
}

impl UniformRand for JoltAkitaField {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut bytes = [0u8; <AkitaField as FixedByteSize>::NUM_BYTES];
        rng.fill_bytes(&mut bytes);
        Self(AkitaField::from_le_bytes_mod_order(&bytes))
    }
}

impl JoltField for JoltAkitaField {
    const NUM_BYTES: usize = <AkitaField as FixedByteSize>::NUM_BYTES;
    const NUM_LIMBS: usize = Self::NUM_BYTES / 8;
    const MONTGOMERY_R: Self = Self(AkitaField::from_i64_const(1));
    const MONTGOMERY_R_SQUARE: Self = Self(AkitaField::from_i64_const(1));

    type UnreducedElem = Self;
    type UnreducedMulU64 = Self;
    type UnreducedMulU128 = Self;
    type UnreducedMulU128Accum = Self;
    type UnreducedProduct = Self;
    type UnreducedProductAccum = Self;

    type SmallValueLookupTables = Vec<u8>;
    type Challenge = Self;

    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(AkitaField::random(rng))
    }

    #[inline]
    fn from_bool(val: bool) -> Self {
        Self(AkitaField::from_bool(val))
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        Self(AkitaField::from_u8(n))
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        Self(AkitaField::from_u16(n))
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        Self(AkitaField::from_u32(n))
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        Self(AkitaField::from_u64(n))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        Self(AkitaField::from_i64(val))
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        Self(AkitaField::from_i128(val))
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        Self(AkitaField::from_u128(val))
    }

    #[inline]
    fn square(&self) -> Self {
        Self(self.0.square())
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        Self(AkitaField::from_le_bytes_mod_order(bytes))
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(Self)
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.0.to_canonical_u64_checked()
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        Self(self.0.mul_u64(n))
    }

    #[inline]
    fn mul_i64(&self, n: i64) -> Self {
        Self(self.0.mul_i64(n))
    }

    #[inline]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    #[inline(always)]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        *self
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64 {
        self.mul_u64(other)
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128 {
        self.mul_u128(other)
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct {
        self * other
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum {
        self * other
    }

    #[inline]
    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64 {
        a.mul_u64(b)
    }

    #[inline]
    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum {
        *a * *b
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedMulU128Accum {
        *self * Self::from_limb_slice(&mag.0)
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedProduct {
        *self * Self::from_limb_slice(&mag.0)
    }

    #[inline]
    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self {
        x
    }

    #[inline]
    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self {
        x
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self {
        x
    }

    #[inline]
    fn reduce_product(x: Self::UnreducedProduct) -> Self {
        x
    }

    #[inline]
    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self {
        x
    }
}

impl JoltAkitaField {
    #[inline]
    fn from_limb_slice(limbs: &[u64]) -> Self {
        let mut bytes = vec![0u8; limbs.len() * 8];
        for (chunk, limb) in bytes.chunks_mut(8).zip(limbs) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        Self(AkitaField::from_le_bytes_mod_order(&bytes))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests assert field API behavior")]

    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

    use super::*;

    #[test]
    fn arithmetic_matches_akita_field() {
        let a = JoltAkitaField::from_u64(7);
        let b = JoltAkitaField::from_u64(11);

        assert_eq!((a + b).into_akita(), AkitaField::from_u64(18));
        assert_eq!((a * b).into_akita(), AkitaField::from_u64(77));
        assert_eq!((b - a).into_akita(), AkitaField::from_u64(4));
        assert_eq!((a * a.inverse().unwrap()).into_akita(), AkitaField::one());
    }

    #[test]
    fn canonical_serialization_round_trips() {
        let value = JoltAkitaField::from_u128(123_456_789_123_456_789);
        let mut bytes = Vec::new();
        value.serialize_uncompressed(&mut bytes).unwrap();
        assert_eq!(bytes.len(), JoltAkitaField::NUM_BYTES);

        let decoded = JoltAkitaField::deserialize_uncompressed(&bytes[..]).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn canonical_deserialization_rejects_reduced_encoding() {
        let bytes = jolt_akita::AKITA_FIELD_MODULUS.to_le_bytes();

        assert!(JoltAkitaField::deserialize_uncompressed(&bytes[..]).is_err());
    }

    #[test]
    fn reduced_unreduced_hooks_match_field_arithmetic() {
        let a = JoltAkitaField::from_u64(13);
        let b = JoltAkitaField::from_u64(17);

        assert_eq!(JoltAkitaField::reduce_product(a.mul_to_product(b)), a * b);
        assert_eq!(
            JoltAkitaField::reduce_product_accum(a.mul_to_product_accum(b)),
            a * b
        );
        assert_eq!(
            JoltAkitaField::reduce_mul_u128(a.mul_u128_unreduced(19)),
            a * JoltAkitaField::from_u64(19)
        );
    }
}
