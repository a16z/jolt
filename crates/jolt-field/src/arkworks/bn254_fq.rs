//! Newtype wrapper around `ark_bn254::Fq` that decouples the public API from arkworks.
//!
//! [`Fq`] is the BN254 base field. In the BN254/Grumpkin cycle it is also the
//! scalar field of Grumpkin.

use crate::{
    AdditiveGroup, CanonicalBitLength, CanonicalBytes, CanonicalU64, Field, FieldCore,
    FixedByteSize, FixedBytes, FromPrimitiveInt, Invertible, Limbs, MulPrimitiveInt,
    NaiveAccumulator, NaiveSignedProductAccumulator, NaiveSignedScalarAccumulator, RandomSampling,
    ReducingBytes, RingCore, TranscriptChallenge, WithAccumulator, WithSignedProductAccumulator,
    WithSmallScalarAccumulator,
};
use ark_ff::{prelude::*, PrimeField, UniformRand};
use rand_core::RngCore;

type InnerFq = ark_bn254::Fq;

/// BN254 base field element.
///
/// A `#[repr(transparent)]` newtype over `ark_bn254::Fq`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Fq(pub(crate) InnerFq);

impl From<bool> for Fq {
    #[inline(always)]
    fn from(v: bool) -> Self {
        <Self as FromPrimitiveInt>::from_bool(v)
    }
}

impl From<u8> for Fq {
    #[inline(always)]
    fn from(v: u8) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v as u64)
    }
}

impl From<u16> for Fq {
    #[inline(always)]
    fn from(v: u16) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v as u64)
    }
}

impl From<u32> for Fq {
    #[inline(always)]
    fn from(v: u32) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v as u64)
    }
}

impl From<u64> for Fq {
    #[inline(always)]
    fn from(v: u64) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v)
    }
}

impl From<i64> for Fq {
    #[inline(always)]
    fn from(v: i64) -> Self {
        <Self as FromPrimitiveInt>::from_i64(v)
    }
}

impl From<i128> for Fq {
    #[inline(always)]
    fn from(v: i128) -> Self {
        <Self as FromPrimitiveInt>::from_i128(v)
    }
}

impl From<u128> for Fq {
    #[inline(always)]
    fn from(v: u128) -> Self {
        <Self as FromPrimitiveInt>::from_u128(v)
    }
}

impl std::fmt::Debug for Fq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for Fq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

macro_rules! delegate_binop {
    ($Trait:ident, $method:ident) => {
        impl std::ops::$Trait for Fq {
            type Output = Fq;
            #[inline(always)]
            fn $method(self, rhs: Fq) -> Fq {
                Fq(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl std::ops::$Trait<&Fq> for Fq {
            type Output = Fq;
            #[inline(always)]
            fn $method(self, rhs: &Fq) -> Fq {
                Fq(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl std::ops::$Trait<Fq> for &Fq {
            type Output = Fq;
            #[inline(always)]
            fn $method(self, rhs: Fq) -> Fq {
                Fq(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl<'a, 'b> std::ops::$Trait<&'b Fq> for &'a Fq {
            type Output = Fq;
            #[inline(always)]
            fn $method(self, rhs: &'b Fq) -> Fq {
                Fq(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }
    };
}

delegate_binop!(Add, add);
delegate_binop!(Sub, sub);
delegate_binop!(Mul, mul);
delegate_binop!(Div, div);

impl std::ops::Neg for Fq {
    type Output = Fq;

    #[inline(always)]
    fn neg(self) -> Fq {
        Fq(self.0.neg())
    }
}

impl std::ops::AddAssign for Fq {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Fq) {
        self.0.add_assign(rhs.0);
    }
}

impl std::ops::SubAssign for Fq {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Fq) {
        self.0.sub_assign(rhs.0);
    }
}

impl std::ops::MulAssign for Fq {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fq) {
        self.0.mul_assign(rhs.0);
    }
}

impl std::iter::Sum for Fq {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fq(iter.map(|f| f.0).sum())
    }
}

impl<'a> std::iter::Sum<&'a Fq> for Fq {
    fn sum<I: Iterator<Item = &'a Fq>>(iter: I) -> Self {
        Fq(iter.map(|f| f.0).sum())
    }
}

impl std::iter::Product for Fq {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fq(iter.map(|f| f.0).product())
    }
}

impl<'a> std::iter::Product<&'a Fq> for Fq {
    fn product<I: Iterator<Item = &'a Fq>>(iter: I) -> Self {
        Fq(iter.map(|f| f.0).product())
    }
}

impl num_traits::Zero for Fq {
    #[inline(always)]
    fn zero() -> Self {
        Fq(InnerFq::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl num_traits::One for Fq {
    #[inline(always)]
    fn one() -> Self {
        Fq(InnerFq::one())
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl serde::Serialize for Fq {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = [0u8; 32];
        self.0
            .serialize_compressed(&mut buf[..])
            .map_err(serde::ser::Error::custom)?;
        <[u8; 32]>::serialize(&buf, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Fq {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <[u8; 32]>::deserialize(deserializer)?;
        let inner = InnerFq::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        Ok(Fq(inner))
    }
}

impl ark_serialize::CanonicalSerialize for Fq {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl ark_serialize::Valid for Fq {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.0.check()
    }
}

impl ark_serialize::CanonicalDeserialize for Fq {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        InnerFq::deserialize_with_mode(reader, compress, validate).map(Fq)
    }
}

impl UniformRand for Fq {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Fq(<InnerFq as UniformRand>::rand(rng))
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for Fq {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

impl Fq {
    /// Deserializes from little-endian bytes, reducing modulo the field prime.
    #[inline]
    pub fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        Fq(InnerFq::from_le_bytes_mod_order(bytes))
    }

    /// Converts a limb array to a field element without checking that it is
    /// less than the modulus.
    #[inline]
    pub fn from_bigint_unchecked(limbs: Limbs<4>) -> Self {
        let Some(inner) = InnerFq::from_bigint(ark_ff::BigInt::new(limbs.0)) else {
            unreachable!("unchecked BN254 Fq construction received non-canonical limbs")
        };
        Fq(inner)
    }

    /// Access the internal Montgomery-form limbs.
    #[inline(always)]
    pub fn inner_limbs(self) -> Limbs<4> {
        Limbs((self.0).0 .0)
    }
}

impl AdditiveGroup for Fq {}

impl RingCore for Fq {
    #[inline]
    fn square(&self) -> Self {
        Fq(<InnerFq as ark_ff::Field>::square(&self.0))
    }
}

impl Invertible for Fq {
    #[inline]
    fn inverse(&self) -> Option<Self> {
        <InnerFq as ark_ff::Field>::inverse(&self.0).map(Fq)
    }
}

impl FieldCore for Fq {}

impl FixedByteSize for Fq {
    const NUM_BYTES: usize = 32;
}

impl CanonicalBytes for Fq {
    #[expect(clippy::expect_used)]
    #[inline]
    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), <Self as FixedByteSize>::NUM_BYTES);
        use ark_serialize::CanonicalSerialize;
        self.0
            .serialize_compressed(out)
            .expect("BN254 Fq always serializes to 32 bytes");
    }
}

impl ReducingBytes for Fq {
    #[inline]
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        Fq::from_le_bytes_mod_order(bytes)
    }
}

impl TranscriptChallenge for Fq {
    #[inline]
    fn from_challenge_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        let len = bytes.len().min(buf.len());
        buf[..len].copy_from_slice(&bytes[..len]);
        let value = u128::from_le_bytes(buf);
        let low = value as u64;
        let high = ((value >> 64) as u64) & (u64::MAX >> 3);
        let Some(inner) = InnerFq::from_bigint(ark_ff::BigInt::new([0, 0, low, high])) else {
            unreachable!("masked 125-bit shifted challenge fits in BN254 Fq")
        };
        Fq(inner)
    }

    #[inline]
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        let mut buf = bytes.to_vec();
        buf.reverse();
        Fq::from_le_bytes_mod_order(&buf)
    }
}

impl FixedBytes<32> for Fq {}

impl CanonicalU64 for Fq {
    #[inline]
    fn to_canonical_u64_checked(&self) -> Option<u64> {
        let bigint = <InnerFq as PrimeField>::into_bigint(self.0);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as FromPrimitiveInt>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }
}

impl CanonicalBitLength for Fq {
    #[inline]
    fn num_bits(&self) -> u32 {
        <InnerFq as PrimeField>::into_bigint(self.0).num_bits()
    }
}

impl RandomSampling for Fq {
    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Fq(<InnerFq as UniformRand>::rand(rng))
    }
}

impl FromPrimitiveInt for Fq {
    #[inline]
    fn from_u64(n: u64) -> Self {
        Fq(InnerFq::from(n))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            -Fq(InnerFq::from(val.unsigned_abs()))
        } else {
            Fq(InnerFq::from(val as u64))
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            -Fq(InnerFq::from(val.unsigned_abs()))
        } else {
            Fq(InnerFq::from(val as u128))
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        Fq(InnerFq::from(val))
    }
}

impl WithAccumulator for Fq {
    type Accumulator = NaiveAccumulator<Fq>;
}

impl WithSmallScalarAccumulator for Fq {
    type SmallScalarAccumulator = NaiveSignedScalarAccumulator<Fq>;
}

impl WithSignedProductAccumulator for Fq {
    type SignedProductAccumulator = NaiveSignedProductAccumulator<Fq>;
}

impl crate::MulPow2 for Fq {}

impl MulPrimitiveInt for Fq {}

impl Field for Fq {}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{CanonicalU64, FixedBytes};

    #[test]
    fn field_arithmetic_basic() {
        let a = Fq::from_u64(7);
        let b = Fq::from_u64(6);
        assert_eq!(a * b, Fq::from_u64(42));
        assert_eq!(a + b, Fq::from_u64(13));
        assert_eq!(b - a, Fq::from_i64(-1));
    }

    #[test]
    fn serialization_roundtrip() {
        let val = Fq::from_u64(123_456_789);
        let bytes = val.to_bytes_array();
        let recovered = Fq::from_bytes_array(&bytes);
        assert_eq!(val, recovered);
    }

    #[test]
    fn inverse_and_square() {
        let a = Fq::from_u64(42);
        let inv = a.inverse().unwrap();
        assert_eq!(a * inv, Fq::one());
        assert!(Fq::zero().inverse().is_none());
        assert_eq!(a.square(), a * a);
    }

    #[test]
    fn to_u64_roundtrip() {
        let value = Fq::from_u64(12345);
        assert_eq!(value.to_canonical_u64_checked(), Some(12345));
    }
}
