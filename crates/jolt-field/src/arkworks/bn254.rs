//! Newtype wrapper around `ark_bn254::Fr` that decouples the public API from arkworks.
//!
//! [`Fr`] is `#[repr(transparent)]` over the inner arkworks scalar field element,
//! so it has identical layout and can be transmuted where needed.
use crate::{Field, Limbs};
use ark_ff::{prelude::*, PrimeField, UniformRand};
use rand_core::RngCore;

use super::bn254_ops;

type InnerFr = ark_bn254::Fr;

/// BN254 scalar field element.
///
/// A `#[repr(transparent)]` newtype over `ark_bn254::Fr`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Fr(pub(crate) InnerFr);

impl From<ark_bn254::Fr> for Fr {
    #[inline(always)]
    fn from(inner: ark_bn254::Fr) -> Self {
        Fr(inner)
    }
}

impl From<Fr> for ark_bn254::Fr {
    #[inline(always)]
    fn from(wrapper: Fr) -> Self {
        wrapper.0
    }
}

impl From<bool> for Fr {
    #[inline(always)]
    fn from(v: bool) -> Self {
        <Self as Field>::from_bool(v)
    }
}

impl From<u8> for Fr {
    #[inline(always)]
    fn from(v: u8) -> Self {
        <Self as Field>::from_u64(v as u64)
    }
}

impl From<u16> for Fr {
    #[inline(always)]
    fn from(v: u16) -> Self {
        <Self as Field>::from_u64(v as u64)
    }
}

impl From<u32> for Fr {
    #[inline(always)]
    fn from(v: u32) -> Self {
        <Self as Field>::from_u64(v as u64)
    }
}

impl From<u64> for Fr {
    #[inline(always)]
    fn from(v: u64) -> Self {
        <Self as Field>::from_u64(v)
    }
}

impl From<i64> for Fr {
    #[inline(always)]
    fn from(v: i64) -> Self {
        <Self as Field>::from_i64(v)
    }
}

impl From<i128> for Fr {
    #[inline(always)]
    fn from(v: i128) -> Self {
        <Self as Field>::from_i128(v)
    }
}

impl From<u128> for Fr {
    #[inline(always)]
    fn from(v: u128) -> Self {
        <Self as Field>::from_u128(v)
    }
}

impl std::fmt::Debug for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

macro_rules! delegate_binop {
    ($Trait:ident, $method:ident) => {
        impl std::ops::$Trait for Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl std::ops::$Trait<&Fr> for Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: &Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, &rhs.0))
            }
        }

        impl std::ops::$Trait<Fr> for &Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl<'a, 'b> std::ops::$Trait<&'b Fr> for &'a Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: &'b Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, &rhs.0))
            }
        }
    };
}

delegate_binop!(Add, add);
delegate_binop!(Sub, sub);
delegate_binop!(Mul, mul);
delegate_binop!(Div, div);

impl std::ops::Neg for Fr {
    type Output = Fr;
    #[inline(always)]
    fn neg(self) -> Fr {
        Fr(self.0.neg())
    }
}

impl std::ops::AddAssign for Fr {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Fr) {
        self.0.add_assign(rhs.0);
    }
}

impl std::ops::SubAssign for Fr {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Fr) {
        self.0.sub_assign(rhs.0);
    }
}

impl std::ops::MulAssign for Fr {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fr) {
        self.0.mul_assign(rhs.0);
    }
}

impl std::iter::Sum for Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).sum())
    }
}

impl<'a> std::iter::Sum<&'a Fr> for Fr {
    fn sum<I: Iterator<Item = &'a Fr>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).sum())
    }
}

impl std::iter::Product for Fr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).product())
    }
}

impl<'a> std::iter::Product<&'a Fr> for Fr {
    fn product<I: Iterator<Item = &'a Fr>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).product())
    }
}

impl num_traits::Zero for Fr {
    #[inline(always)]
    fn zero() -> Self {
        Fr(InnerFr::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl num_traits::One for Fr {
    #[inline(always)]
    fn one() -> Self {
        Fr(InnerFr::one())
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl serde::Serialize for Fr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = [0u8; 32];
        self.0
            .serialize_compressed(&mut buf[..])
            .map_err(serde::ser::Error::custom)?;
        <[u8; 32]>::serialize(&buf, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Fr {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <[u8; 32]>::deserialize(deserializer)?;
        let inner = InnerFr::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        Ok(Fr(inner))
    }
}

impl ark_serialize::CanonicalSerialize for Fr {
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

impl ark_serialize::Valid for Fr {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.0.check()
    }
}

impl ark_serialize::CanonicalDeserialize for Fr {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        InnerFr::deserialize_with_mode(reader, compress, validate).map(Fr)
    }
}

impl UniformRand for Fr {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Fr(<InnerFr as UniformRand>::rand(rng))
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for Fr {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

impl Fr {
    /// Deserializes from little-endian bytes, reducing modulo the field prime.
    #[inline]
    pub fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        Fr(InnerFr::from_le_bytes_mod_order(bytes))
    }

    /// Converts a limb array to a field element without checking that it is
    /// less than the modulus.
    #[inline]
    pub fn from_bigint_unchecked(limbs: Limbs<4>) -> Option<Self> {
        Some(Fr(bn254_ops::from_bigint_unchecked(limbs.to_bigint())))
    }

    /// Access the internal Montgomery-form limbs.
    ///
    /// Used by [`WideAccumulator`](super::wide_accumulator::WideAccumulator)
    /// for deferred-reduction fused multiply-add.
    #[inline(always)]
    pub fn inner_limbs(self) -> Limbs<4> {
        Limbs((self.0).0 .0)
    }

    /// Construct from the inner arkworks element.
    #[inline(always)]
    pub(crate) fn from_inner(inner: InnerFr) -> Self {
        Fr(inner)
    }
}

impl Field for Fr {
    type Accumulator = super::wide_accumulator::WideAccumulator;

    const NUM_BYTES: usize = 32;

    fn to_bytes(&self) -> [u8; 32] {
        use ark_serialize::CanonicalSerialize;
        let mut buf = [0u8; 32];
        self.0
            .serialize_compressed(&mut buf[..])
            .expect("field serialization should not fail");
        buf
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Fr(<InnerFr as UniformRand>::rand(rng))
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Fr::from_le_bytes_mod_order(bytes)
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = <InnerFr as PrimeField>::into_bigint(self.0);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as Field>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    fn num_bits(&self) -> u32 {
        <InnerFr as PrimeField>::into_bigint(self.0).num_bits()
    }

    fn square(&self) -> Self {
        Fr(<InnerFr as ark_ff::Field>::square(&self.0))
    }

    fn inverse(&self) -> Option<Self> {
        <InnerFr as ark_ff::Field>::inverse(&self.0).map(Fr)
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        Fr(bn254_ops::from_u64(n))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            -Fr(bn254_ops::from_u64(val.unsigned_abs()))
        } else {
            Fr(bn254_ops::from_u64(val as u64))
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            -Fr(bn254_ops::from_u128(val.unsigned_abs()))
        } else {
            Fr(bn254_ops::from_u128(val as u128))
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        Fr(bn254_ops::from_u128(val))
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        Fr(bn254_ops::mul_u64(self.0, n))
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        Fr(bn254_ops::mul_i64(self.0, n))
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        Fr(bn254_ops::mul_u128(self.0, n))
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        Fr(bn254_ops::mul_i128(self.0, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn field_arithmetic_basic() {
        let a = Fr::from_u64(7);
        let b = Fr::from_u64(6);
        assert_eq!(a * b, Fr::from_u64(42));
        assert_eq!(a + b, Fr::from_u64(13));
        assert_eq!(b - a, Fr::from_i64(-1));
    }

    #[test]
    fn from_signed() {
        let neg_one = Fr::from_i64(-1);
        let one = Fr::one();
        assert_eq!(neg_one + one, Fr::zero());

        let neg_big = Fr::from_i128(-1_000_000_000_000i128);
        let pos_big = Fr::from_u128(1_000_000_000_000u128);
        assert_eq!(neg_big + pos_big, Fr::zero());
    }

    #[test]
    fn serialization_roundtrip() {
        let val = Fr::from_u64(123_456_789);
        let bytes = val.to_bytes();
        let recovered = Fr::from_bytes(&bytes);
        assert_eq!(val, recovered);
    }

    #[test]
    fn inverse_and_square() {
        let a = Fr::from_u64(42);
        let inv = a.inverse().unwrap();
        assert_eq!(a * inv, Fr::one());
        assert!(Fr::zero().inverse().is_none());

        assert_eq!(a.square(), a * a);
    }

    #[test]
    fn to_u64_roundtrip() {
        assert_eq!(Fr::from_u64(999).to_u64(), Some(999));
        // Large field element should not fit in u64
        let big = Fr::from_u128(u128::MAX / 2);
        assert_eq!(big.to_u64(), None);
    }

    #[test]
    fn inner_limbs_roundtrip() {
        let val = Fr::from_u64(42);
        let limbs = val.inner_limbs();
        let recovered = Fr::from_bigint_unchecked(limbs).unwrap();
        assert_eq!(val, recovered);
    }
}
