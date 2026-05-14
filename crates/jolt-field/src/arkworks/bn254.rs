//! Newtype wrapper around `ark_bn254::Fr` that decouples the public API from arkworks.
//!
//! [`Fr`] is `#[repr(transparent)]` over the inner arkworks scalar field element,
//! so it has identical layout and can be transmuted where needed.
use crate::{
    AdditiveGroup, CanonicalBitLength, CanonicalBytes, CanonicalU64, Field, FieldCore,
    FixedByteSize, FixedBytes, FromPrimitiveInt, Invertible, Limbs, MulPrimitiveInt,
    RandomSampling, ReducingBytes, RingCore, TranscriptChallenge, WithAccumulator,
};
use ark_bn254::Fr as ArkFr;
use ark_ff::{prelude::*, Field as ArkField, PrimeField, UniformRand};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use rand_core::RngCore;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    fmt::{self, Debug, Display},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::bn254_ops;

type InnerFr = ArkFr;

/// BN254 scalar field element.
///
/// A `#[repr(transparent)]` newtype over `ark_bn254::Fr`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Fr(pub(crate) InnerFr);

impl From<ArkFr> for Fr {
    #[inline(always)]
    fn from(inner: ArkFr) -> Self {
        Fr(inner)
    }
}

impl From<Fr> for ArkFr {
    #[inline(always)]
    fn from(wrapper: Fr) -> Self {
        wrapper.0
    }
}

impl From<bool> for Fr {
    #[inline(always)]
    fn from(v: bool) -> Self {
        <Self as FromPrimitiveInt>::from_bool(v)
    }
}

impl From<u8> for Fr {
    #[inline(always)]
    fn from(v: u8) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v as u64)
    }
}

impl From<u16> for Fr {
    #[inline(always)]
    fn from(v: u16) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v as u64)
    }
}

impl From<u32> for Fr {
    #[inline(always)]
    fn from(v: u32) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v as u64)
    }
}

impl From<u64> for Fr {
    #[inline(always)]
    fn from(v: u64) -> Self {
        <Self as FromPrimitiveInt>::from_u64(v)
    }
}

impl From<i64> for Fr {
    #[inline(always)]
    fn from(v: i64) -> Self {
        <Self as FromPrimitiveInt>::from_i64(v)
    }
}

impl From<i128> for Fr {
    #[inline(always)]
    fn from(v: i128) -> Self {
        <Self as FromPrimitiveInt>::from_i128(v)
    }
}

impl From<u128> for Fr {
    #[inline(always)]
    fn from(v: u128) -> Self {
        <Self as FromPrimitiveInt>::from_u128(v)
    }
}

impl Debug for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

macro_rules! delegate_binop {
    ($Trait:ident, $method:ident) => {
        impl $Trait for Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: Fr) -> Fr {
                Fr($Trait::$method(self.0, rhs.0))
            }
        }

        impl $Trait<&Fr> for Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: &Fr) -> Fr {
                Fr($Trait::$method(self.0, &rhs.0))
            }
        }

        impl $Trait<Fr> for &Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: Fr) -> Fr {
                Fr($Trait::$method(self.0, rhs.0))
            }
        }

        impl<'a, 'b> $Trait<&'b Fr> for &'a Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: &'b Fr) -> Fr {
                Fr($Trait::$method(self.0, &rhs.0))
            }
        }
    };
}

delegate_binop!(Add, add);
delegate_binop!(Sub, sub);
delegate_binop!(Mul, mul);
delegate_binop!(Div, div);

impl Neg for Fr {
    type Output = Fr;
    #[inline(always)]
    fn neg(self) -> Fr {
        Fr(self.0.neg())
    }
}

impl AddAssign for Fr {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Fr) {
        self.0.add_assign(rhs.0);
    }
}

impl SubAssign for Fr {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Fr) {
        self.0.sub_assign(rhs.0);
    }
}

impl MulAssign for Fr {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fr) {
        self.0.mul_assign(rhs.0);
    }
}

impl Sum for Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).sum())
    }
}

impl<'a> Sum<&'a Fr> for Fr {
    fn sum<I: Iterator<Item = &'a Fr>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).sum())
    }
}

impl Product for Fr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).product())
    }
}

impl<'a> Product<&'a Fr> for Fr {
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

impl Serialize for Fr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut buf = [0u8; 32];
        self.0
            .serialize_compressed(&mut buf[..])
            .map_err(serde::ser::Error::custom)?;
        <[u8; 32]>::serialize(&buf, serializer)
    }
}

impl<'de> Deserialize<'de> for Fr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf = <[u8; 32]>::deserialize(deserializer)?;
        let inner = InnerFr::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        Ok(Fr(inner))
    }
}

impl CanonicalSerialize for Fr {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl Valid for Fr {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl CanonicalDeserialize for Fr {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
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
    pub fn from_bigint_unchecked(limbs: Limbs<4>) -> Self {
        Fr(bn254_ops::from_bigint_unchecked(limbs.into()))
    }

    /// Access the internal Montgomery-form limbs.
    ///
    /// Used by [`WideAccumulator`](super::wide_accumulator::WideAccumulator)
    /// for deferred-reduction fused multiply-add.
    #[inline(always)]
    pub fn inner_limbs(self) -> Limbs<4> {
        Limbs((self.0).0 .0)
    }

    /// Multiplies this field element by a 125-bit challenge stored in the high
    /// two Montgomery limbs used by Jolt's optimized challenge type.
    #[inline(always)]
    pub fn mul_by_hi_2limbs(&self, limb_lo: u64, limb_hi: u64) -> Self {
        Fr(self.0.mul_by_hi_2limbs(limb_lo, limb_hi))
    }

    /// Construct from the inner arkworks element.
    #[inline(always)]
    pub(crate) fn from_inner(inner: InnerFr) -> Self {
        Fr(inner)
    }
}

impl AdditiveGroup for Fr {}

impl RingCore for Fr {
    #[inline]
    fn square(&self) -> Self {
        Fr(<InnerFr as ArkField>::square(&self.0))
    }
}

impl Invertible for Fr {
    #[inline]
    fn inverse(&self) -> Option<Self> {
        <InnerFr as ArkField>::inverse(&self.0).map(Fr)
    }
}

impl FieldCore for Fr {}

impl FixedByteSize for Fr {
    const NUM_BYTES: usize = 32;
}

impl CanonicalBytes for Fr {
    #[expect(clippy::expect_used)]
    #[inline]
    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), <Self as FixedByteSize>::NUM_BYTES);
        self.0
            .serialize_compressed(out)
            .expect("BN254 Fr always serializes to 32 bytes");
    }
}

impl ReducingBytes for Fr {
    #[inline]
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        Fr::from_le_bytes_mod_order(bytes)
    }
}

impl TranscriptChallenge for Fr {
    #[inline]
    fn from_challenge_bytes(bytes: &[u8]) -> Self {
        Fr::from_le_bytes_mod_order(bytes)
    }
}

impl FixedBytes<32> for Fr {}

impl CanonicalU64 for Fr {
    #[inline]
    fn to_canonical_u64_checked(&self) -> Option<u64> {
        let bigint = <InnerFr as PrimeField>::into_bigint(self.0);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as FromPrimitiveInt>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }
}

impl CanonicalBitLength for Fr {
    #[inline]
    fn num_bits(&self) -> u32 {
        <InnerFr as PrimeField>::into_bigint(self.0).num_bits()
    }
}

impl RandomSampling for Fr {
    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Fr(<InnerFr as UniformRand>::rand(rng))
    }
}

// The custom limb reducers are a native hot path. WASM verification uses
// arkworks' portable constructors and multiplication so proofs generated on
// native targets verify identically in the browser/Node runtime.
impl FromPrimitiveInt for Fr {
    #[inline]
    fn from_u64(n: u64) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Fr(InnerFr::from(n))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Fr(bn254_ops::from_u64(n))
        }
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let abs = Fr(InnerFr::from(val.unsigned_abs()));
            if val.is_negative() {
                -abs
            } else {
                abs
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if val.is_negative() {
                -Fr(bn254_ops::from_u64(val.unsigned_abs()))
            } else {
                Fr(bn254_ops::from_u64(val as u64))
            }
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let abs = Fr(InnerFr::from(val.unsigned_abs()));
            if val.is_negative() {
                -abs
            } else {
                abs
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if val.is_negative() {
                -Fr(bn254_ops::from_u128(val.unsigned_abs()))
            } else {
                Fr(bn254_ops::from_u128(val as u128))
            }
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Fr(InnerFr::from(val))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Fr(bn254_ops::from_u128(val))
        }
    }
}

impl WithAccumulator for Fr {
    type Accumulator = super::wide_accumulator::WideAccumulator;
}

impl crate::MulPow2 for Fr {}

impl MulPrimitiveInt for Fr {
    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Fr(self.0 * InnerFr::from(n))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Fr(bn254_ops::mul_u64(self.0, n))
        }
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let abs = Fr(self.0 * InnerFr::from(n.unsigned_abs()));
            if n.is_negative() {
                -abs
            } else {
                abs
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Fr(bn254_ops::mul_i64(self.0, n))
        }
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Fr(self.0 * InnerFr::from(n))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Fr(bn254_ops::mul_u128(self.0, n))
        }
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let abs = Fr(self.0 * InnerFr::from(n.unsigned_abs()));
            if n.is_negative() {
                -abs
            } else {
                abs
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Fr(bn254_ops::mul_i128(self.0, n))
        }
    }
}

impl Field for Fr {}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{CanonicalU64, FixedBytes};

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
        let bytes = val.to_bytes_array();
        let recovered = Fr::from_bytes_array(&bytes);
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
        assert_eq!(Fr::from_u64(999).to_canonical_u64_checked(), Some(999));
        // Large field element should not fit in u64
        let big = Fr::from_u128(u128::MAX / 2);
        assert_eq!(big.to_canonical_u64_checked(), None);
    }

    #[test]
    fn inner_limbs_roundtrip() {
        let val = Fr::from_u64(42);
        let limbs = val.inner_limbs();
        let recovered = Fr::from_bigint_unchecked(limbs);
        assert_eq!(val, recovered);
    }
}
