use akita_config::proof_optimized::fp128::Field as AkitaField;
use rand_core::RngCore;

use crate::{
    AdditiveGroup, CanonicalBitLength, CanonicalBytes, CanonicalU64, Field, FieldCore,
    FixedByteSize, FixedBytes, FromPrimitiveInt, Invertible, MulPow2, MulPrimitiveInt,
    NaiveAccumulator, NaiveSignedProductAccumulator, NaiveSignedScalarAccumulator, RandomSampling,
    ReducingBytes, RingCore, TranscriptChallenge, WithAccumulator, WithSignedProductAccumulator,
    WithSmallScalarAccumulator,
};

impl AdditiveGroup for AkitaField {}

impl RingCore for AkitaField {}

impl Invertible for AkitaField {
    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Self as akita_field::Invertible>::inverse(self)
    }
}

impl FieldCore for AkitaField {}

impl FromPrimitiveInt for AkitaField {
    #[inline]
    fn from_u64(v: u64) -> Self {
        <Self as akita_field::FromPrimitiveInt>::from_u64(v)
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        <Self as akita_field::FromPrimitiveInt>::from_i64(v)
    }

    #[inline]
    fn from_u128(v: u128) -> Self {
        <Self as akita_field::FromPrimitiveInt>::from_u128(v)
    }

    #[inline]
    fn from_i128(v: i128) -> Self {
        <Self as akita_field::FromPrimitiveInt>::from_i128(v)
    }
}

impl RandomSampling for AkitaField {
    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        <Self as akita_field::RandomSampling>::random(rng)
    }
}

impl MulPow2 for AkitaField {}

impl MulPrimitiveInt for AkitaField {}

impl FixedByteSize for AkitaField {
    const NUM_BYTES: usize = <Self as akita_field::FixedByteSize>::NUM_BYTES;
}

impl CanonicalBytes for AkitaField {
    #[inline(always)]
    fn to_bytes_le(&self, out: &mut [u8]) {
        <Self as akita_field::CanonicalBytes>::to_bytes_le(self, out);
    }
}

impl ReducingBytes for AkitaField {
    #[inline(always)]
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        <Self as akita_field::ReducingBytes>::from_le_bytes_mod_order(bytes)
    }
}

impl TranscriptChallenge for AkitaField {
    #[inline(always)]
    fn from_challenge_bytes(bytes: &[u8]) -> Self {
        <Self as ReducingBytes>::from_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        // Scalar challenges match the legacy transcript convention: digest bytes
        // are interpreted as a big-endian integer before reduction.
        let mut buf = bytes.to_vec();
        buf.reverse();
        <Self as ReducingBytes>::from_le_bytes_mod_order(&buf)
    }
}

impl FixedBytes<16> for AkitaField {}

impl CanonicalBitLength for AkitaField {
    #[inline]
    fn num_bits(&self) -> u32 {
        <Self as akita_field::CanonicalBitLength>::num_bits(self)
    }
}

impl CanonicalU64 for AkitaField {
    #[inline]
    fn to_canonical_u64_checked(&self) -> Option<u64> {
        <Self as akita_field::CanonicalU64>::to_canonical_u64_checked(self)
    }
}

impl WithAccumulator for AkitaField {
    type Accumulator = NaiveAccumulator<Self>;
}

impl WithSmallScalarAccumulator for AkitaField {
    type SmallScalarAccumulator = NaiveSignedScalarAccumulator<Self>;
}

impl WithSignedProductAccumulator for AkitaField {
    type SignedProductAccumulator = NaiveSignedProductAccumulator<Self>;
}

impl Field for AkitaField {}
