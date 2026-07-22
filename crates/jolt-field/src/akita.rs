use akita_config::proof_optimized::fp128::Field as AkitaField;
use rand_core::RngCore;

use crate::{
    AdditiveGroup, CanonicalRepr, Field, FieldCore, FromPrimitiveInt, NaiveAccumulator, RingCore,
    WithAccumulator,
};

impl AdditiveGroup for AkitaField {}

impl RingCore for AkitaField {}

impl FieldCore for AkitaField {
    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Self as akita_field::Invertible>::inverse(self)
    }

    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        <Self as akita_field::RandomSampling>::random(rng)
    }
}

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

impl CanonicalRepr for AkitaField {
    const NUM_BYTES: usize = <Self as akita_field::FixedByteSize>::NUM_BYTES;

    #[inline(always)]
    fn to_bytes_le(&self, out: &mut [u8]) {
        <Self as akita_field::CanonicalBytes>::to_bytes_le(self, out);
    }

    #[inline(always)]
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        <Self as akita_field::ReducingBytes>::from_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn to_canonical_u64_checked(&self) -> Option<u64> {
        <Self as akita_field::CanonicalU64>::to_canonical_u64_checked(self)
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <Self as akita_field::CanonicalBitLength>::num_bits(self)
    }
}

impl WithAccumulator for AkitaField {
    type Accumulator = NaiveAccumulator<Self>;
}

impl Field for AkitaField {}
