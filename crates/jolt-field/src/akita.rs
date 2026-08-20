//! Temporary bootstrap adapter for the pre-cutover `akita-field` type.
//!
//! Implements this crate's contracts for Akita's proof-optimized fp128 field
//! so the adapter stays buildable until the Akita cutover; it is a bootstrap
//! edge, not the target architecture, and is removed in the final migration
//! PR together with the `akita` feature.

use akita_config::proof_optimized::fp128::Field as AkitaField;
use rand_core::RngCore;

use crate::{
    AdditiveGroup, CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring,
    WithAccumulator,
};

impl AdditiveGroup for AkitaField {}

impl Ring for AkitaField {
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

impl Field for AkitaField {
    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Self as akita_field::Invertible>::inverse(self)
    }

    #[inline]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        <Self as akita_field::RandomSampling>::random(rng)
    }
}

impl CanonicalBytes for AkitaField {
    const NUM_BYTES: usize = <Self as akita_field::FixedByteSize>::NUM_BYTES;

    #[inline(always)]
    fn to_bytes_le(&self, out: &mut [u8]) {
        <Self as akita_field::CanonicalBytes>::to_bytes_le(self, out);
    }
}

impl CanonicalEncoding for AkitaField {
    // Akita's proof-optimized field is a 128-bit pseudo-Mersenne prime.
    const MODULUS_BITS: u32 = 128;

    #[inline(always)]
    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        <Self as akita_field::ReducingBytes>::from_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != <Self as CanonicalBytes>::NUM_BYTES {
            return None;
        }
        let value = Self::from_bytes_le_reduced(bytes);
        // Canonical iff decoding round-trips to the identical bytes.
        (value.to_bytes_le_vec() == bytes).then_some(value)
    }

    #[inline]
    fn to_u128_checked(&self) -> Option<u128> {
        let mut buf = [0u8; 16];
        CanonicalBytes::to_bytes_le(self, &mut buf);
        Some(u128::from_le_bytes(buf))
    }

    #[inline]
    fn from_u128_checked(v: u128) -> Option<Self> {
        let value = <Self as akita_field::FromPrimitiveInt>::from_u128(v);
        (value.to_u128_checked() == Some(v)).then_some(value)
    }

    #[inline]
    fn from_u128_reduced(v: u128) -> Self {
        <Self as akita_field::FromPrimitiveInt>::from_u128(v)
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <Self as akita_field::CanonicalBitLength>::num_bits(self)
    }

    /// Legacy convention: digest bytes are interpreted as a big-endian
    /// integer before reduction.
    #[inline]
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        let mut buf = bytes.to_vec();
        buf.reverse();
        Self::from_bytes_le_reduced(&buf)
    }
}

impl WithAccumulator for AkitaField {
    type Accumulator = NaiveAccumulator<Self>;
    type SmallScalarAccumulator = NaiveAccumulator<Self>;
    type SignedProductAccumulator = NaiveAccumulator<Self>;
}
