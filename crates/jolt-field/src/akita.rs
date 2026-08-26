//! Temporary bootstrap adapter for the pre-cutover `akita-field` type.
//!
//! Implements this crate's contracts for Akita's proof-optimized fp128 field
//! so the adapter stays buildable until the Akita cutover; it is a bootstrap
//! edge, not the target architecture, and is removed in the final migration
//! PR together with the `akita` feature.

use akita_config::proof_optimized::fp128::Field as AkitaField;
use rand_core::RngCore;

use crate::{
    AdditiveGroup, AkitaAccumulator, AkitaSignedAccumulator, CanonicalBytes, CanonicalEncoding,
    Field, Ring, WithAccumulator,
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

    /// Akita transcripts interpret digest bytes directly as little-endian.
    #[inline]
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl WithAccumulator for AkitaField {
    type Accumulator = AkitaAccumulator;
    type SmallScalarAccumulator = AkitaSignedAccumulator;
    type SignedProductAccumulator = AkitaSignedAccumulator;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_challenge_uses_akita_little_endian_convention() {
        let bytes = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        let challenge = <AkitaField as CanonicalEncoding>::from_scalar_challenge_bytes(&bytes);
        let direct = <AkitaField as CanonicalEncoding>::from_bytes_le_reduced(&bytes);
        let mut reversed = bytes;
        reversed.reverse();

        assert_eq!(challenge, direct);
        assert_ne!(
            challenge,
            <AkitaField as CanonicalEncoding>::from_bytes_le_reduced(&reversed)
        );
    }
}
