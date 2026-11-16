//! Fixed byte-tuple decoding with register-sized accumulators.

use core::fmt::{Formatter, Result as FormatResult};

use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::Fp128;
use crate::{CanonicalBytes, CanonicalEncoding};

struct WordArrayVisitor;

impl<'de> Visitor<'de> for WordArrayVisitor {
    type Value = u128;

    fn expecting(&self, formatter: &mut Formatter) -> FormatResult {
        formatter.write_str("an array of length 16")
    }

    #[inline]
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<u128, A::Error> {
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..16 {
            let byte = seq
                .next_element::<u8>()?
                .ok_or_else(|| Error::invalid_length(i, &self))?;
            // Each shift is at most 56; every byte owns disjoint bits.
            if i < 8 {
                lo |= u64::from(byte) << (8 * i);
            } else {
                hi |= u64::from(byte) << (8 * (i - 8));
            }
        }
        Ok(u128::from(lo) | (u128::from(hi) << 64))
    }
}

impl<const P: u128> Serialize for Fp128<P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = [0u8; 16];
        self.to_bytes_le(&mut bytes);
        bytes.serialize(serializer)
    }
}

impl<'de, const P: u128> Deserialize<'de> for Fp128<P> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Preserve the byte-tuple grammar, including its per-byte error order.
        let value = deserializer.deserialize_tuple(16, WordArrayVisitor)?;
        Self::from_u128_checked(value)
            .ok_or_else(|| Error::custom("non-canonical field element encoding"))
    }
}
