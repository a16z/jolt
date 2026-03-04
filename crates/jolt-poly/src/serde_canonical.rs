//! Serde bridge for types implementing arkworks `CanonicalSerialize`/`CanonicalDeserialize`.
//!
//! Arkworks field elements use a custom binary encoding (`CanonicalSerialize`)
//! that is more compact than serde's default derive. This module lets us use
//! `#[serde(with = "vec_canonical")]` on `Vec<F>` fields so that the outer
//! struct can derive `Serialize`/`Deserialize` while the inner field elements
//! are encoded in arkworks' canonical compressed form.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use serde::{Deserializer, Serializer};

/// Serde helper for `Vec<T>` where `T: CanonicalSerialize + CanonicalDeserialize`.
///
/// Usage: `#[serde(with = "vec_canonical")]` on a `Vec<F>` field.
pub mod vec_canonical {
    use super::*;

    pub fn serialize<T, S>(vec: &[T], serializer: S) -> Result<S::Ok, S::Error>
    where
        T: CanonicalSerialize,
        S: Serializer,
    {
        let mut bytes = Vec::new();
        let len = vec.len() as u64;
        CanonicalSerialize::serialize_compressed(&len, &mut bytes)
            .map_err(serde::ser::Error::custom)?;
        for item in vec {
            item.serialize_compressed(&mut bytes)
                .map_err(serde::ser::Error::custom)?;
        }
        serializer.serialize_bytes(&bytes)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
    where
        T: CanonicalDeserialize,
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = {
            struct ByteVisitor;
            impl<'de> serde::de::Visitor<'de> for ByteVisitor {
                type Value = Vec<u8>;
                fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str("byte sequence")
                }
                fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Vec<u8>, E> {
                    Ok(v.to_vec())
                }
                fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
                    Ok(v)
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<Vec<u8>, A::Error> {
                    let mut buf = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                    while let Some(b) = seq.next_element()? {
                        buf.push(b);
                    }
                    Ok(buf)
                }
            }
            deserializer.deserialize_bytes(ByteVisitor)?
        };

        let mut cursor = &bytes[..];
        let len =
            u64::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::Yes)
                .map_err(serde::de::Error::custom)?;
        let mut vec = Vec::with_capacity(len as usize);
        for _ in 0..len {
            vec.push(T::deserialize_compressed(&mut cursor).map_err(serde::de::Error::custom)?);
        }
        Ok(vec)
    }
}
