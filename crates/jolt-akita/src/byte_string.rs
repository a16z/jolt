//! Serde form of a byte payload: one length-prefixed byte string, so a guest
//! decodes it as one copy where bincode would walk a `Vec<u8>` sequence
//! element by element, at roughly 27 guest cycles per byte.
//!
//! Decoding requests `deserialize_bytes`, never `deserialize_byte_buf`
//! (`serde_bytes` on a `Vec<u8>`, or its `ByteBuf`). For the latter bincode 2
//! allocates the announced length before it reads a byte, so a length prefix
//! alone could abort the host. For the former its slice decoder
//! (`bincode::serde::decode_from_slice`) lends the bytes from the input and
//! fails on a length past its end before anything is allocated. A
//! reader-backed bincode decoder allocates the announced length on both
//! requests and must not be given untrusted input.

use std::borrow::Cow;

use serde::Deserializer;
pub(crate) use serde_bytes::serialize;

pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    serde_bytes::deserialize::<Cow<'de, [u8]>, D>(deserializer).map(Cow::into_owned)
}

/// An optional payload, written as `Option<&serde_bytes::Bytes>`.
pub(crate) fn deserialize_option<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<u8>>, D::Error> {
    Ok(serde_bytes::deserialize::<Option<Cow<'de, [u8]>>, D>(deserializer)?.map(Cow::into_owned))
}
