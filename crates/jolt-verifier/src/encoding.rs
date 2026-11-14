use bincode::error::{DecodeError, EncodeError};
use serde::{de::DeserializeOwned, Serialize};

/// A failure while decoding a verifier object at an untrusted byte boundary.
#[derive(Debug, thiserror::Error)]
pub enum VerifierObjectDecodeError {
    #[error("failed to decode verifier object: {0}")]
    Decode(#[from] DecodeError),
    #[error("failed to re-encode verifier object: {0}")]
    Encode(#[from] EncodeError),
    #[error("verifier object has a non-canonical encoding")]
    NonCanonicalEncoding,
}

pub fn serialize_verifier_object<T: Serialize>(item: &T) -> Result<Vec<u8>, EncodeError> {
    bincode::serde::encode_to_vec(item, bincode::config::standard())
}

/// Decodes a verifier object only when `bytes` is its canonical encoding.
///
/// Bincode's standard decoder accepts non-minimal integer encodings. The
/// deterministic encoder maps every decoded value to one byte string, so an
/// exact decode-reencode comparison rejects overlong integers, non-canonical
/// nested encodings, and trailing bytes at the boundary.
pub fn deserialize_verifier_object<T>(bytes: &[u8]) -> Result<T, VerifierObjectDecodeError>
where
    T: DeserializeOwned + Serialize,
{
    let (value, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
    let canonical = serialize_verifier_object(&value)?;
    if canonical != bytes {
        return Err(VerifierObjectDecodeError::NonCanonicalEncoding);
    }
    Ok(value)
}
