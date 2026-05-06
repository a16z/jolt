use crate::{CanonicalBytes, FixedByteSize, ReducingBytes};

/// Fixed-array convenience API for canonical field/value encodings.
pub trait FixedBytes<const N: usize>: CanonicalBytes + ReducingBytes + FixedByteSize {
    /// Returns the canonical fixed-size byte encoding.
    #[inline]
    fn to_bytes_array(&self) -> [u8; N] {
        debug_assert_eq!(Self::NUM_BYTES, N);
        let mut out = [0u8; N];
        self.to_bytes_le(&mut out);
        out
    }

    /// Reducing constructor from a fixed-size byte array.
    #[inline]
    fn from_bytes_array(bytes: &[u8; N]) -> Self {
        Self::from_le_bytes_mod_order(bytes)
    }
}
