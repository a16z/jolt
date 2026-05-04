/// Canonical little-endian byte encoding.
pub trait CanonicalBytes: Sized + crate::FixedByteSize {
    /// Writes the canonical little-endian encoding into `out`.
    fn to_bytes_le(&self, out: &mut [u8]);

    /// Returns the canonical little-endian encoding as a vector.
    #[inline]
    fn to_bytes_le_vec(&self) -> Vec<u8> {
        let mut out = vec![0u8; Self::NUM_BYTES];
        self.to_bytes_le(&mut out);
        out
    }
}
