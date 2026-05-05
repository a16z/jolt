/// Reducing little-endian byte constructor.
pub trait ReducingBytes: Sized {
    /// Deserializes little-endian bytes by reducing into this type.
    fn from_le_bytes_mod_order(bytes: &[u8]) -> Self;
}
