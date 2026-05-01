/// Significant-bit introspection for canonical representatives.
pub trait CanonicalBitLength {
    /// Number of significant bits in this element's canonical representative.
    fn num_bits(&self) -> u32;
}
