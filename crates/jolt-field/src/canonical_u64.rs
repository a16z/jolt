/// Checked extraction of canonical representatives that fit in `u64`.
pub trait CanonicalU64 {
    /// Returns the canonical representative as `u64` if it fits.
    fn to_canonical_u64_checked(&self) -> Option<u64>;
}
