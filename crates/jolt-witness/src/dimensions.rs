/// Dimensions of a multilinear witness polynomial.
///
/// The evaluation domain is always a power of two, so only the variable count
/// is stored; the row count is derived.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WitnessDimensions {
    pub log_rows: usize,
}

impl WitnessDimensions {
    /// Panics if `log_rows >= usize::BITS`; a domain that large is a
    /// programming error, not recoverable input.
    pub const fn new(log_rows: usize) -> Self {
        assert!(
            log_rows < usize::BITS as usize,
            "log_rows must be smaller than usize::BITS"
        );
        Self { log_rows }
    }

    pub const fn rows(&self) -> usize {
        1 << self.log_rows
    }
}
