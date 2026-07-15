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

/// Physical representation of a polynomial's evaluations in views and stream
/// chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolynomialEncoding {
    /// One field element per row.
    #[default]
    Dense,
    /// Small-scalar values (e.g. `u64` words or signed increments) promoted to
    /// field elements on use.
    Compact,
    /// Rows form a `K x T` grid in which each cycle column has at most one
    /// nonzero entry; only the hot index per cycle is stored.
    OneHot,
}

/// A witness polynomial's declared domain and physical representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dimensions: WitnessDimensions,
    pub encoding: PolynomialEncoding,
}

impl Shape {
    pub const fn new(dimensions: WitnessDimensions, encoding: PolynomialEncoding) -> Self {
        Self {
            dimensions,
            encoding,
        }
    }
}
