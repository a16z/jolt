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
