#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PolynomialEncoding {
    #[default]
    Dense,
    Compact,
    OneHot,
    SparseEvents,
    Derived,
    Streaming,
}
