use serde::{Deserialize, Serialize};

/// The domain a sumcheck runs over. Shared by every protocol's relations and
/// returned by [`SymbolicSumcheck::domain`](crate::SymbolicSumcheck::domain).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SumcheckDomain {
    BooleanHypercube,
    CenteredInteger { domain_size: usize },
}

impl SumcheckDomain {
    pub const fn centered_integer(domain_size: usize) -> Self {
        Self::CenteredInteger { domain_size }
    }
}
