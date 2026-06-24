use serde::{Deserialize, Serialize};

/// The domain a sumcheck runs over. Shared by every protocol's relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SumcheckDomain {
    BooleanHypercube,
    CenteredInteger { domain_size: usize },
}

/// A sumcheck's shape: its domain, round count, and per-round degree. Shared
/// across protocols (the jolt and field_inline relations both use it), so it is
/// the return type of [`SymbolicSumcheck::sumcheck`](crate::SymbolicSumcheck::sumcheck).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckSpec {
    pub domain: SumcheckDomain,
    pub rounds: usize,
    pub degree: usize,
}

impl SumcheckSpec {
    pub const fn boolean(rounds: usize, degree: usize) -> Self {
        Self {
            domain: SumcheckDomain::BooleanHypercube,
            rounds,
            degree,
        }
    }

    pub const fn centered_integer(domain_size: usize, rounds: usize, degree: usize) -> Self {
        Self {
            domain: SumcheckDomain::CenteredInteger { domain_size },
            rounds,
            degree,
        }
    }
}
