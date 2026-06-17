use crate::{OracleRef, WitnessNamespace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningWitness<F, N: WitnessNamespace> {
    pub oracle: OracleRef<N>,
    pub point: Vec<F>,
    pub value: F,
}

impl<F, N: WitnessNamespace> OpeningWitness<F, N> {
    pub fn new(oracle: OracleRef<N>, point: Vec<F>, value: F) -> Self {
        Self {
            oracle,
            point,
            value,
        }
    }
}
