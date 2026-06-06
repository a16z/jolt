use crate::{WitnessError, WitnessNamespace, WitnessProvider};

pub trait WitnessBuilder<F> {
    type Namespace: WitnessNamespace;
    type Config;
    type Inputs<'a>
    where
        Self: 'a,
        F: 'a;
    type Witness<'a>: WitnessProvider<F, Self::Namespace>
    where
        Self: 'a,
        F: 'a;

    fn build<'a>(
        &mut self,
        config: &Self::Config,
        inputs: Self::Inputs<'a>,
    ) -> Result<Self::Witness<'a>, WitnessError>
    where
        Self: 'a,
        F: 'a;
}
