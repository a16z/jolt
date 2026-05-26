use jolt_witness::{ViewRequirement, WitnessNamespace};

use crate::{BackendRelationId, BackendValueSlot};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SumcheckSlot(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckInstanceRequest<N: WitnessNamespace> {
    pub slot: SumcheckSlot,
    pub relation: BackendRelationId,
    pub witness_views: Vec<ViewRequirement<N>>,
    pub rounds: usize,
    pub degree: usize,
    pub input_claim: BackendValueSlot,
    pub output_claim: BackendValueSlot,
    #[cfg(feature = "zk")]
    pub committed_rounds: bool,
}

impl<N: WitnessNamespace> SumcheckInstanceRequest<N> {
    pub fn new(
        slot: SumcheckSlot,
        relation: BackendRelationId,
        witness_views: Vec<ViewRequirement<N>>,
        rounds: usize,
        degree: usize,
        input_claim: BackendValueSlot,
        output_claim: BackendValueSlot,
    ) -> Self {
        Self {
            slot,
            relation,
            witness_views,
            rounds,
            degree,
            input_claim,
            output_claim,
            #[cfg(feature = "zk")]
            committed_rounds: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRequest<N: WitnessNamespace> {
    pub label: &'static str,
    pub instances: Vec<SumcheckInstanceRequest<N>>,
}

impl<N: WitnessNamespace> SumcheckRequest<N> {
    pub const fn new(label: &'static str, instances: Vec<SumcheckInstanceRequest<N>>) -> Self {
        Self { label, instances }
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}
