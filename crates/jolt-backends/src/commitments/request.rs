use jolt_witness::{ViewRequirement, WitnessNamespace};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CommitmentSlot(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRequestItem<N: WitnessNamespace> {
    pub slot: CommitmentSlot,
    pub requirement: ViewRequirement<N>,
}

impl<N: WitnessNamespace> CommitmentRequestItem<N> {
    pub const fn new(slot: CommitmentSlot, requirement: ViewRequirement<N>) -> Self {
        Self { slot, requirement }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRequest<N: WitnessNamespace> {
    pub items: Vec<CommitmentRequestItem<N>>,
}

impl<N: WitnessNamespace> CommitmentRequest<N> {
    pub const fn new(items: Vec<CommitmentRequestItem<N>>) -> Self {
        Self { items }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
