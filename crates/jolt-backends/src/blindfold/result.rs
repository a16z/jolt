use jolt_field::Field;

use crate::BackendValueSlot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldPrivateOpening<F: Field> {
    pub slot: BackendValueSlot,
    pub value: F,
}

impl<F: Field> BlindFoldPrivateOpening<F> {
    pub const fn new(slot: BackendValueSlot, value: F) -> Self {
        Self { slot, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldResult<F: Field, Proof> {
    pub proof: Proof,
    pub private_openings: Vec<BlindFoldPrivateOpening<F>>,
}

impl<F: Field, Proof> BlindFoldResult<F, Proof> {
    pub const fn new(proof: Proof, private_openings: Vec<BlindFoldPrivateOpening<F>>) -> Self {
        Self {
            proof,
            private_openings,
        }
    }
}
