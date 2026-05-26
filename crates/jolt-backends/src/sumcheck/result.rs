use jolt_field::Field;

use crate::{BackendValueSlot, SumcheckSlot};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProofOutput<Proof> {
    pub slot: SumcheckSlot,
    pub proof: Proof,
}

impl<Proof> SumcheckProofOutput<Proof> {
    pub const fn new(slot: SumcheckSlot, proof: Proof) -> Self {
        Self { slot, proof }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckEvaluationOutput<F: Field> {
    pub slot: BackendValueSlot,
    pub value: F,
}

impl<F: Field> SumcheckEvaluationOutput<F> {
    pub const fn new(slot: BackendValueSlot, value: F) -> Self {
        Self { slot, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckResult<F: Field, Proof> {
    pub proofs: Vec<SumcheckProofOutput<Proof>>,
    pub evaluations: Vec<SumcheckEvaluationOutput<F>>,
}

impl<F: Field, Proof> SumcheckResult<F, Proof> {
    pub const fn new(
        proofs: Vec<SumcheckProofOutput<Proof>>,
        evaluations: Vec<SumcheckEvaluationOutput<F>>,
    ) -> Self {
        Self {
            proofs,
            evaluations,
        }
    }
}
