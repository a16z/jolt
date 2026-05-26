use jolt_field::Field;

use crate::{BackendValueSlot, OpeningSlot};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningProofOutput<Proof> {
    pub proof: Proof,
}

impl<Proof> OpeningProofOutput<Proof> {
    pub const fn new(proof: Proof) -> Self {
        Self { proof }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningEvaluationOutput<F: Field> {
    pub slot: BackendValueSlot,
    pub query: OpeningSlot,
    pub value: F,
}

impl<F: Field> OpeningEvaluationOutput<F> {
    pub const fn new(slot: BackendValueSlot, query: OpeningSlot, value: F) -> Self {
        Self { slot, query, value }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningResult<F: Field, Proof> {
    pub proof: OpeningProofOutput<Proof>,
    pub joint_claim: F,
    pub evaluations: Vec<OpeningEvaluationOutput<F>>,
}

impl<F: Field, Proof> OpeningResult<F, Proof> {
    pub const fn new(
        proof: OpeningProofOutput<Proof>,
        joint_claim: F,
        evaluations: Vec<OpeningEvaluationOutput<F>>,
    ) -> Self {
        Self {
            proof,
            joint_claim,
            evaluations,
        }
    }
}
