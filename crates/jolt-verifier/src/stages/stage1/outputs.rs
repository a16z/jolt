//! Typed outputs produced by stage 1 verification.

use jolt_claims::protocols::jolt::{
    formulas::spartan::SpartanOuterDimensions, JoltVirtualPolynomial,
};
use jolt_field::Field;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};

use super::inputs::SpartanOuterClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1PublicOutput<F: Field> {
    pub uniskip_challenge: F,
    pub remainder_batching_coefficient: F,
    pub remainder_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ClearOutput<F: Field> {
    pub public: Stage1PublicOutput<F>,
    pub uniskip: VerifiedSpartanOuterSumcheck<F>,
    pub remainder: VerifiedSpartanOuterSumcheck<F>,
    pub outer: SpartanOuterClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ZkOutput<F: Field, C> {
    pub public: Stage1PublicOutput<F>,
    pub uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub remainder_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub outer: SpartanOuterClaimSlots,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage1Output<F: Field, C> {
    Clear(Stage1ClearOutput<F>),
    Zk(Stage1ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSpartanOuterSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOuterClaimSlots {
    pub r1cs_inputs: Vec<JoltVirtualPolynomial>,
}

impl SpartanOuterClaimSlots {
    pub fn from_dimensions(dimensions: &SpartanOuterDimensions) -> Self {
        Self {
            r1cs_inputs: dimensions.variables().to_vec(),
        }
    }
}
