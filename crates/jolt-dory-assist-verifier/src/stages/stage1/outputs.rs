//! Typed outputs produced by stage 1 verification.

use jolt_claims::protocols::dory_assist::{DoryAssistChallengeId, DoryAssistRelationId};
use jolt_field::Fq;
use jolt_poly::{Point, HIGH_TO_LOW};

use crate::proof::DoryAssistOpeningClaim;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Output {
    pub relation_count: u32,
    pub relation_outputs: Vec<Stage1RelationOutput>,
    pub challenge: Fq,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1RelationOutput {
    pub id: DoryAssistRelationId,
    pub relation_challenges: Vec<DoryAssistChallengeValue>,
    pub input_claim: Fq,
    pub sumcheck_point: Point<HIGH_TO_LOW, Fq>,
    pub sumcheck_final_claim: Fq,
    pub expected_output_claim: Fq,
    pub opening_claims: Vec<DoryAssistOpeningClaim>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryAssistChallengeValue {
    pub id: DoryAssistChallengeId,
    pub value: Fq,
}
