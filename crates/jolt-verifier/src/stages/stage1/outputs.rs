//! Typed outputs produced by stage 1 verification.

use jolt_field::Field;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::SpartanOuterClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1PublicOutput<F: Field> {
    pub tau: Vec<F>,
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
    pub uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub remainder_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub remainder_output_claims: CommittedOutputClaimOutput<C>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage1Output<F: Field, C> {
    Clear(Stage1ClearOutput<F>),
    Zk(Stage1ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSpartanOuterSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}
