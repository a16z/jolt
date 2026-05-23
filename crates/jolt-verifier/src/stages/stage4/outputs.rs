//! Typed outputs produced by stage 4 verification.

use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage4Claims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub registers_gamma: F,
    pub ram_val_check_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ClearOutput<F: Field> {
    pub public: Stage4PublicOutput<F>,
    pub output_claims: Stage4Claims<F>,
    pub batch: VerifiedStage4Batch<F>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ZkOutput<F: Field, C> {
    pub public: Stage4PublicOutput<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub ram_val_check_public_eval: F,
    pub registers_read_write_opening_point: Vec<F>,
    pub ram_val_check_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Output<F: Field, C> {
    Clear(Stage4ClearOutput<F>),
    Zk(Stage4ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage4Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub registers_read_write: VerifiedStage4Sumcheck<F>,
    pub ram_val_check: VerifiedStage4Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage4Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    pub advice_contributions: Vec<VerifiedRamValCheckAdviceContribution<F>>,
    pub full_eval: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedRamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}
