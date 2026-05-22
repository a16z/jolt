//! Typed outputs produced by stage 2 verification.

use jolt_field::Field;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage2BatchOutputOpeningClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PublicOutput<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub product_uniskip_challenge: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ClearOutput<F: Field> {
    pub public: Stage2PublicOutput<F>,
    pub output_claims: Stage2BatchOutputOpeningClaims<F>,
    pub product_uniskip: VerifiedProductUniSkip<F>,
    pub batch: VerifiedStage2Batch<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ZkOutput<F: Field, C> {
    pub public: Stage2PublicOutput<F>,
    pub product_uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub product_uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub ram_val_check_inputs: Stage2RamValCheckInputs<F>,
    pub ram_ra_claim_reduction_inputs: Stage2RamRaClaimReductionInputs<F>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage2ZkOutput<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamValCheckInputs<F: Field> {
    pub ram_read_write_opening_point: Vec<F>,
    pub ram_output_check_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamRaClaimReductionInputs<F: Field> {
    pub ram_raf_evaluation_opening_point: Vec<F>,
    pub ram_read_write_opening_point: Vec<F>,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInlineStage2ZkOutput<F: Field> {
    pub field_registers_claim_reduction_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Output<F: Field, C> {
    Clear(Stage2ClearOutput<F>),
    Zk(Stage2ZkOutput<F, C>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedProductUniSkip<F: Field> {
    pub tau_low: Vec<F>,
    pub tau_high: F,
    pub input_claim: F,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage2Batch<F: Field> {
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction_gamma: F,
    pub output_address_challenges: Vec<F>,
    pub ram_read_write: VerifiedStage2Sumcheck<F>,
    pub product_remainder: VerifiedStage2Sumcheck<F>,
    pub instruction_claim_reduction: VerifiedStage2Sumcheck<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction: VerifiedStage2Sumcheck<F>,
    pub ram_raf_evaluation: VerifiedStage2Sumcheck<F>,
    pub ram_output_check: VerifiedStage2Sumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage2Sumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}
