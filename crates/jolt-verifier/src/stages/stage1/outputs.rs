//! Typed outputs produced by stage 1 verification.

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial;
use jolt_claims::protocols::jolt::JoltVirtualPolynomial;
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};

use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

#[cfg(feature = "field-inline")]
use super::inputs::{field_inline_stage1_claims_from_r1cs_inputs, FieldInlineStage1Claims};
use super::inputs::{spartan_outer_claims_from_r1cs_inputs, SpartanOuterClaims};

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
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage1Claims<F>,
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
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}

#[cfg(not(feature = "field-inline"))]
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors Stage1ClearOutput, which decomposes the uni-skip/remainder reductions into distinct Fiat-Shamir values."
)]
pub fn stage1_clear_output<F: Field>(
    tau: Vec<F>,
    uniskip_challenge: F,
    uniskip_output_claim: F,
    remainder_batching_coefficient: F,
    remainder_challenges: Vec<F>,
    remainder_output_claim: F,
    expected_remainder_output_claim: F,
    r1cs_input_claims: impl IntoIterator<Item = (JoltVirtualPolynomial, F)>,
) -> Result<Stage1ClearOutput<F>, VerifierError> {
    let uniskip = stage1_uniskip_output(uniskip_challenge, uniskip_output_claim);
    let remainder = stage1_remainder_output(
        uniskip_output_claim,
        remainder_batching_coefficient,
        remainder_challenges,
        remainder_output_claim,
        expected_remainder_output_claim,
    );
    let public = stage1_public_output(
        tau,
        uniskip_challenge,
        remainder_batching_coefficient,
        &remainder,
    );
    Ok(Stage1ClearOutput {
        public,
        uniskip,
        remainder,
        outer: spartan_outer_claims_from_r1cs_inputs(r1cs_input_claims)?,
    })
}

#[cfg(feature = "field-inline")]
#[expect(
    clippy::too_many_arguments,
    reason = "Mirrors Stage1ClearOutput, which decomposes the uni-skip/remainder reductions into distinct Fiat-Shamir values."
)]
pub fn stage1_clear_output<F: Field>(
    tau: Vec<F>,
    uniskip_challenge: F,
    uniskip_output_claim: F,
    remainder_batching_coefficient: F,
    remainder_challenges: Vec<F>,
    remainder_output_claim: F,
    expected_remainder_output_claim: F,
    r1cs_input_claims: impl IntoIterator<Item = (JoltVirtualPolynomial, F)>,
    field_inline_r1cs_input_claims: impl IntoIterator<Item = (FieldInlineVirtualPolynomial, F)>,
) -> Result<Stage1ClearOutput<F>, VerifierError> {
    let uniskip = stage1_uniskip_output(uniskip_challenge, uniskip_output_claim);
    let remainder = stage1_remainder_output(
        uniskip_output_claim,
        remainder_batching_coefficient,
        remainder_challenges,
        remainder_output_claim,
        expected_remainder_output_claim,
    );
    let public = stage1_public_output(
        tau,
        uniskip_challenge,
        remainder_batching_coefficient,
        &remainder,
    );
    Ok(Stage1ClearOutput {
        public,
        uniskip,
        remainder,
        outer: spartan_outer_claims_from_r1cs_inputs(r1cs_input_claims)?,
        field_inline: field_inline_stage1_claims_from_r1cs_inputs(field_inline_r1cs_input_claims)?,
    })
}

fn stage1_uniskip_output<F: Field>(
    uniskip_challenge: F,
    uniskip_output_claim: F,
) -> VerifiedSpartanOuterSumcheck<F> {
    VerifiedSpartanOuterSumcheck {
        input_claim: F::from_u64(0),
        sumcheck_point: Point::high_to_low(vec![uniskip_challenge]),
        sumcheck_final_claim: uniskip_output_claim,
        expected_output_claim: uniskip_output_claim,
    }
}

fn stage1_remainder_output<F: Field>(
    uniskip_output_claim: F,
    remainder_batching_coefficient: F,
    remainder_challenges: Vec<F>,
    remainder_output_claim: F,
    expected_remainder_output_claim: F,
) -> VerifiedSpartanOuterSumcheck<F> {
    VerifiedSpartanOuterSumcheck {
        input_claim: uniskip_output_claim * remainder_batching_coefficient,
        sumcheck_point: Point::high_to_low(remainder_challenges),
        sumcheck_final_claim: remainder_output_claim,
        expected_output_claim: expected_remainder_output_claim,
    }
}

fn stage1_public_output<F: Field>(
    tau: Vec<F>,
    uniskip_challenge: F,
    remainder_batching_coefficient: F,
    remainder: &VerifiedSpartanOuterSumcheck<F>,
) -> Stage1PublicOutput<F> {
    Stage1PublicOutput {
        tau,
        uniskip_challenge,
        remainder_batching_coefficient,
        remainder_challenges: remainder.sumcheck_point.as_slice().to_vec(),
    }
}
