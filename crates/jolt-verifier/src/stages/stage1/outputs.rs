//! Typed outputs produced by stage 1 verification.

use jolt_field::Field;

use super::inputs::SpartanOuterClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Output<F: Field> {
    pub uniskip_challenge: F,
    pub remainder_batching_coefficient: F,
    pub remainder_challenges: Vec<F>,
    pub uniskip: VerifiedSpartanOuterSumcheck<F>,
    pub remainder: VerifiedSpartanOuterSumcheck<F>,
    pub outer: SpartanOuterClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSpartanOuterSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}
