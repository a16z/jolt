//! Typed outputs produced by stage 6a verification.

use jolt_field::Field;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedStage6AddressPhaseSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6AddressPhasePublicOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
}
