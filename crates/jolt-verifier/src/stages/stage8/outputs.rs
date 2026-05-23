use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_openings::VerifierOpeningClaim;
use jolt_poly::{Point, HIGH_TO_LOW};

#[derive(Clone, Debug)]
pub struct Stage8ClearOutput<F: Field, C> {
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub opening_point: Point<HIGH_TO_LOW, F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
    pub joint_commitment: C,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkOutput<F: Field, C, H> {
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub opening_point: Point<HIGH_TO_LOW, F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
}

#[derive(Clone, Debug)]
pub enum Stage8Output<F: Field, C, H> {
    Clear(Stage8ClearOutput<F, C>),
    Zk(Stage8ZkOutput<F, C, H>),
}
