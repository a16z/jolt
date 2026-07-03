use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_openings::VerifierOpeningClaim;
use jolt_poly::{Point, HIGH_TO_LOW};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8OpeningId {
    Jolt(JoltOpeningId),
}

impl From<JoltOpeningId> for Stage8OpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

#[derive(Clone, Debug)]
pub struct Stage8ClearOutput<F: Field, C> {
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub opening_ids: Vec<Stage8OpeningId>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkOutput<F: Field, C, H> {
    pub opening_ids: Vec<Stage8OpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
}

#[derive(Clone, Debug)]
pub enum Stage8Output<F: Field, C, H> {
    Clear(Stage8ClearOutput<F, C>),
    Zk(Stage8ZkOutput<F, C, H>),
}
