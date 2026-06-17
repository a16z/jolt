#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineOpeningId;
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_openings::{BatchOpeningStatement, VerifierOpeningClaim};
use jolt_poly::{Point, HIGH_TO_LOW};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8OpeningId {
    Jolt(JoltOpeningId),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineOpeningId),
}

impl From<JoltOpeningId> for Stage8OpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlineOpeningId> for Stage8OpeningId {
    fn from(id: FieldInlineOpeningId) -> Self {
        Self::FieldInline(id)
    }
}

pub type Stage8OpeningStatement<F, C, Claim> =
    BatchOpeningStatement<F, C, Stage8OpeningId, Stage8OpeningId, Claim>;

#[derive(Clone, Debug)]
pub struct Stage8ClearBatchStatement<F: Field, C> {
    pub opening_ids: Vec<Stage8OpeningId>,
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub statement: Stage8OpeningStatement<F, C, F>,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkBatchStatement<F: Field, C> {
    pub opening_ids: Vec<Stage8OpeningId>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub statement: Stage8OpeningStatement<F, C, ()>,
}

#[derive(Clone, Debug)]
pub enum Stage8BatchStatement<F: Field, C> {
    Clear(Stage8ClearBatchStatement<F, C>),
    Zk(Stage8ZkBatchStatement<F, C>),
}

#[derive(Clone, Debug)]
pub struct Stage8ClearOutput<F: Field, C> {
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub opening_ids: Vec<Stage8OpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
    pub joint_commitment: C,
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
