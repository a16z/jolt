use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};

/// The stage-8 pieces BlindFold consumes: which openings were batched, each
/// entry's `gamma · scale` constraint coefficient, and the joint opening's
/// point, combined commitment, and hiding evaluation commitment.
#[derive(Clone, Debug)]
pub struct Stage8ZkOutput<F: Field, C, H> {
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
}

/// Stage 8 is terminal: the clear arm fully discharges the final opening
/// inside `verify` and carries nothing downstream; only the ZK arm produces
/// data (for BlindFold).
#[derive(Clone, Debug)]
pub enum Stage8Output<F: Field, C, H> {
    Clear,
    Zk(Stage8ZkOutput<F, C, H>),
}

impl<F: Field, C, H> Stage8Output<F, C, H> {
    pub fn zk(&self) -> Result<&Stage8ZkOutput<F, C, H>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage8" }),
        }
    }
}
