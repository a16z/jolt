use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::Field;
#[cfg(not(feature = "akita"))]
use jolt_openings::VerifierOpeningClaim;
use jolt_poly::{Point, HIGH_TO_LOW};

#[cfg(not(feature = "akita"))]
#[derive(Clone, Debug)]
pub struct Stage8ClearOutput<F: Field, C> {
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
    pub joint_commitment: C,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkOutput<F: Field, C, H> {
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
}

#[derive(Clone, Debug)]
pub enum Stage8Output<F: Field, C, H> {
    #[cfg(not(feature = "akita"))]
    Clear(Stage8ClearOutput<F, C>),
    /// The akita build's clear stage 8 verifies to completion inside
    /// [`super::verify`] (native W_jolt batch + auxiliary packed openings),
    /// so no per-opening payload survives it.
    #[cfg(feature = "akita")]
    Clear,
    Zk(Stage8ZkOutput<F, C, H>),
}

impl<F: Field, C, H> Stage8Output<F, C, H> {
    pub fn zk(&self) -> Result<&Stage8ZkOutput<F, C, H>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            #[cfg(not(feature = "akita"))]
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage8" }),
            #[cfg(feature = "akita")]
            Self::Clear => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage8" }),
        }
    }
}
