//! Typed outputs produced by stage 5 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::{GetPoint, OpeningClaim};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage5OutputClaims;

/// The Fiat-Shamir challenges the verifier draws during stage 5: the instruction
/// and RAM batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5Challenges<F: Field> {
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ClearOutput<F: Field> {
    pub challenges: Stage5Challenges<F>,
    /// The produced stage-5 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell. Later stages read each opening's value and point
    /// directly off these opening claims.
    pub output_claims: Stage5OutputClaims<OpeningClaim<F>>,
    /// The instruction read-RAF address point, materialized contiguously from the
    /// virtual-RA opening cells (which tile it as `chunk ++ r_cycle`). Stored
    /// because stage 6 re-chunks it by the committed-chunk width — a different
    /// split than the virtual-RA cells carry — so it needs a contiguous copy that
    /// downstream code can borrow.
    pub instruction_r_address: Vec<F>,
}

impl<F: Field> Stage5ClearOutput<F> {
    /// The instruction read-RAF cycle point, shared by the lookup-table-flag and
    /// RAF-flag openings.
    pub fn instruction_r_cycle(&self) -> &[F] {
        self.output_claims.instruction_read_raf.r_cycle()
    }

    /// The reduced RAM-RA opening point (`address ++ cycle`, `log_k + log_t` vars).
    pub fn ram_reduced_opening_point(&self) -> &[F] {
        self.output_claims.ram_ra_claim_reduction.ram_ra.point()
    }

    /// The register value-evaluation opening point (`REGISTER_ADDRESS_BITS + log_t`
    /// vars), shared by the `rd_inc` and `rd_wa` openings.
    pub fn registers_opening_point(&self) -> &[F] {
        self.output_claims.registers_val_evaluation.rd_inc.point()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ZkOutput<F: Field, C> {
    pub challenges: Stage5Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub instruction_r_address: Vec<F>,
    pub instruction_r_cycle: Vec<F>,
    pub ram_reduced_opening_point: Vec<F>,
    pub registers_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5Output<F: Field, C> {
    Clear(Stage5ClearOutput<F>),
    Zk(Stage5ZkOutput<F, C>),
}
