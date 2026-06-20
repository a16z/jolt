//! Typed outputs produced by stage 5 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::OpeningClaim;
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
        self.output_claims.instruction_r_cycle()
    }

    /// The reduced RAM-RA opening point (`address ++ cycle`, `log_k + log_t` vars).
    pub fn ram_reduced_opening_point(&self) -> &[F] {
        self.output_claims.ram_reduced_opening_point()
    }

    /// The register value-evaluation opening point (`REGISTER_ADDRESS_BITS + log_t`
    /// vars), shared by the `rd_inc` and `rd_wa` openings.
    pub fn registers_opening_point(&self) -> &[F] {
        self.output_claims.registers_opening_point()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ZkOutput<F: Field, C> {
    pub challenges: Stage5Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening points (point-only cell), the ZK counterpart of the
    /// clear path's `output_claims`. Read through the same `*_point()` accessors.
    pub output_points: Stage5OutputClaims<Vec<F>>,
    /// The contiguous instruction address point, stored (rather than reconstructed
    /// from `output_points` on demand) so stage 6 can borrow it — the per-chunk
    /// virtual-RA cells don't hold it contiguously. Mirrors `Stage5ClearOutput`.
    pub instruction_r_address: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5Output<F: Field, C> {
    Clear(Stage5ClearOutput<F>),
    Zk(Stage5ZkOutput<F, C>),
}

impl<F: Field, C> Stage5Output<F, C> {
    pub fn clear(&self) -> Result<&Stage5ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage5" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage5ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage5" }),
        }
    }
}
