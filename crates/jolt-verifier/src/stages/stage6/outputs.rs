//! Typed outputs produced by stage 6 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::inputs::Stage6OutputClaims;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PublicOutput<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub inc_gamma: F,
    /// Committed program mode only: bytecode claim-reduction batching
    /// challenge (core's `eta`).
    pub bytecode_reduction_eta: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ClearOutput<F: Field> {
    pub public: Stage6PublicOutput<F>,
    /// The produced opening *values* (wire form); read by later stages and the
    /// Fiat-Shamir opening-claim encoder.
    pub output_claims: Stage6OutputClaims<F>,
    /// The produced opening *points* (point-only cell), paired field-for-field with
    /// `output_claims`. Stages 7 and 8 read each relation's opening point off these
    /// cells (via the `Stage6OutputClaims<Vec<F>>` accessors).
    pub output_points: Stage6OutputClaims<Vec<F>>,
    /// Committed-program mode only: the bytecode claim-reduction's per-chunk
    /// weights (`r_bc`, chunk weights, gamma-folded lane weights). These are
    /// public derived data (not openings), so stage 7's bytecode address phase
    /// reads them here rather than recomputing them.
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ZkOutput<F: Field, C> {
    pub public: Stage6PublicOutput<F>,
    pub address_phase_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub address_phase_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening *points* (point-only cell), the ZK counterpart of the
    /// clear path's `Stage6ClearOutput::output_points`. Stages 7/8 and BlindFold
    /// read each relation's opening point off these cells through the same
    /// `Stage6OutputClaims<Vec<F>>` accessors. (BlindFold recomputes the bytecode
    /// reduction weights locally, so the ZK output carries no weights aux.)
    pub output_points: Stage6OutputClaims<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6Output<F: Field, C> {
    Clear(Stage6ClearOutput<F>),
    Zk(Stage6ZkOutput<F, C>),
}

impl<F: Field, C> Stage6Output<F, C> {
    pub fn clear(&self) -> Result<&Stage6ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage6" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage6ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage6" }),
        }
    }
}

/// Public bytecode claim-reduction state shared by the cycle and address
/// phases: the per-chunk weights over dropped address bits, the chunk-local
/// cycle point, and the gamma-folded lane weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReductionWeights<F: Field> {
    pub r_bc: Vec<F>,
    pub chunk_rbc_weights: Vec<F>,
    pub lane_weights: Vec<F>,
}
