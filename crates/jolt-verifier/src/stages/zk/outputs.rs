use jolt_field::JoltField;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedOutputClaims};

use crate::stages::relations::CommittedClaimLayout;
#[cfg(doc)]
use crate::verifier::CheckedInputs;

/// The row layout of a committed sumcheck's hidden output claims: the
/// committed claim cells in row order, packed `row_len` to a row, and the
/// aliased cells that read their source rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimShape {
    pub layout: CommittedClaimLayout,
    pub row_len: usize,
}

impl CommittedOutputClaimShape {
    /// `layout` packed `row_len` cells to a row
    /// ([`CheckedInputs::committed_row_len`]).
    pub fn new(row_len: usize, layout: CommittedClaimLayout) -> Self {
        Self { layout, row_len }
    }

    /// Number of output-claim row commitments the committed sumcheck sends.
    pub fn row_count(&self) -> usize {
        self.layout.ids.len().div_ceil(self.row_len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimOutput<C> {
    pub shape: CommittedOutputClaimShape,
    pub commitments: CommittedOutputClaims<C>,
}

/// A committed stage batch as the ZK verifier reads it: the round
/// consistency, the output points derived at the batch point, and the
/// output-claim row commitments over those points' committed cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCommittedBatch<F: JoltField, C, P> {
    pub consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub output_points: P,
    pub output_claims: CommittedOutputClaimOutput<C>,
}
