use jolt_sumcheck::CommittedOutputClaims;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimShape {
    pub output_claim_count: usize,
    pub row_len: usize,
}

impl CommittedOutputClaimShape {
    /// Number of output-claim row commitments the committed sumcheck sends.
    pub fn row_count(&self) -> usize {
        self.output_claim_count.div_ceil(self.row_len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimOutput<C> {
    pub shape: CommittedOutputClaimShape,
    pub commitments: CommittedOutputClaims<C>,
}
