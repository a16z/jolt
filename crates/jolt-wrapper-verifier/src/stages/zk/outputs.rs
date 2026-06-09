use jolt_blindfold::BlindFoldProtocol;
use jolt_field::Field;
use jolt_sumcheck::CommittedOutputClaims;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimShape {
    pub output_claim_count: usize,
    pub row_count: usize,
    pub row_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimOutput<C> {
    pub shape: CommittedOutputClaimShape,
    pub commitments: CommittedOutputClaims<C>,
}

#[derive(Clone, Debug)]
pub struct BlindFoldOutput<F: Field, C> {
    pub protocol: BlindFoldProtocol<F, C>,
}
