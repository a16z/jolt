use jolt_backends::{CommitmentRequest, CommitmentRequestItem, CommitmentSlot};
use jolt_witness::{
    CommittedWitnessProvider, MaterializationPolicy, OracleRef, RetentionHint, ViewRequirement,
    WitnessNamespace,
};

use crate::ProverError;

pub fn build_commitment_request<F, N, W>(witness: &W) -> Result<CommitmentRequest<N>, ProverError>
where
    N: WitnessNamespace,
    W: CommittedWitnessProvider<F, N>,
{
    let mut items = Vec::new();
    for (index, committed) in witness.committed_oracle_order()?.into_iter().enumerate() {
        let oracle = OracleRef::committed(committed);
        let descriptor = witness.describe_oracle(oracle)?;
        let retention = witness
            .view_requirements(oracle)?
            .first()
            .map_or(RetentionHint::ThroughStage8, |requirement| {
                requirement.retention
            });

        items.push(CommitmentRequestItem::new(
            CommitmentSlot(index as u32),
            ViewRequirement::new(
                oracle,
                descriptor.encoding,
                MaterializationPolicy::Streaming,
                retention,
            ),
        ));
    }

    Ok(CommitmentRequest::new(items))
}
