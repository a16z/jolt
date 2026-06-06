use jolt_backends::{
    CommitmentMode, CommitmentRequest, CommitmentRequestItem, CommitmentSlot,
    TracePolynomialEmbedding,
};
use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_witness::{
    CommittedWitnessProvider, MaterializationPolicy, OracleRef, RetentionHint, ViewRequirement,
    WitnessNamespace,
};

use crate::ProverError;

pub fn build_commitment_request<F, N, W>(
    witness: &W,
    mode: CommitmentMode,
    trace_polynomial_order: TracePolynomialOrder,
    trace_embedding: Option<TracePolynomialEmbedding>,
) -> Result<CommitmentRequest<N>, ProverError>
where
    N: WitnessNamespace,
    W: CommittedWitnessProvider<F, N> + ?Sized,
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

        items.push(
            CommitmentRequestItem::with_mode(
                CommitmentSlot(index as u32),
                ViewRequirement::new(
                    oracle,
                    descriptor.encoding,
                    MaterializationPolicy::Streaming,
                    retention,
                ),
                mode,
            )
            .with_trace_polynomial_order(trace_polynomial_order)
            .with_trace_embedding(trace_embedding),
        );
    }

    Ok(CommitmentRequest::new(items))
}
