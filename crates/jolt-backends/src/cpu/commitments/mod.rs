mod stream;

use crate::{
    BackendError, CommitmentBackend, CommitmentRequest, CommitmentResult,
    ResolvedWitnessRequirement,
};
use jolt_openings::StreamingCommitment;
use jolt_witness::{
    MaterializationPolicy, OracleKind, WitnessError, WitnessNamespace, WitnessProvider,
};

use super::CpuBackend;

impl<F, N, PCS> CommitmentBackend<F, N, PCS> for CpuBackend
where
    F: jolt_field::Field,
    N: WitnessNamespace,
    PCS: StreamingCommitment<Field = F>,
{
    fn commit<W>(
        &mut self,
        request: &CommitmentRequest<N>,
        witness: &W,
        setup: &PCS::ProverSetup,
    ) -> Result<CommitmentResult<N, PCS>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        let mut resolved_witness = Vec::with_capacity(request.items.len());
        let mut streamed_witness = Vec::new();
        let mut commitments = Vec::new();

        for item in &request.items {
            let descriptor = witness.describe_oracle(item.requirement.oracle)?;
            if descriptor.encoding != item.requirement.encoding {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: format!(
                        "request asked for {:?} encoding, provider exposes {:?}",
                        item.requirement.encoding, descriptor.encoding,
                    ),
                }
                .into());
            }
            resolved_witness.push(ResolvedWitnessRequirement::new(
                item.slot,
                item.requirement,
                descriptor,
            ));

            if item.requirement.materialization != MaterializationPolicy::Streaming {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "CPU commitment backend currently requires streaming materialization"
                        .to_owned(),
                }
                .into());
            }
            let OracleKind::Committed(id) = item.requirement.oracle.kind else {
                return Err(WitnessError::InvalidWitnessData {
                    namespace: N::ID.name,
                    reason: "commitment requests require committed oracles".to_owned(),
                }
                .into());
            };

            let mut stream = witness.committed_stream(id, self.config.commitment_chunk_size)?;
            let committed = stream::commit_streamed_witness::<F, PCS, N>(
                item.slot,
                item.requirement.oracle,
                descriptor.dimensions.rows,
                stream.as_mut(),
                setup,
            )?;
            streamed_witness.push(committed.streamed);
            commitments.push(committed.output);
        }

        Ok(CommitmentResult::new(
            resolved_witness,
            streamed_witness,
            commitments,
        ))
    }
}
