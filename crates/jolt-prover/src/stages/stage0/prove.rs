use jolt_backends::{CommitmentBackend, CommitmentResult};
use jolt_openings::CommitmentScheme;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, CommittedWitnessProvider, WitnessNamespace,
};

use crate::ProverError;

#[cfg(feature = "field-inline")]
use super::output::FieldInlineCommittedPolynomialOutput;
use super::{
    input::CommitmentStageConfig, output::CommitmentStageOutput, request::build_commitment_request,
};

pub fn commit_witness<F, N, W, B, PCS>(
    witness: &W,
    backend: &mut B,
    setup: &PCS::ProverSetup,
) -> Result<CommitmentResult<N, PCS>, ProverError>
where
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, N>,
    B: CommitmentBackend<F, N, PCS>,
{
    let request = build_commitment_request::<F, N, W>(witness)?;
    Ok(backend.commit(&request, witness, setup)?)
}

#[cfg(not(feature = "field-inline"))]
pub fn prove_jolt_vm_commitments<F, W, B, PCS>(
    witness: &W,
    backend: &mut B,
    setup: &PCS::ProverSetup,
    config: CommitmentStageConfig,
) -> Result<CommitmentStageOutput<PCS>, ProverError>
where
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, JoltVmNamespace>,
    B: CommitmentBackend<F, JoltVmNamespace, PCS>,
{
    let result = commit_witness::<F, JoltVmNamespace, W, B, PCS>(witness, backend, setup)?;
    CommitmentStageOutput::from_backend_result(result, config)
}

#[cfg(feature = "field-inline")]
pub fn prove_jolt_vm_commitments<F, W, B, PCS>(
    witness: &W,
    backend: &mut B,
    setup: &PCS::ProverSetup,
    config: CommitmentStageConfig,
    field_inline_outputs: Vec<FieldInlineCommittedPolynomialOutput<PCS>>,
) -> Result<CommitmentStageOutput<PCS>, ProverError>
where
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, JoltVmNamespace>,
    B: CommitmentBackend<F, JoltVmNamespace, PCS>,
{
    let result = commit_witness::<F, JoltVmNamespace, W, B, PCS>(witness, backend, setup)?;
    CommitmentStageOutput::from_backend_result(result, config, field_inline_outputs)
}
