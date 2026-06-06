use jolt_backends::{
    CommitmentBackend, CommitmentMode, CommitmentResult, TracePolynomialEmbedding,
};
use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_openings::CommitmentScheme;
use jolt_verifier::config::ZkConfig;
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, CommittedWitnessProvider, WitnessNamespace,
};

use crate::ProverError;

#[cfg(feature = "frontier-harness")]
fn timed_stage0<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage0<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(feature = "field-inline")]
use super::input::FieldInlineCommitmentWitness;
use super::{
    input::CommitmentStageInput, output::CommitmentStageOutput, request::build_commitment_request,
};

pub fn prove<F, W, B, PCS>(
    input: CommitmentStageInput<'_, W, PCS>,
    backend: &mut B,
) -> Result<CommitmentStageOutput<PCS>, ProverError>
where
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, JoltVmNamespace> + Sync,
    B: CommitmentStageBackend<F, PCS>,
{
    let mode = match input.protocol.zk {
        ZkConfig::Transparent => CommitmentMode::Transparent,
        ZkConfig::BlindFold => CommitmentMode::Zk,
    };
    let trace_embedding = final_opening_trace_embedding(&input.config)?;
    let result = timed_stage0("stage0.commit_jolt", || {
        commit::<F, JoltVmNamespace, W, B, PCS>(
            input.witness,
            backend,
            input.setup,
            mode,
            input.config.trace_polynomial_order,
            trace_embedding,
        )
    })?;
    #[cfg(feature = "field-inline")]
    let field_inline_result = timed_stage0("stage0.commit_field_inline", || {
        backend.commit_field_inline(
            input.field_inline_witness,
            input.setup,
            mode,
            input.config.trace_polynomial_order,
            trace_embedding,
        )
    })?;

    CommitmentStageOutput::from_backend_result(
        result,
        #[cfg(feature = "field-inline")]
        field_inline_result,
        input.config,
    )
}

pub trait CommitmentStageBackend<F, PCS>: CommitmentBackend<F, JoltVmNamespace, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    #[cfg(feature = "field-inline")]
    fn commit_field_inline(
        &mut self,
        witness: &FieldInlineCommitmentWitness<'_, F>,
        setup: &PCS::ProverSetup,
        mode: CommitmentMode,
        trace_polynomial_order: TracePolynomialOrder,
        trace_embedding: Option<TracePolynomialEmbedding>,
    ) -> Result<CommitmentResult<FieldInlineNamespace, PCS>, ProverError>;
}

#[cfg(not(feature = "field-inline"))]
impl<F, PCS, B> CommitmentStageBackend<F, PCS> for B
where
    PCS: CommitmentScheme<Field = F>,
    B: CommitmentBackend<F, JoltVmNamespace, PCS>,
{
}

#[cfg(feature = "field-inline")]
impl<F, PCS, B> CommitmentStageBackend<F, PCS> for B
where
    PCS: CommitmentScheme<Field = F>,
    B: CommitmentBackend<F, JoltVmNamespace, PCS> + CommitmentBackend<F, FieldInlineNamespace, PCS>,
{
    fn commit_field_inline(
        &mut self,
        witness: &FieldInlineCommitmentWitness<'_, F>,
        setup: &PCS::ProverSetup,
        mode: CommitmentMode,
        trace_polynomial_order: TracePolynomialOrder,
        trace_embedding: Option<TracePolynomialEmbedding>,
    ) -> Result<CommitmentResult<FieldInlineNamespace, PCS>, ProverError> {
        commit::<F, FieldInlineNamespace, FieldInlineCommitmentWitness<'_, F>, Self, PCS>(
            witness,
            self,
            setup,
            mode,
            trace_polynomial_order,
            trace_embedding,
        )
    }
}

pub(super) fn commit<F, N, W, B, PCS>(
    witness: &W,
    backend: &mut B,
    setup: &PCS::ProverSetup,
    mode: CommitmentMode,
    trace_polynomial_order: TracePolynomialOrder,
    trace_embedding: Option<TracePolynomialEmbedding>,
) -> Result<CommitmentResult<N, PCS>, ProverError>
where
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, N> + Sync + ?Sized,
    B: CommitmentBackend<F, N, PCS>,
{
    let request = timed_stage0("stage0.commit.build_request", || {
        build_commitment_request::<F, N, W>(witness, mode, trace_polynomial_order, trace_embedding)
    })?;
    timed_stage0("stage0.commit.backend", || {
        Ok(backend.commit(&request, witness, setup)?)
    })
}

fn final_opening_trace_embedding(
    config: &super::input::CommitmentStageConfig,
) -> Result<Option<TracePolynomialEmbedding>, ProverError> {
    match (
        config.final_opening_trace_rows,
        config.final_opening_address_columns,
    ) {
        (Some(trace_rows), Some(address_columns)) => Ok(Some(TracePolynomialEmbedding::new(
            trace_rows,
            address_columns,
            config.trace_polynomial_order,
        ))),
        (None, None) => Ok(None),
        _ => Err(ProverError::InvalidStageRequest {
            reason:
                "Stage 0 final-opening trace embedding requires both trace rows and address columns"
                    .to_owned(),
        }),
    }
}
