use crate::{
    field_inline::FieldIncrementColumn, trace_column::DenseTraceColumnPoly, CommitmentGrid,
};
use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_verifier::VerifierError;

use crate::akita::commitment::FieldIncObject;
use crate::KernelError;

fn commit_failed<F: JoltField>(reason: impl ToString) -> KernelError<F> {
    KernelError::Verifier(VerifierError::FinalOpeningVerificationFailed {
        reason: reason.to_string(),
    })
}

/// Commit `FieldRdInc` directly as field elements, padding with zeros to the
/// minimum dense-object arity. An identically zero table still commits because
/// the dense schedule depends on its shape, not its contents.
pub fn commit_field_inc<F, PCS>(
    setup: &PCS::ProverSetup,
    log_t: usize,
    column: FieldIncrementColumn<F>,
) -> Result<FieldIncObject<PCS>, KernelError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
{
    let layout = FieldIncLayout::new(log_t);
    let polynomial = DenseTraceColumnPoly::new(
        column,
        CommitmentGrid {
            total_vars: layout.num_vars(),
            log_t,
            log_k_chunk: 0,
            order: TracePolynomialOrder::CycleMajor,
        },
    )
    .ok_or_else(|| commit_failed("FieldRdInc disagrees with the Akita trace arity"))?;
    let (commitment, hint) = tracing::info_span!("commit_field_inc", num_vars = layout.num_vars())
        .in_scope(|| PCS::commit_full_width_object(setup, &polynomial, layout.layout_digest()))
        .map_err(commit_failed)?;
    Ok(FieldIncObject { commitment, hint })
}
