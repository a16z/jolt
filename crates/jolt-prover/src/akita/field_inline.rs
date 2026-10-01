//! Akita commitment of the field-inline register increment polynomial.

use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_field::JoltField;
use jolt_kernels::{
    field_inline::FieldIncrementColumn, optimized::opening::DenseTraceColumnPoly, CommitmentGrid,
};
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_verifier::VerifierError;

use crate::ProverError;

fn commit_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningVerificationFailed {
        reason: reason.to_string(),
    })
}

/// The field increment commitment and its retained PCS opening hint.
pub struct FieldIncObject<PCS: CommitmentScheme> {
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub column: FieldIncrementColumn<PCS::Field>,
}

/// Commit `FieldRdInc` directly as field elements, padding with zeros to the
/// minimum dense-object arity. An identically zero table still commits because
/// the dense schedule depends on its shape, not its contents.
pub fn commit_field_inc<F, PCS>(
    setup: &PCS::ProverSetup,
    log_t: usize,
    column: FieldIncrementColumn<F>,
) -> Result<FieldIncObject<PCS>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
{
    let layout = FieldIncLayout::new(log_t);
    let polynomial = DenseTraceColumnPoly::new(
        column.clone(),
        CommitmentGrid {
            total_vars: layout.num_vars(),
            log_t,
            log_k_chunk: 0,
            order: TracePolynomialOrder::CycleMajor,
        },
    )
    .ok_or_else(|| commit_failed("FieldRdInc disagrees with the packed trace arity"))?;
    let (commitment, hint) = tracing::info_span!("commit_field_inc", num_vars = layout.num_vars())
        .in_scope(|| PCS::commit_full_width_object(setup, &polynomial, layout.layout_digest()))
        .map_err(commit_failed)?;
    Ok(FieldIncObject {
        commitment,
        hint,
        column,
    })
}
