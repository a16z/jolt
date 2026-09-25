//! Akita commitment of the field-inline register increment polynomial.

use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
};
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_poly::Polynomial;
use jolt_verifier::VerifierError;
use jolt_witness::{JoltWitnessPlane, WitnessError};

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
}

/// Commit `FieldRdInc` directly as field elements, padding with zeros to the
/// minimum dense-object arity. An identically zero table still commits because
/// the dense schedule depends on its shape, not its contents.
pub fn commit_field_inc<F, PCS>(
    setup_context: &PCS::SetupContext,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<FieldIncObject<PCS>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
{
    let oracle =
        witness
            .field_inline()
            .ok_or(ProverError::Witness(WitnessError::UnavailableView {
                label: "packed field-inline commit oracle",
            }))?;
    let mut rd_inc: Vec<F> = oracle
        .oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))
        .map_err(ProverError::Witness)?;
    let num_rows = 1usize << log_t;
    if rd_inc.len() != num_rows {
        return Err(commit_failed(
            "the field-inline oracle's FieldRdInc table disagrees with the trace arity",
        ));
    }

    let layout = FieldIncLayout::new(log_t);
    rd_inc.resize(1usize << layout.num_vars(), F::zero());
    let polynomial = Polynomial::new(rd_inc);
    let (commitment, hint) = tracing::info_span!("commit_field_inc", num_vars = layout.num_vars())
        .in_scope(|| {
            PCS::commit_full_width_object(setup_context, &polynomial, layout.layout_digest())
        })
        .map_err(commit_failed)?;
    Ok(FieldIncObject { commitment, hint })
}
