//! The final opening's `OneHotTrace` group claim, shared by the clear
//! batched opening and Whiteout's masked opening.

use std::collections::BTreeMap;

use jolt_algebra::JoltField;
use jolt_claims::protocols::jolt::lattice::packing::OneHotTraceShape;
use jolt_claims::protocols::jolt::lattice::strategy::ONE_HOT_TRACE_LAYOUT;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_openings::{CommitmentScheme, EvaluationClaim, GroupOpeningClaim};
use jolt_verifier::stages::stage8::akita::one_hot_trace_claim;
use jolt_verifier::{CheckedInputs, VerifierError};

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

pub(super) fn batch_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

/// The `OneHotTrace` group's claim at the final opening, from the committed
/// columns' `leaves`.
pub(super) fn trace_group_claim<F, PCS, VC>(
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    commitment: &PCS::Output,
) -> Result<GroupOpeningClaim<F, PCS::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let chunk_width = config.one_hot_config.committed_chunk_bits();
    let formula_dimensions = crate::stages::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::HammingWeightClaimReduction,
    )?;
    let plan = ONE_HOT_TRACE_LAYOUT
        .plan(&OneHotTraceShape {
            ra_layout: formula_dimensions.ra_layout,
            log_t,
            log_k_chunk: chunk_width,
        })
        .map_err(batch_failed::<F>)?;
    one_hot_trace_claim(&plan, chunk_width, leaves, commitment).map_err(ProverError::Verifier)
}
