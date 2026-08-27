//! The packed FR prover seam: limb-word extraction from the FR oracle, the
//! stage-0 dense limb-group commit, and the stage-8 heterogeneous batch
//! entry. `stage0.rs`, `stage8.rs`, and `prover.rs` interact with the packed
//! FR protocol only through this module (the prover half of the
//! `jolt_verifier::stages::stage8::field_inline_packed` seam).

use jolt_claims::protocols::field_inline::lattice::{canonical_limbs, FieldIncLimbPackingPlan};
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
};
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, PrecommittedClaim, TransparentObjectSetup};
use jolt_poly::{boolean_point_msb, Polynomial};
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage8::field_inline_packed::{
    limb_plan, reduced_field_rd_inc, reduced_precommitted_claim, FieldIncLimbClaims,
};
use jolt_verifier::VerifierError;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use crate::ProverError;

fn commit_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningVerificationFailed {
        reason: reason.to_string(),
    })
}

/// The committed FR limb group: the canonical plan, the packed dense
/// limb-word polynomial (retained so stage 8 evaluates the limb claims from
/// the committed data), the commitment the proof carries, and the opening
/// hint the heterogeneous batch consumes.
pub struct FieldIncLimbsObject<PCS: CommitmentScheme> {
    pub plan: FieldIncLimbPackingPlan,
    pub polynomial: Polynomial<PCS::Field>,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Stage 0's FR commit: `FieldRdInc`'s canonical u64 limbs, slot-major over
/// `(limb ‖ cycle)`, as one dense commitment object under the plan's
/// transparent setup — the advice-object treatment. Unconditional: an
/// identically zero `FieldRdInc` still commits (all-zero content is legal;
/// dense schedules are keyed by shape, never content).
pub fn commit_field_inc_limbs<F, PCS>(
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<FieldIncLimbsObject<PCS>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
{
    let oracle =
        witness
            .field_inline()
            .ok_or(ProverError::Witness(WitnessError::UnavailableView {
                label: "packed field-inline limb commit oracle",
            }))?;
    let rd_inc: Vec<F> = oracle
        .oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))
        .map_err(ProverError::Witness)?;
    let num_rows = 1usize << log_t;
    if rd_inc.len() != num_rows {
        return Err(commit_failed(
            "the FR oracle's FieldRdInc table disagrees with the trace arity",
        ));
    }

    let plan = limb_plan::<F>(log_t).map_err(ProverError::Verifier)?;
    let limb_count = plan.packing().ids().len();
    let mut evaluations = vec![F::default(); 1usize << plan.packing().packed_num_vars()];
    for (cycle, value) in rd_inc.iter().enumerate() {
        // canonical_limbs allocates per cycle; a slice-writing encoder is the
        // upgrade path if this shows in trace-scale profiles.
        let limbs = canonical_limbs(value);
        if limbs.len() != limb_count {
            return Err(commit_failed(
                "the canonical limb decomposition disagrees with the limb plan",
            ));
        }
        for (limb, word) in limbs.into_iter().enumerate() {
            evaluations[(limb << log_t) | cycle] = F::from_u64(word);
        }
    }
    let polynomial = Polynomial::new(evaluations);
    let (setup, _verifier_setup) =
        PCS::transparent_object_setup(plan.packing().packed_num_vars(), plan.layout_digest())
            .map_err(commit_failed)?;
    let (commitment, hint) = tracing::info_span!(
        "commit_field_inc_limbs",
        packed_num_vars = plan.packing().packed_num_vars()
    )
    .in_scope(|| PCS::commit(&polynomial, &setup))
    .map_err(commit_failed)?;
    Ok(FieldIncLimbsObject {
        plan,
        polynomial,
        commitment,
        hint,
    })
}

/// The FR group's reduced precommitted claim paired with the wire claims the
/// proof carries beside it.
pub type FieldIncLimbBatchEntry<F, C> = (PrecommittedClaim<F, C>, FieldIncLimbClaims<F>);

/// Stage 8's FR batch entry: the limb-column evaluations at the stage-6b
/// reduced `FieldRdInc` point (each read off the committed packed polynomial
/// at its boolean slot prefix, so the claims use the exact convention the
/// PCS proves), reduced through the shared verifier seam into the
/// heterogeneous batch's FR precommitted claim. Returns the wire claims the
/// proof carries beside the reduced entry.
pub fn stage8_batch_entry<F, PCS, T>(
    object: &FieldIncLimbsObject<PCS>,
    stage6b: &Stage6bClearOutput<F>,
    transcript: &mut T,
) -> Result<FieldIncLimbBatchEntry<F, PCS::Output>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone,
    T: Transcript<Challenge = F>,
{
    let (reduced_value, reduced_point) = reduced_field_rd_inc(stage6b);
    let selector_vars = object.plan.packing().selector_num_vars();
    let limbs = (0..object.plan.packing().ids().len())
        .map(|slot| {
            let mut point = boolean_point_msb::<F>(selector_vars, slot);
            point.extend_from_slice(reduced_point);
            object.polynomial.evaluate(&point)
        })
        .collect();
    let claims = FieldIncLimbClaims { limbs };
    let claim = reduced_precommitted_claim(
        &object.plan,
        &object.commitment,
        &claims,
        (reduced_value, reduced_point),
        transcript,
    )
    .map_err(ProverError::Verifier)?;
    Ok((claim, claims))
}
