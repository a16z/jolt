//! Stage 8's packed field-inline seam: the FR limb group's presence resolve,
//! the linear recomposition check against the stage-6b reduced `FieldRdInc`
//! claim, and the semantic-to-physical claim reduction that places the group
//! in the heterogeneous batch. `packed.rs` interacts with the packed FR
//! protocol only through this module; the prover's packed stage-8 recipe
//! derives its identical statement through [`reduced_precommitted_claim`].
//!
//! The group is ALWAYS present on an FR-on packed build (all-zero content is
//! legal — dense schedules are keyed by shape, never content), so presence is
//! not claim-gated: the schedule marker, the proof's commitment slot, and the
//! proof's claims slot must all agree, fail-closed both ways.

use jolt_claims::protocols::field_inline::lattice::{
    field_inc_limb_count, field_inc_limbs_precommitted_role, recompose_limbs,
    FieldIncLimbPackingPlan, FieldIncLimbShape,
};
use jolt_field::JoltField;
use jolt_openings::{GroupOpeningClaim, PrecommittedClaim};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::stage6b::outputs::Stage6bClearOutput;
use crate::stages::PrecommittedSchedule;
use crate::VerifierError;

fn batch_failed(reason: impl ToString) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    }
}

/// The FR limb group's schedule marker: carried by
/// [`PrecommittedSchedule`], always present on an FR-on packed build. The
/// group's geometry is fully derived from `log_T` and the proof field
/// ([`limb_plan`]), so the marker carries no data of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldIncLimbsScheduled;

/// The proof-carried FR limb-group evaluations at the stage-6b reduced
/// `FieldRdInc` point, in little-endian limb order (fp128: two). The
/// recomposition check and the batched opening bind them; they are absorbed
/// into the transcript by the prefix-pack reduction before its selector
/// draw.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct FieldIncLimbClaims<F> {
    pub limbs: Vec<F>,
}

/// The canonical per-proof limb plan, from the trace arity and the proof
/// field's limb count — the same shape both fronts derive.
pub fn limb_plan<F: JoltField>(log_t: usize) -> Result<FieldIncLimbPackingPlan, VerifierError> {
    FieldIncLimbPackingPlan::new(&FieldIncLimbShape {
        limbs: field_inc_limb_count::<F>(),
        log_t,
    })
    .map_err(batch_failed)
}

/// Resolve the FR limb group's proof slots against the schedule marker,
/// fail-closed both ways. Every arm except full agreement rejects: the
/// marker is constructed on every FR-on schedule, so a missing marker means
/// broken input validation, not a legal FR-absent proof.
pub fn resolve_proof_slots<'a, F, C>(
    schedule: &PrecommittedSchedule,
    commitment: Option<&'a C>,
    claims: Option<&'a FieldIncLimbClaims<F>>,
) -> Result<(&'a C, &'a FieldIncLimbClaims<F>), VerifierError> {
    match (schedule.field_inc_limbs, commitment, claims) {
        (Some(FieldIncLimbsScheduled), Some(commitment), Some(claims)) => Ok((commitment, claims)),
        (Some(FieldIncLimbsScheduled), None, _) => Err(VerifierError::MissingProofPayload {
            field: "field_inc_limbs_commitment",
        }),
        (Some(FieldIncLimbsScheduled), Some(_), None) => Err(VerifierError::MissingProofPayload {
            field: "claims.field_inc_limbs",
        }),
        (None, Some(_), _) | (None, _, Some(_)) => Err(batch_failed(
            "FR limb payload supplied without a scheduled limb group",
        )),
        (None, None, None) => Err(batch_failed(
            "an FR-on packed schedule must carry the FR limb group",
        )),
    }
}

/// The stage-6b reduced `FieldRdInc` claim `(value, point)` the limb group
/// binds to — read here by both fronts so neither can consume a different
/// cell.
pub fn reduced_field_rd_inc<F: JoltField>(stage6b: &Stage6bClearOutput<F>) -> (F, &[F]) {
    (
        stage6b
            .output_values
            .field_registers_inc_claim_reduction
            .rd_inc,
        stage6b.output_points.field_registers_inc_opening_point(),
    )
}

/// The FR limb group's reduced batch entry, shared verbatim by the packed
/// prover's stage 8 so both sides derive the same statement and transcript:
///
/// 1. the linear recomposition check binds the proof-carried limb
///    evaluations to the stage-6b reduced `FieldRdInc` claim (rejecting with
///    [`VerifierError::FieldIncLimbRecompositionMismatch`]) BEFORE any
///    reduction — the packing reduction itself stays claim-agnostic;
/// 2. the prefix-pack reduction absorbs the limb evaluations and the reduced
///    point, draws the slot selector, and yields the one physical claim the
///    heterogeneous batch discharges under the frozen FR role.
///
/// The check pins the weighted sum only: nothing verifies that the committed
/// columns are the canonical u64 decomposition (the u64 envelope is the
/// dense backend's norm budget, not a verifier range check), so a prover may
/// commit any limb split recomposing to the same claim. Benign — no relation
/// consumes a limb column individually; only the recomposed `FieldRdInc`
/// value reaches protocol state.
pub fn reduced_precommitted_claim<F, C, T>(
    plan: &FieldIncLimbPackingPlan,
    commitment: &C,
    claims: &FieldIncLimbClaims<F>,
    reduced_rd_inc: (F, &[F]),
    transcript: &mut T,
) -> Result<PrecommittedClaim<F, C>, VerifierError>
where
    F: JoltField,
    C: Clone,
    T: Transcript<Challenge = F>,
{
    let (reduced_value, reduced_point) = reduced_rd_inc;
    if recompose_limbs(&claims.limbs) != reduced_value {
        return Err(VerifierError::FieldIncLimbRecompositionMismatch);
    }
    let semantic = plan
        .packed_claims(reduced_point.to_vec(), claims.limbs.clone())
        .map_err(batch_failed)?;
    let physical = plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed)?;
    Ok(PrecommittedClaim::new(
        field_inc_limbs_precommitted_role(),
        GroupOpeningClaim::new(
            commitment.clone(),
            physical.point.as_slice().to_vec(),
            vec![physical.value],
        ),
    ))
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::indexing_slicing,
    clippy::as_conversions,
    reason = "tests index fixture data"
)]
mod tests {
    use jolt_field::{Fr, Ring};
    use jolt_poly::eq_index_msb;
    use jolt_transcript::Blake2bTranscript;

    use super::*;

    const LOG_T: usize = 4;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn limbs() -> FieldIncLimbClaims<Fr> {
        FieldIncLimbClaims {
            limbs: vec![fr(5), fr(7), fr(0), fr(0)],
        }
    }

    fn reduced_value() -> Fr {
        fr(5) + Fr::pow2(64) * fr(7)
    }

    #[test]
    fn recomposition_binds_the_reduced_claim() {
        let plan = limb_plan::<Fr>(LOG_T).unwrap();
        let point = vec![fr(3); LOG_T];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"fr-limb-seam-test");
        let claim = reduced_precommitted_claim(
            &plan,
            &(),
            &limbs(),
            (reduced_value(), point.as_slice()),
            &mut transcript,
        )
        .unwrap();
        assert_eq!(claim.role.order(), 2);
        assert_eq!(claim.claim.point.len(), plan.packing().packed_num_vars());
        // The physical value is the selector-weighted limb combination.
        let selector = &claim.claim.point[..plan.packing().selector_num_vars()];
        let expected: Fr = limbs()
            .limbs
            .iter()
            .enumerate()
            .map(|(slot, limb)| eq_index_msb(selector, slot as u128) * *limb)
            .sum();
        assert_eq!(claim.claim.evaluations, vec![expected]);

        let mut tampered = limbs();
        tampered.limbs[0] += fr(1);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"fr-limb-seam-test");
        assert!(matches!(
            reduced_precommitted_claim(
                &plan,
                &(),
                &tampered,
                (reduced_value(), point.as_slice()),
                &mut transcript,
            ),
            Err(VerifierError::FieldIncLimbRecompositionMismatch)
        ));
    }

    #[test]
    fn resolve_rejects_every_presence_disagreement() {
        let schedule = PrecommittedSchedule {
            bytecode: None,
            program_image: None,
            field_inc_limbs: Some(FieldIncLimbsScheduled),
        };
        let commitment = ();
        let claims = limbs();
        assert!(resolve_proof_slots(&schedule, Some(&commitment), Some(&claims)).is_ok());
        assert!(matches!(
            resolve_proof_slots::<Fr, ()>(&schedule, None, Some(&claims)),
            Err(VerifierError::MissingProofPayload {
                field: "field_inc_limbs_commitment"
            })
        ));
        assert!(matches!(
            resolve_proof_slots::<Fr, ()>(&schedule, Some(&commitment), None),
            Err(VerifierError::MissingProofPayload {
                field: "claims.field_inc_limbs"
            })
        ));

        let unscheduled = PrecommittedSchedule {
            field_inc_limbs: None,
            ..schedule
        };
        assert!(resolve_proof_slots(&unscheduled, Some(&commitment), Some(&claims)).is_err());
        assert!(resolve_proof_slots::<Fr, ()>(&unscheduled, None, None).is_err());
    }
}
