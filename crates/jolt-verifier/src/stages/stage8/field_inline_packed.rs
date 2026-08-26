//! Stage 8's packed field-inline seam: the FR limb reconstruction member of
//! the reconstruction batch and the packed limb object's final opening. The
//! packed axis never RLC-splices `FieldRdInc` (that is the homomorphic seam,
//! `super::field_inline`); instead the stage-6b reduced claim feeds the
//! reconstruction member here, whose per-column openings discharge against
//! the committed limb object. `reconstruction.rs`, `verify.rs`, and the
//! prover's packed recipe interact with the packed FR protocol only through
//! this module.

use jolt_claims::protocols::field_inline::lattice::{
    field_inc_limb_count, FieldIncLimbPackingPlan, FieldIncLimbReconstruction,
    FieldIncLimbReconstructionChallenges, FieldIncLimbReconstructionInputClaims,
    FieldIncLimbReconstructionOutputClaims, FieldIncLimbShape,
};
use jolt_claims::protocols::field_inline::{
    FieldIncLimbReconstructionPublic, FieldInlineDerivedId, FieldInlineRelationId,
};
use jolt_claims::protocols::jolt::JoltOneHotConfig;
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, EvaluationClaim};
use jolt_poly::{try_eq_mle, Point};
use jolt_transcript::Transcript;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage6b::Stage6bClearOutput;
use crate::stages::stage8::{OneHotTraceCommitmentMetadata, OneHotTraceSetupMetadata};
use crate::VerifierError;

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{:?}", FieldInlineRelationId::FieldIncLimbReconstruction),
        reason: reason.to_string(),
    }
}

fn batch_failed(reason: impl ToString) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    }
}

/// The per-proof FR limb object shape: the proof field's canonical limb
/// count, the trace arity, and the shared one-hot chunk width.
pub fn limb_shape<F: JoltField>(
    log_t: usize,
    one_hot_config: JoltOneHotConfig,
) -> FieldIncLimbShape {
    FieldIncLimbShape {
        limbs: field_inc_limb_count::<F>(),
        log_t,
        log_k_chunk: one_hot_config.committed_chunk_bits(),
    }
}

/// The FR limb reconstruction member: booleanity legs per committed column
/// against a fresh reference point, plus the balanced-digit decode leg that
/// settles the stage-6b reduced `FieldRdInc` claim.
#[derive(Clone)]
pub struct FieldIncLimbReconstructionInstance<F: JoltField> {
    symbolic: FieldIncLimbReconstruction,
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField> FieldIncLimbReconstructionInstance<F> {
    pub fn new(shape: FieldIncLimbShape) -> Self {
        Self {
            symbolic: FieldIncLimbReconstruction::new(shape),
            _field: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for FieldIncLimbReconstructionInstance<F> {
    type Symbolic = FieldIncLimbReconstruction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// The booleanity reference point is a full-cell-domain vector whose
    /// width is runtime shape data, so the generic scalar-stream draw cannot
    /// build it (the untrusted-advice member's idiom).
    fn draw_challenges<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
    ) -> Result<FieldIncLimbReconstructionChallenges<F>, VerifierError> {
        Ok(FieldIncLimbReconstructionChallenges {
            r_reference: transcript.challenge_vector(self.symbolic.rounds()),
            gamma: transcript.challenge_scalar(),
        })
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &FieldIncLimbReconstructionInputClaims<Vec<F>>,
    ) -> Result<FieldIncLimbReconstructionOutputClaims<Vec<F>>, VerifierError> {
        // Cycle variables bind first (low bits of the `(digit-value ‖ cycle)`
        // cell order), so the reversed sumcheck point is msb-first; every
        // column opens at the same point.
        let point: Vec<F> = sumcheck_point.iter().rev().copied().collect();
        let columns = self.symbolic.expected_output_openings::<F>().len();
        Ok(FieldIncLimbReconstructionOutputClaims {
            columns: vec![point; columns],
        })
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        input_points: &FieldIncLimbReconstructionInputClaims<Vec<F>>,
        output_points: &FieldIncLimbReconstructionOutputClaims<Vec<F>>,
        challenges: &FieldIncLimbReconstructionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let FieldInlineDerivedId::FieldIncLimbReconstruction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        let opening_point = output_points
            .columns
            .first()
            .ok_or_else(|| public_input_failed("the limb reconstruction produced no openings"))?;
        let chunk_width = self.symbolic.shape().log_k_chunk;
        let (r_digit, r_cycle) = opening_point.split_at_checked(chunk_width).ok_or_else(|| {
            public_input_failed("cell point is below the digit-value variable block")
        })?;
        match public {
            FieldIncLimbReconstructionPublic::EqReference => {
                try_eq_mle(opening_point, &challenges.r_reference).map_err(public_input_failed)
            }
            FieldIncLimbReconstructionPublic::EqCycle => {
                try_eq_mle(r_cycle, input_points.rd_inc()).map_err(public_input_failed)
            }
            FieldIncLimbReconstructionPublic::DigitValue => {
                Ok(jolt_claims::lattice::balanced_inc_value(r_digit))
            }
        }
    }
}

/// The assembled FR limb member and the claim it consumes.
pub struct FieldIncLimbMember<F: JoltField> {
    pub instance: FieldIncLimbReconstructionInstance<F>,
    pub input_values: FieldIncLimbReconstructionInputClaims<F>,
    pub input_points: FieldIncLimbReconstructionInputClaims<Vec<F>>,
}

/// The member and its consumed claim, built from the stage-6b FR increment
/// reduction's terminus: the packed FR phase always runs (every FR-on proof
/// carries the limb object, `FieldRdInc` zero or not).
pub fn build_member<F: JoltField>(
    shape: FieldIncLimbShape,
    stage6b: &Stage6bClearOutput<F>,
) -> Result<FieldIncLimbMember<F>, VerifierError> {
    let value = stage6b
        .output_values
        .field_registers_inc_claim_reduction
        .rd_inc;
    let point = stage6b.output_points.field_registers_inc_opening_point();
    if point.len() != shape.log_t {
        return Err(public_input_failed(format!(
            "reduced FieldRdInc claim has {} variables, expected {}",
            point.len(),
            shape.log_t
        )));
    }
    Ok(FieldIncLimbMember {
        instance: FieldIncLimbReconstructionInstance::new(shape),
        input_values: FieldIncLimbReconstructionInputClaims { rd_inc: value },
        input_points: FieldIncLimbReconstructionInputClaims {
            rd_inc: point.to_vec(),
        },
    })
}

/// Validate the packed FR limb commitment against the canonical plan: the
/// one-hot backend, the shared chunk width, the plan's digest, and the packed
/// arity (the `validate_auxiliary_metadata` discipline; the setup is the
/// trace's own — same arity by the norm-budget geometry — so its default
/// digest is the trace's, not this object's).
pub fn validate_field_inc_limbs_metadata<C, S>(
    commitment: &C,
    setup: &S,
    plan: &FieldIncLimbPackingPlan,
) -> Result<(), VerifierError>
where
    C: OneHotTraceCommitmentMetadata,
    S: OneHotTraceSetupMetadata,
{
    if !commitment.is_one_hot_backend() {
        return Err(batch_failed(
            "the FR limb commitment must use Akita's one-hot backend",
        ));
    }
    let one_hot_k = 1usize << plan.chunk_width();
    if commitment.one_hot_k() != one_hot_k || setup.one_hot_k() != one_hot_k {
        return Err(batch_failed(format!(
            "FR limb commitment/setup one-hot chunk size must equal canonical K={one_hot_k}"
        )));
    }
    if commitment.layout_digest() != plan.layout_digest() {
        return Err(batch_failed(
            "the FR limb commitment has a noncanonical layout digest",
        ));
    }
    let packed_num_vars = plan.packing().packed_num_vars();
    if commitment.num_vars() != packed_num_vars || setup.max_num_vars() != packed_num_vars {
        return Err(batch_failed(format!(
            "FR limb commitment/setup arity must equal canonical packed arity {packed_num_vars}"
        )));
    }
    if commitment.poly_count() != 1 || setup.max_num_polys_per_commitment_group() != 1 {
        return Err(batch_failed(
            "the FR limb object must contain one physical polynomial",
        ));
    }
    Ok(())
}

/// Assemble the limb object's prefix-packed claims from the reconstruction
/// member's per-column openings: every canonical column, its point mapped to
/// the committed row-major order, all sharing one canonical point. Shared
/// verbatim by the packed prover's stage 8.
pub fn field_inc_limb_packed_claims<F: JoltField>(
    plan: &FieldIncLimbPackingPlan,
    values: &FieldIncLimbReconstructionOutputClaims<F>,
    points: &FieldIncLimbReconstructionOutputClaims<Vec<F>>,
) -> Result<jolt_openings::PrefixPackedClaims<F>, VerifierError> {
    let ids = plan.packing().ids();
    if values.columns.len() != ids.len() || points.columns.len() != ids.len() {
        return Err(batch_failed(format!(
            "the limb reconstruction produced {} claims for {} canonical columns",
            values.columns.len(),
            ids.len()
        )));
    }
    let mut common_point: Option<Vec<F>> = None;
    let mut evaluations = Vec::with_capacity(ids.len());
    for ((polynomial, value), leaf_point) in ids.iter().zip(&values.columns).zip(&points.columns) {
        let claim = EvaluationClaim::new(Point::high_to_low(leaf_point.clone()), *value);
        let point = plan
            .column_point(*polynomial, claim.point.as_slice())
            .map_err(batch_failed)?;
        if let Some(expected) = &common_point {
            if expected != &point {
                return Err(batch_failed(format!(
                    "FR limb column {polynomial:?} does not share the canonical opening point"
                )));
            }
        } else {
            common_point = Some(point);
        }
        evaluations.push(claim.value);
    }
    let common_point =
        common_point.ok_or_else(|| batch_failed("the FR limb object has no columns"))?;
    Ok(plan.packed_claims(common_point, evaluations))
}

/// The canonical per-proof plan, from the same shape both fronts derive.
pub fn limb_plan<F: JoltField>(
    log_t: usize,
    one_hot_config: JoltOneHotConfig,
) -> Result<FieldIncLimbPackingPlan, VerifierError> {
    FieldIncLimbPackingPlan::new(&limb_shape::<F>(log_t, one_hot_config)).map_err(batch_failed)
}

/// The stage-8 entry: resolve the proof's FR limb slots (commitment, opening,
/// reconstruction leaves) fail-closed and verify the packed opening.
pub fn verify_proof_opening<PCS, T>(
    log_t: usize,
    one_hot_config: JoltOneHotConfig,
    setup: &PCS::VerifierSetup,
    commitment: Option<&PCS::Output>,
    opening_proof: Option<&PCS::Proof>,
    reconstruction: &super::reconstruction::ReconstructionClearOutput<PCS::Field>,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    T: Transcript<Challenge = PCS::Field>,
{
    let commitment = commitment.ok_or(VerifierError::MissingProofPayload {
        field: "field_inc_limbs_commitment",
    })?;
    let opening_proof = opening_proof.ok_or(VerifierError::MissingProofPayload {
        field: "joint_opening_proof.field_inc_limbs",
    })?;
    let values = reconstruction
        .output_values
        .field_inc_limbs
        .as_ref()
        .ok_or(VerifierError::MissingProofPayload {
            field: "reconstruction.field_inc_limbs",
        })?;
    let points = reconstruction
        .output_points
        .field_inc_limbs
        .as_ref()
        .ok_or(VerifierError::MissingProofPayload {
            field: "reconstruction.field_inc_limbs",
        })?;
    let plan = limb_plan::<PCS::Field>(log_t, one_hot_config)?;
    verify_final_opening::<PCS, T>(
        &plan,
        commitment,
        setup,
        values,
        points,
        opening_proof,
        transcript,
    )
}

/// Verify the packed FR limb object's final opening: metadata fail-closed,
/// the reduced physical claim from the reconstruction member's leaves, then
/// the PCS batch verification against the proof's limb slot.
pub fn verify_final_opening<PCS, T>(
    plan: &FieldIncLimbPackingPlan,
    commitment: &PCS::Output,
    setup: &PCS::VerifierSetup,
    values: &FieldIncLimbReconstructionOutputClaims<PCS::Field>,
    points: &FieldIncLimbReconstructionOutputClaims<Vec<PCS::Field>>,
    proof: &PCS::Proof,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    T: Transcript<Challenge = PCS::Field>,
{
    validate_field_inc_limbs_metadata(commitment, setup, plan)?;
    let packed_claims = field_inc_limb_packed_claims(plan, values, points)?;
    let physical_claim = plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed)?;
    PCS::verify_batch(
        commitment,
        physical_claim.point.as_slice(),
        std::slice::from_ref(&physical_claim.value),
        proof,
        setup,
        transcript,
    )
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })
}
