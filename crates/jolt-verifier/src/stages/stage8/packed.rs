//! The Akita final opening.
//!
//! `OneHotTrace` prefix-packs its semantic columns into one physical
//! polynomial. Advice and direct committed-program objects join it as
//! precommitted Akita groups and are discharged by one joint opening.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, committed_program_packing_plan, OneHotTraceShape, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::lattice::strategy::{
    OneHotTraceLayoutPlan, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, JoltOneHotConfig};
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, EvaluationClaim, GroupOpeningClaim, PrecommittedClaim};
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, Transcript};

use super::precommitted::precommitted_final_openings;
#[cfg(feature = "akita")]
use crate::stages::stage4::outputs::Stage4ClearOutput;
use crate::stages::stage6b::outputs::Stage6bClearOutput;
use crate::stages::stage7::outputs::Stage7ClearOutput;
use crate::stages::stage8::{OneHotTraceCommitmentMetadata, OneHotTraceSetupMetadata};
use crate::stages::PrecommittedSchedule;
use crate::VerifierError;

fn batch_failed(reason: impl ToString) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    }
}

fn opening_failed(reason: impl ToString) -> VerifierError {
    VerifierError::FinalOpeningVerificationFailed {
        reason: reason.to_string(),
    }
}

fn validate_one_hot_trace_metadata<C, S>(
    commitment: &C,
    setup: &S,
    canonical_digest: [u8; 32],
    packed_arity: usize,
    physical_poly_count: usize,
    one_hot_k: usize,
) -> Result<(), VerifierError>
where
    C: OneHotTraceCommitmentMetadata,
    S: OneHotTraceSetupMetadata,
{
    if !commitment.is_one_hot_backend() {
        return Err(batch_failed(
            "OneHotTrace commitment must use Akita's one-hot backend",
        ));
    }
    if commitment.one_hot_k() != one_hot_k || setup.one_hot_k() != one_hot_k {
        return Err(batch_failed(format!(
            "OneHotTrace commitment/setup one-hot chunk size must equal canonical K={one_hot_k}"
        )));
    }
    if commitment.layout_digest() != canonical_digest {
        return Err(batch_failed(
            "OneHotTrace commitment has a noncanonical layout digest",
        ));
    }
    if commitment.num_vars() != packed_arity || setup.max_num_vars() != packed_arity {
        return Err(batch_failed(format!(
            "OneHotTrace commitment/setup arity must equal canonical packed arity {packed_arity}"
        )));
    }
    if commitment.poly_count() != physical_poly_count
        || setup.max_num_polys_per_commitment_group() != physical_poly_count
    {
        return Err(batch_failed(format!(
            "OneHotTrace commitment/setup physical polynomial count must equal {physical_poly_count}"
        )));
    }
    if setup.default_layout_digest() != canonical_digest {
        return Err(batch_failed(
            "OneHotTrace verifier setup has a noncanonical layout digest",
        ));
    }
    Ok(())
}

fn validate_precommitted_metadata<C>(
    commitment: &C,
    plan: &PrefixPackedObjectPlan,
) -> Result<(), VerifierError>
where
    C: OneHotTraceCommitmentMetadata,
{
    if commitment.is_one_hot_backend() {
        return Err(batch_failed(
            "precommitted prefix-packed commitments must use Akita's dense backend",
        ));
    }
    let packed_num_vars = plan.packing().packed_num_vars();
    if commitment.layout_digest() != plan.layout_digest() {
        return Err(batch_failed(
            "precommitted commitment has a noncanonical layout digest",
        ));
    }
    if commitment.num_vars() != packed_num_vars {
        return Err(batch_failed(format!(
            "precommitted commitment arity must equal canonical packed arity {packed_num_vars}"
        )));
    }
    if commitment.poly_count() != 1 {
        return Err(batch_failed(
            "precommitted prefix-packed objects must contain one physical polynomial",
        ));
    }
    Ok(())
}

/// One resolved commitment object and its canonical packing.
struct ResolvedObject<'a, PCS: CommitmentScheme> {
    plan: PrefixPackedObjectPlan,
    commitment: &'a PCS::Output,
}

fn reduce_object<PCS, T>(
    object: &ResolvedObject<'_, PCS>,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<PCS::Field>>,
    transcript: &mut T,
) -> Result<EvaluationClaim<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    T: Transcript<Challenge = PCS::Field>,
{
    let claims = object_leaf_claims(&object.plan, leaves)?;
    let semantic = object.plan.packed_claims(&claims).map_err(batch_failed)?;
    object
        .plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed)
}

/// Resolve one advice object's packing and commitment, or `None` when the
/// reduction schedule says the kind is absent.
fn advice_object<'a, PCS: CommitmentScheme>(
    present: bool,
    leaf: Option<&EvaluationClaim<PCS::Field>>,
    commitment: Option<&'a PCS::Output>,
    kind: JoltAdviceKind,
) -> Result<Option<ResolvedObject<'a, PCS>>, VerifierError> {
    if !present {
        if commitment.is_some() || leaf.is_some() {
            return Err(batch_failed(format!(
                "{kind:?} advice commitment or final claim supplied without a scheduled reduction"
            )));
        }
        return Ok(None);
    }
    let (Some(leaf), Some(commitment)) = (leaf, commitment) else {
        return Err(batch_failed(format!(
            "{kind:?} advice object without a final claim or commitment"
        )));
    };
    let plan = advice_packing_plan(kind, leaf.point.len()).map_err(batch_failed)?;
    Ok(Some(ResolvedObject { plan, commitment }))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the stage inputs are passed separately by the verifier driver"
)]
pub fn verify<PCS, VC, T>(
    formula_dimensions: &JoltFormulaDimensions,
    one_hot_config: JoltOneHotConfig,
    preprocessing: &crate::preprocessing::JoltVerifierPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    proof: &PCS::Proof,
    transcript: &mut T,
    schedule: &PrecommittedSchedule,
    #[cfg(feature = "akita")] stage4: &Stage4ClearOutput<PCS::Field>,
    stage6b: &Stage6bClearOutput<PCS::Field>,
    stage7: &Stage7ClearOutput<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: Clone + AppendToTranscript + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    // Precommitted objects precede the OneHotTrace group in canonical role order.
    // Optional objects join exactly when their direct final reductions exist;
    // presence must agree with the proof/preprocessing commitment slots.
    let chunk_width = one_hot_config.committed_chunk_bits();
    let one_hot_trace_shape = OneHotTraceShape {
        ra_layout: formula_dimensions.ra_layout,
        log_t: formula_dimensions.trace.log_t(),
        log_k_chunk: chunk_width,
    };
    let plan = ONE_HOT_TRACE_LAYOUT
        .plan(&one_hot_trace_shape)
        .map_err(batch_failed)?;
    validate_one_hot_trace_metadata(
        one_hot_trace_commitment,
        &preprocessing.pcs_setup,
        plan.layout_digest(),
        plan.packing().packed_num_vars(),
        1,
        1 << chunk_width,
    )?;
    let leaves = leaf_claims(
        schedule,
        #[cfg(feature = "akita")]
        stage4,
        stage6b,
        stage7,
    )?;
    let packed_claims = one_hot_trace_packed_claims(&plan, chunk_width, &leaves)?;
    let packed_claim = plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed)?;
    let untrusted = advice_object::<PCS>(
        schedule.untrusted_advice.is_some(),
        leaves.get(&JoltCommittedPolynomial::UntrustedAdvice),
        untrusted_advice_commitment,
        JoltAdviceKind::Untrusted,
    )?;
    let trusted = advice_object::<PCS>(
        schedule.trusted_advice.is_some(),
        leaves.get(&JoltCommittedPolynomial::TrustedAdvice),
        trusted_advice_commitment,
        JoltAdviceKind::Trusted,
    )?;

    let untrusted_claim = if let Some(object) = untrusted.as_ref() {
        validate_precommitted_metadata(object.commitment, &object.plan)?;
        Some(reduce_object(object, &leaves, transcript)?)
    } else {
        None
    };
    let trusted_claim = if let Some(object) = trusted.as_ref() {
        validate_precommitted_metadata(object.commitment, &object.plan)?;
        Some(reduce_object(object, &leaves, transcript)?)
    } else {
        None
    };

    let committed = preprocessing.program.committed();
    let program_plan = committed
        .map(|committed| {
            committed_program_packing_plan(
                preprocessing.program.bytecode_len(),
                committed.bytecode_chunk_count(),
                preprocessing.program.program_image_len_words(),
                committed.trace_order,
            )
            .map_err(batch_failed)
        })
        .transpose()?;
    let plans = program_plan
        .as_ref()
        .map(|plan| plan.objects().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    if committed.map_or(0, |program| program.direct_program_commitments.len()) != plans.len() {
        return Err(batch_failed(
            "direct committed-program commitments do not match the canonical plan",
        ));
    }

    let capacity = 2usize
        .checked_add(plans.len())
        .ok_or_else(|| batch_failed("precommitted group capacity overflows"))?;
    let mut precommitted = Vec::with_capacity(capacity);
    for (object, claim) in [
        (untrusted.as_ref(), untrusted_claim.as_ref()),
        (trusted.as_ref(), trusted_claim.as_ref()),
    ] {
        if let (Some(object), Some(claim)) = (object, claim) {
            precommitted.push(PrecommittedClaim::new(
                object.plan.precommitted_role(),
                GroupOpeningClaim::new(
                    (*object.commitment).clone(),
                    claim.point.as_slice().to_vec(),
                    vec![claim.value],
                ),
            ));
        }
    }

    if let Some(committed) = committed {
        for (plan, commitment) in plans.into_iter().zip(&committed.direct_program_commitments) {
            let object: ResolvedObject<'_, PCS> = ResolvedObject { plan, commitment };
            validate_precommitted_metadata(object.commitment, &object.plan)?;
            let physical = reduce_object(&object, &leaves, transcript)?;
            precommitted.push(PrecommittedClaim::new(
                object.plan.precommitted_role(),
                GroupOpeningClaim::new(
                    (*object.commitment).clone(),
                    physical.point.as_slice().to_vec(),
                    vec![physical.value],
                ),
            ));
        }
    }

    let main_group = GroupOpeningClaim::new(
        one_hot_trace_commitment.clone(),
        packed_claim.point.as_slice().to_vec(),
        vec![packed_claim.value],
    );
    PCS::verify_batch(
        &preprocessing.pcs_setup,
        &precommitted,
        &main_group,
        proof,
        transcript,
    )
    .map_err(opening_failed)?;

    Ok(())
}

/// Assembles the `OneHotTrace` prefix-packed claims: every canonical
/// column's leaf claim, its point mapped to the committed row-major order,
/// all required to share one canonical opening point. Shared verbatim by the
/// packed prover's stage 8, so both sides derive the same packed statement.
pub fn one_hot_trace_packed_claims<F: JoltField>(
    plan: &OneHotTraceLayoutPlan,
    chunk_width: usize,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
) -> Result<jolt_openings::PrefixPackedClaims<F>, VerifierError> {
    let mut common_point: Option<Vec<F>> = None;
    let mut evaluations = Vec::with_capacity(plan.packing().ids().len());
    for polynomial in plan.packing().ids() {
        let claim = leaves.get(polynomial).ok_or_else(|| {
            batch_failed(format!(
                "missing final OneHotTrace claim for {polynomial:?}"
            ))
        })?;
        let point = ONE_HOT_TRACE_LAYOUT
            .column_point(*polynomial, chunk_width, claim.point.as_slice())
            .map_err(batch_failed)?;
        if let Some(expected) = &common_point {
            if expected != &point {
                return Err(batch_failed(format!(
                    "OneHotTrace column {polynomial:?} does not share the canonical opening point"
                )));
            }
        } else {
            common_point = Some(point);
        }
        evaluations.push(claim.value);
    }
    let common_point = common_point.ok_or_else(|| batch_failed("OneHotTrace has no columns"))?;
    Ok(plan.packed_claims(common_point, evaluations))
}

/// One precommitted object's leaf claims: each of the plan's canonical columns
/// paired with its resolved leaf claim. Shared verbatim by the packed
/// prover's stage 8, so both sides fail on the same missing leaf.
pub fn object_leaf_claims<F: JoltField>(
    plan: &PrefixPackedObjectPlan,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
) -> Result<BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>, VerifierError> {
    plan.packing()
        .ids()
        .iter()
        .map(|id| {
            leaves
                .get(id)
                .cloned()
                .map(|claim| (*id, claim))
                .ok_or_else(|| {
                    batch_failed(format!(
                        "missing final precommitted claim for packed leaf {id:?}"
                    ))
                })
        })
        .collect()
}

/// Every packed column's single leaf claim, resolved from the precommitted
/// reductions and stage 7, keyed by committed polynomial. The canonical
/// object plans check coverage, point arity, and suffix compatibility.
/// Shared verbatim by the packed prover's stage 8.
pub fn leaf_claims<F: JoltField>(
    schedule: &PrecommittedSchedule,
    #[cfg(feature = "akita")] stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>, VerifierError> {
    use JoltCommittedPolynomial as Poly;

    fn leaf<F: JoltField>(value: F, point: &[F]) -> EvaluationClaim<F> {
        EvaluationClaim::new(Point::high_to_low(point.to_vec()), value)
    }
    fn insert<F: JoltField>(
        leaves: &mut BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
        polynomial: JoltCommittedPolynomial,
        claim: EvaluationClaim<F>,
    ) -> Result<(), VerifierError> {
        if leaves.insert(polynomial, claim).is_some() {
            return Err(batch_failed(format!(
                "duplicate packed final claim for {polynomial:?}"
            )));
        }
        Ok(())
    }
    fn insert_indexed<F: JoltField>(
        leaves: &mut BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
        values: &[F],
        points: &[Vec<F>],
        polynomial: impl Fn(usize) -> JoltCommittedPolynomial,
    ) -> Result<(), VerifierError> {
        for (index, (value, point)) in values.iter().zip(points).enumerate() {
            insert(leaves, polynomial(index), leaf(*value, point))?;
        }
        Ok(())
    }
    let mut leaves = BTreeMap::new();

    let hamming_values = &stage7.output_values.hamming_weight_claim_reduction;
    let hamming_points = &stage7.output_points.hamming_weight_claim_reduction;
    insert_indexed(
        &mut leaves,
        &hamming_values.instruction_ra,
        &hamming_points.instruction_ra,
        Poly::InstructionRa,
    )?;
    insert_indexed(
        &mut leaves,
        &hamming_values.bytecode_ra,
        &hamming_points.bytecode_ra,
        Poly::BytecodeRa,
    )?;
    insert_indexed(
        &mut leaves,
        &hamming_values.ram_ra,
        &hamming_points.ram_ra,
        Poly::RamRa,
    )?;

    insert_indexed(
        &mut leaves,
        &hamming_values.balanced_inc_digits,
        &hamming_points.balanced_inc_digits,
        Poly::BalancedIncDigit,
    )?;
    insert(
        &mut leaves,
        Poly::BalancedIncCarry,
        leaf(
            hamming_values.balanced_inc_carry,
            &hamming_points.balanced_inc_carry,
        ),
    )?;

    #[cfg(feature = "akita")]
    {
        for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
            if schedule.advice(kind).is_some() {
                let contribution = stage4
                    .ram_val_check_init
                    .advice_contribution(kind)
                    .ok_or_else(|| {
                        batch_failed(format!("missing {kind:?} advice stage-4 contribution"))
                    })?;
                let polynomial = match kind {
                    JoltAdviceKind::Trusted => Poly::TrustedAdvice,
                    JoltAdviceKind::Untrusted => Poly::UntrustedAdvice,
                };
                insert(
                    &mut leaves,
                    polynomial,
                    leaf(contribution.opening_value, &contribution.opening_point),
                )?;
            }
        }
        let reduced_precommitted = PrecommittedSchedule {
            trusted_advice: None,
            untrusted_advice: None,
            bytecode: schedule.bytecode.clone(),
            program_image: schedule.program_image.clone(),
        };
        for opening in precommitted_final_openings(
            &reduced_precommitted,
            &stage7.output_points,
            &stage6b.output_points,
            Some((&stage7.output_values, &stage6b.output_values)),
        )? {
            let value = opening.opening_claim.ok_or_else(|| {
                batch_failed(format!(
                    "missing clear final value for {:?}",
                    opening.polynomial
                ))
            })?;
            insert(&mut leaves, opening.polynomial, leaf(value, &opening.point))?;
        }
    }
    #[cfg(not(feature = "akita"))]
    {
        for opening in precommitted_final_openings(
            schedule,
            &stage7.output_points,
            &stage6b.output_points,
            Some((&stage7.output_values, &stage6b.output_values)),
        )? {
            let value = opening.opening_claim.ok_or_else(|| {
                batch_failed(format!(
                    "missing clear final value for {:?}",
                    opening.polynomial
                ))
            })?;
            insert(&mut leaves, opening.polynomial, leaf(value, &opening.point))?;
        }
    }

    Ok(leaves)
}
