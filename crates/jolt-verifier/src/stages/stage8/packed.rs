//! The Akita final opening.
//!
//! `OneHotTrace` prefix-packs its semantic columns into one physical
//! polynomial. Advice and direct committed-program objects join it as
//! precommitted Akita groups and are discharged by one joint opening.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, precommitted_packing_plan, OneHotTraceShape, PrecommittedPackingShape,
    PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::lattice::strategy::{
    OneHotTraceLayoutPlan, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, JoltOneHotConfig};
use jolt_field::Field;
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, GroupOpeningClaim, PrecommittedClaim, PrecommittedRole,
};
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, Transcript};

use super::precommitted::precommitted_final_openings;
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

fn validate_auxiliary_metadata<C, S>(
    commitment: &C,
    setup: &S,
    plan: &PrefixPackedObjectPlan,
) -> Result<(), VerifierError>
where
    C: OneHotTraceCommitmentMetadata,
    S: OneHotTraceSetupMetadata,
{
    if commitment.is_one_hot_backend() {
        return Err(batch_failed(
            "auxiliary prefix-packed commitments must use Akita's dense backend",
        ));
    }
    let packed_num_vars = plan.packing().packed_num_vars();
    if commitment.layout_digest() != plan.layout_digest()
        || setup.default_layout_digest() != plan.layout_digest()
    {
        return Err(batch_failed(
            "auxiliary commitment/setup has a noncanonical layout digest",
        ));
    }
    if commitment.num_vars() != packed_num_vars || setup.max_num_vars() != packed_num_vars {
        return Err(batch_failed(format!(
            "auxiliary commitment/setup arity must equal canonical packed arity {packed_num_vars}"
        )));
    }
    if commitment.poly_count() != 1 || setup.max_num_polys_per_commitment_group() != 1 {
        return Err(batch_failed(
            "auxiliary prefix-packed objects must contain one physical polynomial",
        ));
    }
    Ok(())
}

/// One resolved commitment object: its canonical packing plus the borrowed
/// commitment and shape-exact setup the final PCS opening runs against.
struct ResolvedObject<'a, PCS: CommitmentScheme> {
    plan: PrefixPackedObjectPlan,
    commitment: &'a PCS::Output,
    setup: &'a PCS::VerifierSetup,
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
    let claims = object
        .plan
        .packing()
        .ids()
        .iter()
        .map(|id| {
            leaves
                .get(id)
                .cloned()
                .map(|claim| (*id, claim))
                .ok_or_else(|| {
                    batch_failed(format!(
                        "missing final auxiliary claim for packed leaf {id:?}"
                    ))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let semantic = object.plan.packed_claims(&claims).map_err(batch_failed)?;
    object
        .plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed)
}

/// Resolve one advice object's packing/commitment/setup triple, or `None`
/// when the reduction schedule says the kind is absent. A setup may exist for
/// an absent per-proof object because preprocessing is capacity-derived.
fn advice_object<'a, PCS: CommitmentScheme>(
    present: bool,
    leaf: Option<&EvaluationClaim<PCS::Field>>,
    commitment: Option<&'a PCS::Output>,
    setup: Option<&'a PCS::VerifierSetup>,
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
    let (Some(leaf), Some(commitment), Some(setup)) = (leaf, commitment, setup) else {
        return Err(batch_failed(format!(
            "{kind:?} advice object without a final claim, commitment, or setup"
        )));
    };
    let plan = advice_packing_plan(kind, leaf.point.len()).map_err(batch_failed)?;
    Ok(Some(ResolvedObject {
        plan,
        commitment,
        setup,
    }))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the per-object commitments and their preprocessing setups, resolved here in one place"
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
    // Per-object packings, commitments, and setups in canonical object order:
    // `OneHotTrace` is one prefix-packed polynomial, followed
    // by the optional auxiliary commitment objects. The shared layout is the
    // same one the prover committed under.
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
    let leaves = leaf_claims(schedule, stage6b, stage7)?;
    let packed_claims = one_hot_trace_packed_claims(&plan, chunk_width, &leaves)?;
    let packed_claim = plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed)?;
    let untrusted = advice_object::<PCS>(
        schedule.untrusted_advice.is_some(),
        leaves.get(&JoltCommittedPolynomial::UntrustedAdvice),
        untrusted_advice_commitment,
        preprocessing.untrusted_advice_setup.as_ref(),
        JoltAdviceKind::Untrusted,
    )?;
    let trusted = advice_object::<PCS>(
        schedule.trusted_advice.is_some(),
        leaves.get(&JoltCommittedPolynomial::TrustedAdvice),
        trusted_advice_commitment,
        preprocessing.trusted_advice_setup.as_ref(),
        JoltAdviceKind::Trusted,
    )?;

    let untrusted_claim = if let Some(object) = untrusted.as_ref() {
        validate_auxiliary_metadata(object.commitment, object.setup, &object.plan)?;
        Some(reduce_object(object, &leaves, transcript)?)
    } else {
        None
    };
    let trusted_claim = if let Some(object) = trusted.as_ref() {
        validate_auxiliary_metadata(object.commitment, object.setup, &object.plan)?;
        Some(reduce_object(object, &leaves, transcript)?)
    } else {
        None
    };

    let committed = preprocessing.program.committed();
    let program_plan = committed
        .map(|committed| {
            let bytecode_len = preprocessing.program.bytecode_len();
            if !bytecode_len.is_multiple_of(committed.bytecode_chunk_count()) {
                return Err(batch_failed(
                    "bytecode chunk count does not divide bytecode length",
                ));
            }
            let chunk_rows = bytecode_len
                .checked_div(committed.bytecode_chunk_count())
                .ok_or_else(|| batch_failed("direct bytecode chunk count must be nonzero"))?;
            if !chunk_rows.is_power_of_two() {
                return Err(batch_failed(
                    "direct bytecode chunk row count must be a power of two",
                ));
            }
            let image_words = preprocessing
                .program
                .program_image_len_words()
                .next_power_of_two()
                .max(2);
            precommitted_packing_plan(&PrecommittedPackingShape {
                bytecode_chunks: committed.bytecode_chunk_count(),
                log_bytecode_rows: crate::num::ilog2(chunk_rows),
                trace_order: committed.trace_order,
                program_image_log_words: Some(crate::num::ilog2(image_words)),
            })
            .map_err(batch_failed)
        })
        .transpose()?;
    let plans = program_plan
        .as_ref()
        .map(|plan| plan.objects().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    if committed.map_or(0, |program| program.direct_program_commitments.len()) != plans.len()
        || preprocessing.direct_program_setups.len() != plans.len()
    {
        return Err(batch_failed(
            "direct committed-program commitments/setups do not match the canonical plan",
        ));
    }

    // Canonical public batch order precedes OneHotTrace.
    let capacity = 2usize
        .checked_add(plans.len())
        .ok_or_else(|| batch_failed("precommitted group capacity overflows"))?;
    let mut precommitted = Vec::with_capacity(capacity);
    for (role, object, claim) in [
        (
            PrecommittedRole::UntrustedAdvice,
            untrusted.as_ref(),
            untrusted_claim.as_ref(),
        ),
        (
            PrecommittedRole::TrustedAdvice,
            trusted.as_ref(),
            trusted_claim.as_ref(),
        ),
    ] {
        if let (Some(object), Some(claim)) = (object, claim) {
            precommitted.push(PrecommittedClaim::new(
                role,
                GroupOpeningClaim::new(
                    (*object.commitment).clone(),
                    claim.point.as_slice().to_vec(),
                    vec![claim.value],
                ),
            ));
        }
    }

    if let Some(committed) = committed {
        for ((plan, commitment), setup) in plans
            .into_iter()
            .zip(&committed.direct_program_commitments)
            .zip(&preprocessing.direct_program_setups)
        {
            let object: ResolvedObject<'_, PCS> = ResolvedObject {
                plan,
                commitment,
                setup,
            };
            validate_auxiliary_metadata(object.commitment, object.setup, &object.plan)?;
            let physical = reduce_object(&object, &leaves, transcript)?;
            let id = object
                .plan
                .packing()
                .ids()
                .first()
                .copied()
                .ok_or_else(|| batch_failed("direct program object has no polynomial id"))?;
            let role = match id {
                JoltCommittedPolynomial::BytecodeChunk(index) => {
                    PrecommittedRole::BytecodeChunk(index)
                }
                JoltCommittedPolynomial::ProgramImageInit => PrecommittedRole::ProgramImageInit,
                JoltCommittedPolynomial::RdInc
                | JoltCommittedPolynomial::RamInc
                | JoltCommittedPolynomial::InstructionRa(_)
                | JoltCommittedPolynomial::BytecodeRa(_)
                | JoltCommittedPolynomial::RamRa(_)
                | JoltCommittedPolynomial::TrustedAdvice
                | JoltCommittedPolynomial::UntrustedAdvice
                | JoltCommittedPolynomial::BalancedIncDigit(_)
                | JoltCommittedPolynomial::BalancedIncCarry => {
                    return Err(batch_failed(
                        "unexpected direct committed-program object role",
                    ))
                }
            };
            precommitted.push(PrecommittedClaim::new(
                role,
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
pub fn one_hot_trace_packed_claims<F: Field>(
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

/// One auxiliary object's leaf claims: each of the plan's canonical columns
/// paired with its resolved leaf claim. Shared verbatim by the packed
/// prover's stage 8, so both sides fail on the same missing leaf.
pub fn object_leaf_claims<F: Field>(
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
                        "missing final auxiliary claim for packed leaf {id:?}"
                    ))
                })
        })
        .collect()
}

/// Every packed column's single leaf claim, resolved from the precommitted
/// reductions and stage 7, keyed by committed polynomial. The canonical
/// object plans check coverage, point arity, and suffix compatibility.
/// Shared verbatim by the packed prover's stage 8.
pub fn leaf_claims<F: Field>(
    schedule: &PrecommittedSchedule,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>, VerifierError> {
    use JoltCommittedPolynomial as Poly;

    fn leaf<F: Field>(value: F, point: &[F]) -> EvaluationClaim<F> {
        EvaluationClaim::new(Point::high_to_low(point.to_vec()), value)
    }
    fn insert<F: Field>(
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
    fn insert_indexed<F: Field>(
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

    Ok(leaves)
}
