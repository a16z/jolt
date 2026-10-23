//! The Akita final opening.
//!
//! `OneHotTrace` batches its native columns at one common point. Advice,
//! field increments, and direct committed-program objects
//! join it as auxiliary Akita groups and are discharged by one joint opening.

use std::collections::BTreeMap;

use jolt_algebra::JoltField;
use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::lattice::packing::{
    advice_packing_plan, committed_program_packing_plan, OneHotTraceShape, PrefixPackedObjectPlan,
};
use jolt_claims::protocols::jolt::lattice::strategy::{
    OneHotTraceLayoutPlan, ONE_HOT_TRACE_LAYOUT,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, JoltOneHotConfig};
use jolt_crypto::VectorCommitment;
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, GroupOpeningClaim, TaggedGroupOpeningClaim,
};
use jolt_poly::Point;
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use super::precommitted::precommitted_final_openings;
#[cfg(feature = "akita")]
use crate::stages::stage4::outputs::Stage4ClearOutput;
use crate::stages::stage6b::outputs::{Stage6bClearOutput, Stage6bOutputPoints};
use crate::stages::stage7::outputs::{Stage7ClearOutput, Stage7OutputPoints};
use crate::stages::stage8::{OneHotTraceCommitmentMetadata, OneHotTraceSetupMetadata};
use crate::stages::PrecommittedSchedule;
use crate::VerifierError;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::lattice::{field_inc_group_role, FieldIncLayout};

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
    column_arity: usize,
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
    if commitment.num_vars() != column_arity || setup.max_num_vars() != column_arity {
        return Err(batch_failed(format!(
            "OneHotTrace commitment/setup arity must equal canonical column arity {column_arity}"
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

fn validate_group_commitment_metadata<C>(
    commitment: &C,
    layout_digest: [u8; 32],
    num_vars: usize,
) -> Result<(), VerifierError>
where
    C: OneHotTraceCommitmentMetadata,
{
    if commitment.is_one_hot_backend() {
        return Err(batch_failed(
            "auxiliary commitments must use Akita's dense backend",
        ));
    }
    if commitment.layout_digest() != layout_digest {
        return Err(batch_failed(
            "auxiliary commitment has a noncanonical layout digest",
        ));
    }
    if commitment.num_vars() != num_vars {
        return Err(batch_failed(format!(
            "auxiliary commitment arity must equal canonical arity {num_vars}"
        )));
    }
    if commitment.poly_count() != 1 {
        return Err(batch_failed(
            "auxiliary groups must contain one physical polynomial",
        ));
    }
    Ok(())
}

fn validate_packed_object_metadata<C>(
    commitment: &C,
    plan: &PrefixPackedObjectPlan,
) -> Result<(), VerifierError>
where
    C: OneHotTraceCommitmentMetadata,
{
    validate_group_commitment_metadata(
        commitment,
        plan.layout_digest(),
        plan.packing().packed_num_vars(),
    )
}

struct ResolvedObject<'a, PCS: CommitmentScheme> {
    plan: PrefixPackedObjectPlan,
    commitment: &'a PCS::Output,
}

fn reduce_object<PCS, C>(
    object: &ResolvedObject<'_, PCS>,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<PCS::Field>>,
    transcript: &mut C,
) -> Result<EvaluationClaim<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    C: Channel,
{
    let claims = object_leaf_claims(&object.plan, leaves)?;
    let semantic = object.plan.packed_claims(&claims).map_err(batch_failed)?;
    object
        .plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed)
}

fn advice_object<'a, PCS: CommitmentScheme>(
    leaf: Option<&EvaluationClaim<PCS::Field>>,
    commitment: Option<&'a PCS::Output>,
    kind: JoltAdviceKind,
) -> Result<Option<ResolvedObject<'a, PCS>>, VerifierError> {
    let (leaf, commitment) = match (leaf, commitment) {
        (None, None) => return Ok(None),
        (Some(_), None) => {
            return Err(batch_failed(format!(
                "{kind:?} advice final claim supplied without a commitment"
            )));
        }
        (None, Some(_)) => {
            return Err(batch_failed(format!(
                "{kind:?} advice commitment supplied without a final claim"
            )));
        }
        (Some(leaf), Some(commitment)) => (leaf, commitment),
    };
    let plan = advice_packing_plan(kind, leaf.point.len()).map_err(batch_failed)?;
    Ok(Some(ResolvedObject { plan, commitment }))
}

/// Bind the existing stage-6b reduced claim directly to the full-field commitment.
/// Shared by the prover and verifier so they consume the same claim and point.
#[cfg(feature = "field-inline")]
pub fn field_inc_claim<F: JoltField, C: Clone>(
    commitment: &C,
    stage6b: &Stage6bClearOutput<F>,
) -> Result<TaggedGroupOpeningClaim<F, C>, VerifierError> {
    let cycle_point = stage6b.output_points.field_registers_inc_opening_point();
    let point = FieldIncLayout::new(cycle_point.len())
        .opening_point(cycle_point)
        .map_err(|error| VerifierError::FinalOpeningBatchFailed {
            reason: error.to_string(),
        })?;
    Ok(TaggedGroupOpeningClaim::new(
        field_inc_group_role(),
        GroupOpeningClaim::new(
            commitment.clone(),
            point,
            vec![
                stage6b
                    .output_values
                    .field_registers_inc_claim_reduction
                    .rd_inc,
            ],
        ),
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the stage inputs are passed separately by the verifier driver"
)]
pub fn verify<PCS, VC, H>(
    formula_dimensions: &JoltFormulaDimensions,
    one_hot_config: JoltOneHotConfig,
    preprocessing: &crate::preprocessing::JoltVerifierPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    #[cfg(feature = "field-inline")] field_inc_commitment: &PCS::Output,
    transcript: &mut VerifierTranscript<'_, H>,
    schedule: &PrecommittedSchedule,
    #[cfg(feature = "akita")] stage4: &Stage4ClearOutput<PCS::Field>,
    stage6b: &Stage6bClearOutput<PCS::Field>,
    stage7: &Stage7ClearOutput<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = PCS::Field>,
    H: Sponge,
{
    // Auxiliary objects precede the OneHotTrace group in canonical role order: advice,
    // (field-inline) the always-present field-increment commitment, then the direct
    // committed-program objects. Optional objects join exactly when their direct final
    // reductions exist; presence must agree with the proof/preprocessing commitment slots.
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
        plan.num_vars(),
        plan.ids().len(),
        1 << chunk_width,
    )?;
    let leaves = leaf_claims(
        schedule,
        #[cfg(feature = "akita")]
        stage4,
        stage6b,
        stage7,
    )?;
    let main_group = one_hot_trace_claim(&plan, chunk_width, &leaves, one_hot_trace_commitment)?;
    let untrusted = advice_object::<PCS>(
        leaves.get(&JoltCommittedPolynomial::UntrustedAdvice),
        untrusted_advice_commitment,
        JoltAdviceKind::Untrusted,
    )?;
    let trusted = advice_object::<PCS>(
        leaves.get(&JoltCommittedPolynomial::TrustedAdvice),
        trusted_advice_commitment,
        JoltAdviceKind::Trusted,
    )?;

    let untrusted_claim = if let Some(object) = untrusted.as_ref() {
        validate_packed_object_metadata(object.commitment, &object.plan)?;
        Some(reduce_object(object, &leaves, transcript)?)
    } else {
        None
    };
    let trusted_claim = if let Some(object) = trusted.as_ref() {
        validate_packed_object_metadata(object.commitment, &object.plan)?;
        Some(reduce_object(object, &leaves, transcript)?)
    } else {
        None
    };

    let program = preprocessing
        .program
        .committed()
        .map(|committed| {
            committed_program_packing_plan(
                preprocessing.program.bytecode_len(),
                preprocessing.program.program_image_len_words(),
                committed.trace_order,
            )
            .map(|plan| (committed, plan))
            .map_err(batch_failed)
        })
        .transpose()?;
    let mut auxiliary_groups = Vec::with_capacity(5);
    for (object, claim) in [
        (untrusted.as_ref(), untrusted_claim.as_ref()),
        (trusted.as_ref(), trusted_claim.as_ref()),
    ] {
        if let (Some(object), Some(claim)) = (object, claim) {
            auxiliary_groups.push(TaggedGroupOpeningClaim::new(
                object.plan.group_role(),
                GroupOpeningClaim::new(
                    (*object.commitment).clone(),
                    claim.point.as_slice().to_vec(),
                    vec![claim.value],
                ),
            ));
        }
    }
    #[cfg(feature = "field-inline")]
    {
        let commitment = field_inc_commitment;
        let layout = FieldIncLayout::new(formula_dimensions.trace.log_t());
        validate_group_commitment_metadata(commitment, layout.layout_digest(), layout.num_vars())?;
        auxiliary_groups.push(field_inc_claim(commitment, stage6b)?);
    }

    if let Some((committed, plan)) = program {
        for (plan, commitment) in [
            (plan.bytecode, &committed.bytecode_commitment),
            (plan.program_image, &committed.program_image_commitment),
        ] {
            let object: ResolvedObject<'_, PCS> = ResolvedObject { plan, commitment };
            validate_packed_object_metadata(object.commitment, &object.plan)?;
            let physical = reduce_object(&object, &leaves, transcript)?;
            auxiliary_groups.push(TaggedGroupOpeningClaim::new(
                object.plan.group_role(),
                GroupOpeningClaim::new(
                    object.commitment.clone(),
                    physical.point.as_slice().to_vec(),
                    vec![physical.value],
                ),
            ));
        }
    }

    PCS::verify_batch(
        &preprocessing.pcs_setup,
        &auxiliary_groups,
        &main_group,
        transcript,
    )
    .map_err(opening_failed)?;

    Ok(())
}

/// Whiteout's stage 8: the trace group's claims as affine forms over the
/// pads, each column's form the masked opening of its final claim, for the
/// masked Akita opening. Nothing is absorbed here: the forms are built from
/// masked messages already absorbed, under challenges that precede none of
/// them, and Akita's masked root binds them without absorbing values
/// (`specs/whiteout-jolt-integration.md` §8, D25).
///
/// # Errors
///
/// [`VerifierError::ProtocolAxisUnimplemented`] for a proof with advice, a
/// committed program or field increments (their groups are W6d), and the
/// errors of the clear stage 8's shape checks.
#[cfg(all(feature = "akita", feature = "zk"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the stage inputs, as in `verify`"
)]
pub fn masked_trace_claim<F, PCS, VC>(
    formula_dimensions: &JoltFormulaDimensions,
    one_hot_config: JoltOneHotConfig,
    preprocessing: &crate::preprocessing::JoltVerifierPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    schedule: &PrecommittedSchedule,
    stage6b_points: &Stage6bOutputPoints<F>,
    stage7_points: &Stage7OutputPoints<F>,
    residual: &crate::stages::zk::residual::JoltResidual<F>,
) -> Result<jolt_zk::whiteout::MaskedGroupClaim<F, PCS::Output>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
{
    use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;

    if untrusted_advice_commitment.is_some()
        || trusted_advice_commitment.is_some()
        || preprocessing.program.committed().is_some()
        || cfg!(feature = "field-inline")
    {
        return Err(VerifierError::ProtocolAxisUnimplemented {
            axis: "zk (Whiteout)",
            pending: "precommitted groups: advice, committed programs, field increments (W6d)",
        });
    }
    let chunk_width = one_hot_config.committed_chunk_bits();
    let plan = ONE_HOT_TRACE_LAYOUT
        .plan(&OneHotTraceShape {
            ra_layout: formula_dimensions.ra_layout,
            log_t: formula_dimensions.trace.log_t(),
            log_k_chunk: chunk_width,
        })
        .map_err(batch_failed)?;
    validate_one_hot_trace_metadata(
        one_hot_trace_commitment,
        &preprocessing.pcs_setup,
        plan.layout_digest(),
        plan.num_vars(),
        plan.ids().len(),
        1 << chunk_width,
    )?;
    let leaves = leaves(
        schedule,
        stage6b_points,
        stage7_points,
        &mut |_| Ok(None),
        &mut |polynomial| residual.opening_form(&final_opening_id(polynomial).into()),
    )?;
    let (point, evaluations) = one_hot_trace_group(&plan, chunk_width, &leaves)?;
    Ok(jolt_zk::whiteout::MaskedGroupClaim {
        commitment: one_hot_trace_commitment.clone(),
        point,
        evaluations,
    })
}

/// Assembles the native `OneHotTrace` group claim: every canonical
/// column's leaf claim, its point mapped to the committed row-major order,
/// all required to share one canonical opening point. Shared verbatim by the
/// Akita prover's stage 8, so both sides derive the same native statement.
pub fn one_hot_trace_claim<F: JoltField, C: Clone>(
    plan: &OneHotTraceLayoutPlan,
    chunk_width: usize,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    commitment: &C,
) -> Result<GroupOpeningClaim<F, C>, VerifierError> {
    let leaves = leaves
        .iter()
        .map(|(polynomial, claim)| {
            (
                *polynomial,
                Leaf {
                    point: claim.point.as_slice().to_vec(),
                    value: claim.value,
                },
            )
        })
        .collect();
    let (point, evaluations) = one_hot_trace_group(plan, chunk_width, &leaves)?;
    Ok(GroupOpeningClaim::new(
        commitment.clone(),
        point,
        evaluations,
    ))
}

/// One committed column's leaf: its opening point and its value, a clear
/// value or, under Whiteout, an affine form over the pads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf<F, V> {
    pub point: Vec<F>,
    pub value: V,
}

/// The `OneHotTrace` group's common opening point and its columns' values in
/// the plan's canonical order: each column's point is mapped to the
/// committed row-major order, and all must agree.
pub fn one_hot_trace_group<F: JoltField, V: Clone>(
    plan: &OneHotTraceLayoutPlan,
    chunk_width: usize,
    leaves: &BTreeMap<JoltCommittedPolynomial, Leaf<F, V>>,
) -> Result<(Vec<F>, Vec<V>), VerifierError> {
    let mut common_point: Option<Vec<F>> = None;
    let mut evaluations = Vec::with_capacity(plan.ids().len());
    for polynomial in plan.ids() {
        let leaf = leaves.get(polynomial).ok_or_else(|| {
            batch_failed(format!(
                "missing final OneHotTrace claim for {polynomial:?}"
            ))
        })?;
        let point = ONE_HOT_TRACE_LAYOUT
            .column_point(*polynomial, chunk_width, &leaf.point)
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
        evaluations.push(leaf.value.clone());
    }
    let common_point = common_point.ok_or_else(|| batch_failed("OneHotTrace has no columns"))?;
    if common_point.len() != plan.num_vars() {
        return Err(batch_failed(
            "OneHotTrace opening point has incorrect arity",
        ));
    }
    Ok((common_point, evaluations))
}

/// One precommitted object's leaf claims: each of the plan's canonical columns
/// paired with its resolved leaf claim. Shared verbatim by the Akita
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

/// Every committed column's single leaf claim, resolved from stage 4, the
/// precommitted reductions, and stage 7, keyed by committed polynomial. The
/// canonical object plans check coverage, point arity, and suffix compatibility.
/// Shared verbatim by the Akita prover's stage 8.
pub fn leaf_claims<F: JoltField>(
    schedule: &PrecommittedSchedule,
    #[cfg(feature = "akita")] stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>, VerifierError> {
    use JoltCommittedPolynomial as Poly;

    let hamming = &stage7.output_values.hamming_weight_claim_reduction;
    let finals = precommitted_final_openings(
        schedule,
        &stage7.output_points,
        &stage6b.output_points,
        Some((&stage7.output_values, &stage6b.output_values)),
    )?;
    let indexed = |values: &[F], index: usize, polynomial: Poly| {
        values
            .get(index)
            .copied()
            .ok_or_else(|| batch_failed(format!("missing final value for {polynomial:?}")))
    };
    let leaves = leaves(
        schedule,
        &stage6b.output_points,
        &stage7.output_points,
        &mut |_kind| {
            #[cfg(feature = "akita")]
            if let Some(contribution) = stage4.ram_val_check_init.advice_contribution(_kind) {
                return Ok(Some(Leaf {
                    point: contribution.opening_point.clone(),
                    value: contribution.opening_value,
                }));
            }
            Ok(None)
        },
        &mut |polynomial| match polynomial {
            Poly::InstructionRa(index) => indexed(&hamming.instruction_ra, index, polynomial),
            Poly::BytecodeRa(index) => indexed(&hamming.bytecode_ra, index, polynomial),
            Poly::RamRa(index) => indexed(&hamming.ram_ra, index, polynomial),
            Poly::BalancedIncDigit(index) => {
                indexed(&hamming.balanced_inc_digits, index, polynomial)
            }
            Poly::BalancedIncCarry => Ok(hamming.balanced_inc_carry),
            Poly::RdInc
            | Poly::RamInc
            | Poly::Reserved4(_)
            | Poly::TrustedAdvice
            | Poly::UntrustedAdvice
            | Poly::ProgramImageInit
            | Poly::ProgramBytecode => finals
                .iter()
                .find(|opening| opening.polynomial == polynomial)
                .and_then(|opening| opening.opening_claim)
                .ok_or_else(|| {
                    batch_failed(format!("missing clear final value for {polynomial:?}"))
                }),
        },
    )?;
    Ok(leaves
        .into_iter()
        .map(|(polynomial, leaf)| {
            (
                polynomial,
                EvaluationClaim::new(Point::high_to_low(leaf.point), leaf.value),
            )
        })
        .collect())
}

/// Every committed column's leaf, keyed by committed polynomial: the one
/// owner of which stage supplies each column's opening point. The hamming
/// reduction's columns come from stage 7, the precommitted polynomials from
/// whichever phase completed their reduction, and (Akita) advice from
/// `advice`; every value from `value`.
pub fn leaves<F: JoltField, V>(
    schedule: &PrecommittedSchedule,
    stage6b_points: &Stage6bOutputPoints<F>,
    stage7_points: &Stage7OutputPoints<F>,
    advice: &mut dyn FnMut(JoltAdviceKind) -> Result<Option<Leaf<F, V>>, VerifierError>,
    value: &mut dyn FnMut(JoltCommittedPolynomial) -> Result<V, VerifierError>,
) -> Result<BTreeMap<JoltCommittedPolynomial, Leaf<F, V>>, VerifierError> {
    use JoltCommittedPolynomial as Poly;

    let mut leaves = BTreeMap::new();
    let mut insert = |polynomial: Poly, point: &[F], value: V| {
        if leaves
            .insert(
                polynomial,
                Leaf {
                    point: point.to_vec(),
                    value,
                },
            )
            .is_some()
        {
            return Err(batch_failed(format!(
                "duplicate Akita final claim for {polynomial:?}"
            )));
        }
        Ok(())
    };
    let hamming = &stage7_points.hamming_weight_claim_reduction;
    type Family<'a, F> = (&'a [Vec<F>], fn(usize) -> Poly);
    let families: [Family<'_, F>; 4] = [
        (&hamming.instruction_ra, Poly::InstructionRa),
        (&hamming.bytecode_ra, Poly::BytecodeRa),
        (&hamming.ram_ra, Poly::RamRa),
        (&hamming.balanced_inc_digits, Poly::BalancedIncDigit),
    ];
    for (points, polynomial) in families {
        for (index, point) in points.iter().enumerate() {
            insert(polynomial(index), point, value(polynomial(index))?)?;
        }
    }
    insert(
        Poly::BalancedIncCarry,
        &hamming.balanced_inc_carry,
        value(Poly::BalancedIncCarry)?,
    )?;

    for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
        if let Some(leaf) = advice(kind)? {
            let polynomial = match kind {
                JoltAdviceKind::Trusted => Poly::TrustedAdvice,
                JoltAdviceKind::Untrusted => Poly::UntrustedAdvice,
            };
            insert(polynomial, &leaf.point, leaf.value)?;
        }
    }

    for opening in precommitted_final_openings(schedule, stage7_points, stage6b_points, None)? {
        insert(
            opening.polynomial,
            &opening.point,
            value(opening.polynomial)?,
        )?;
    }

    Ok(leaves)
}
