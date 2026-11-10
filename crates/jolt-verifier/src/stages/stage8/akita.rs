//! The Akita final opening.
//!
//! `OneHotTrace` batches its native columns at one common point. Advice,
//! RV64 increments, field-register increments, and direct committed-program objects
//! join it as auxiliary Akita groups and are discharged by one joint opening.

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::increments::{fused_inc_group_role, FusedIncLayout};
use std::borrow::BorrowMut;
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
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_openings::CommitmentGroupRole;
use jolt_openings::{
    CommitmentScheme, EvaluationClaim, GroupBatchOpening, GroupOpeningClaim,
    TaggedGroupOpeningClaim,
};
use jolt_poly::Point;
use jolt_transcript::{Encoding, FieldTranscript, VerifierState};

use super::precommitted::{precommitted_final_openings, PrecommittedFinalOpening};
use crate::preprocessing::JoltVerifierPreprocessing;
use crate::stages::pow::{check_pow, PowSite, StagePowNonces};
#[cfg(feature = "akita")]
use crate::stages::stage4::outputs::Stage4ClearOutput;
#[cfg(all(feature = "akita", feature = "zk"))]
use crate::stages::stage4::ram_val_check::RamValCheckStagedOpenings;
use crate::stages::stage6b::outputs::Stage6bClearOutput;
#[cfg(all(feature = "akita", feature = "zk"))]
use crate::stages::stage6b::outputs::Stage6bOutputPoints;
use crate::stages::stage7::outputs::{Stage7ClearOutput, Stage7OutputPoints};
use crate::stages::stage8::{OneHotTraceCommitmentMetadata, OneHotTraceSetupMetadata};
#[cfg(all(feature = "akita", feature = "zk"))]
use crate::stages::zk::residual::JoltResidual;
use crate::stages::PrecommittedSchedule;
use crate::VerifierError;
#[cfg(all(feature = "akita", feature = "zk", feature = "field-inline"))]
use jolt_claims::protocols::field_inline::geometry::claim_reductions::increments::field_rd_inc_reduced;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::lattice::{field_inc_group_role, FieldIncLayout};
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_claims::protocols::jolt::geometry::bytecode::fused_inc_read_raf_opening;
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_claims::protocols::jolt::geometry::dimensions::CommitmentMatrixShape;
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_claims::protocols::jolt::geometry::ram::val_check_advice_opening;
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_openings::PackedEvaluations;
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_zk::masking::AffineRow;
#[cfg(all(feature = "akita", feature = "zk"))]
use jolt_zk::whiteout::{DenseSource, MaskedGroupClaim, MaskedOpeningShape, PrecommittedShape};

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

fn reduce_object<PCS, T>(
    object: &ResolvedObject<'_, PCS>,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<PCS::Field>>,
    transcript: &mut T,
) -> Result<EvaluationClaim<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    T: FieldTranscript<PCS::Field>,
{
    let claims = object_leaf_claims(&object.plan, leaves)?;
    let semantic = object.plan.packed_claims(&claims).map_err(batch_failed)?;
    object
        .plan
        .packing()
        .reduce_claims(&semantic, transcript)
        .map_err(batch_failed)
}

/// Whether any of `plans` draws a selector point, so that stage 8 has a
/// challenge burst for proof of work to protect.
pub fn has_selectors<'a>(plans: impl IntoIterator<Item = &'a PrefixPackedObjectPlan>) -> bool {
    plans
        .into_iter()
        .any(|plan| plan.packing().selector_num_vars() > 0)
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

/// The ReadRAF reduction already produces the increment claim at a cycle point.
#[cfg(feature = "akita")]
pub fn fused_inc_claim<F: JoltField, C: Clone>(
    commitment: &C,
    stage6b: &Stage6bClearOutput<F>,
) -> Result<TaggedGroupOpeningClaim<F, C>, VerifierError> {
    let point = stage6b.output_points.fused_inc_opening_point();
    Ok(TaggedGroupOpeningClaim::new(
        fused_inc_group_role(),
        GroupOpeningClaim::new(
            commitment.clone(),
            FusedIncLayout::new(point.len())
                .opening_point(point)
                .map_err(batch_failed)?,
            vec![stage6b.output_values.bytecode_read_raf.fused_inc],
        ),
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the stage inputs are passed separately by the verifier driver"
)]
pub fn verify<'a, PCS, VC, T>(
    formula_dimensions: &JoltFormulaDimensions,
    one_hot_config: JoltOneHotConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    #[cfg(feature = "akita")] fused_inc_commitment: Option<&PCS::Output>,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    #[cfg(feature = "field-inline")] field_inc_commitment: Option<&PCS::Output>,
    proof: &PCS::Proof,
    pow_nonces: &StagePowNonces,
    transcript: &mut T,
    schedule: &PrecommittedSchedule,
    #[cfg(feature = "akita")] stage4: &Stage4ClearOutput<PCS::Field>,
    stage6b: &Stage6bClearOutput<PCS::Field>,
    stage7: &Stage7ClearOutput<PCS::Field>,
    context: &mut PCS::VerificationContext<'_>,
) -> Result<PCS::PendingEof, VerifierError>
where
    PCS: CommitmentScheme + GroupBatchOpening,
    PCS::Output: Clone + Encoding + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FieldTranscript<PCS::Field> + BorrowMut<VerifierState<'a, PCS::Sponge>>,
{
    // Auxiliary objects precede the OneHotTrace group in canonical role order: advice,
    // (field-inline) the always-present field-increment commitment, then the direct
    // committed-program objects, then FusedInc. Optional objects join exactly when their direct final
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

    // One nonce covers every object's selector point: the reductions absorb
    // only claims fixed by earlier messages, so no prover message falls
    // between the draws. With no selector there is no burst and no nonce.
    let selectors = has_selectors(
        [untrusted.as_ref(), trusted.as_ref()]
            .into_iter()
            .flatten()
            .map(|object| &object.plan)
            .chain(
                program
                    .iter()
                    .flat_map(|(_, plan)| [&plan.bytecode, &plan.program_image]),
            ),
    );
    if selectors {
        check_pow(PowSite::Stage8, pow_nonces, transcript)?;
    } else {
        pow_nonces.require_absent(PowSite::Stage8)?;
    }

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
        let commitment = field_inc_commitment.ok_or(VerifierError::MissingProofPayload {
            field: "field_inc_commitment",
        })?;
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

    #[cfg(feature = "akita")]
    {
        let commitment = fused_inc_commitment.ok_or(VerifierError::MissingProofPayload {
            field: "fused_inc_commitment",
        })?;
        let layout = FusedIncLayout::new(formula_dimensions.trace.log_t());
        validate_group_commitment_metadata(commitment, layout.layout_digest(), layout.num_vars())?;
        auxiliary_groups.push(fused_inc_claim(commitment, stage6b)?);
    }

    PCS::verify_batch(
        &preprocessing.pcs_setup,
        &auxiliary_groups,
        &main_group,
        proof,
        BorrowMut::<VerifierState<'a, PCS::Sponge>>::borrow_mut(transcript),
        context,
    )
    .map_err(opening_failed)
}

/// One Whiteout precommitted object, as the statement fixes it.
#[cfg(all(feature = "akita", feature = "zk"))]
#[derive(Clone, Debug)]
pub enum WhiteoutObject<C> {
    /// An advice or committed-program object, prefix-packed.
    Packed {
        plan: PrefixPackedObjectPlan,
        commitment: C,
    },
    /// The field-increment polynomial, committed full width.
    #[cfg(feature = "field-inline")]
    FieldInc {
        layout: FieldIncLayout,
        commitment: C,
    },
    /// The RV64 increment polynomial, committed bounded.
    FusedInc {
        layout: FusedIncLayout,
        commitment: C,
    },
}

#[cfg(all(feature = "akita", feature = "zk"))]
impl<C> WhiteoutObject<C> {
    /// The object's group role in the clear opening's canonical order.
    pub fn role(&self) -> CommitmentGroupRole {
        match self {
            Self::Packed { plan, .. } => plan.group_role(),
            #[cfg(feature = "field-inline")]
            Self::FieldInc { .. } => field_inc_group_role(),
            Self::FusedInc { .. } => fused_inc_group_role(),
        }
    }

    fn commitment(&self) -> &C {
        match self {
            Self::Packed { commitment, .. } => commitment,
            #[cfg(feature = "field-inline")]
            Self::FieldInc { commitment, .. } => commitment,
            Self::FusedInc { commitment, .. } => commitment,
        }
    }

    fn shape(&self) -> PrecommittedShape {
        match self {
            Self::Packed { plan, .. } => PrecommittedShape {
                num_vars: plan.packing().packed_num_vars(),
                source: DenseSource::Bounded,
            },
            #[cfg(feature = "field-inline")]
            Self::FieldInc { layout, .. } => PrecommittedShape {
                num_vars: layout.num_vars(),
                source: DenseSource::FullField,
            },
            Self::FusedInc { layout, .. } => PrecommittedShape {
                num_vars: layout.num_vars(),
                source: DenseSource::Bounded,
            },
        }
    }
}

/// Whiteout's final opening, from the statement: the trace group's layout
/// plan and the precommitted objects in the clear stage 8's canonical order
/// (advice, then the field increments or the committed program, then the
/// RV64 increments), every
/// commitment's and the setup's metadata checked against them. The masked
/// opening's shape is derived from this plan, never from a commitment
/// header.
#[cfg(all(feature = "akita", feature = "zk"))]
#[derive(Clone, Debug)]
pub struct WhiteoutOpeningPlan<C> {
    pub trace: OneHotTraceLayoutPlan,
    pub objects: Vec<WhiteoutObject<C>>,
}

#[cfg(all(feature = "akita", feature = "zk"))]
impl<C> WhiteoutOpeningPlan<C> {
    /// Whether stage 8 draws packing selectors, so a proof of work precedes
    /// them ([`PowSite::Stage8`]), as on the clear Akita path.
    pub fn has_selectors(&self) -> bool {
        has_selectors(self.objects.iter().filter_map(|object| match object {
            WhiteoutObject::Packed { plan, .. } => Some(plan),
            #[cfg(feature = "field-inline")]
            WhiteoutObject::FieldInc { .. } => None,
            WhiteoutObject::FusedInc { .. } => None,
        }))
    }

    /// The masked opening's groups: the objects, then the trace group.
    pub fn shape(&self) -> MaskedOpeningShape {
        MaskedOpeningShape {
            precommitted: self.objects.iter().map(WhiteoutObject::shape).collect(),
            num_vars: self.trace.num_vars(),
            num_polys: self.trace.ids().len(),
        }
    }
}

/// [`WhiteoutOpeningPlan`] of a statement. An advice object's word arity is
/// the memory layout's, the arity stage 4 opens it at.
///
/// # Errors
///
/// The errors of the clear stage 8's shape checks.
#[cfg(all(feature = "akita", feature = "zk"))]
#[cfg_attr(
    feature = "field-inline",
    expect(
        clippy::too_many_arguments,
        reason = "one commitment argument per precommitted object kind, the field increments included"
    )
)]
pub fn whiteout_opening_plan<PCS, VC>(
    formula_dimensions: &JoltFormulaDimensions,
    one_hot_config: JoltOneHotConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    one_hot_trace_commitment: &PCS::Output,
    fused_inc_commitment: &PCS::Output,
    untrusted_advice_commitment: Option<&PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    #[cfg(feature = "field-inline")] field_inc_commitment: &PCS::Output,
) -> Result<WhiteoutOpeningPlan<PCS::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let chunk_width = one_hot_config.committed_chunk_bits();
    let trace = ONE_HOT_TRACE_LAYOUT
        .plan(&OneHotTraceShape {
            ra_layout: formula_dimensions.ra_layout,
            log_t: formula_dimensions.trace.log_t(),
            log_k_chunk: chunk_width,
        })
        .map_err(batch_failed)?;
    validate_one_hot_trace_metadata(
        one_hot_trace_commitment,
        &preprocessing.pcs_setup,
        trace.layout_digest(),
        trace.num_vars(),
        trace.ids().len(),
        1 << chunk_width,
    )?;
    let memory_layout = preprocessing.program.memory_layout();
    let mut objects = Vec::with_capacity(4);
    for (kind, commitment, max_bytes) in [
        (
            JoltAdviceKind::Untrusted,
            untrusted_advice_commitment,
            memory_layout.max_untrusted_advice_size,
        ),
        (
            JoltAdviceKind::Trusted,
            trusted_advice_commitment,
            memory_layout.max_trusted_advice_size,
        ),
    ] {
        if let Some(commitment) = commitment {
            let max_bytes = usize::try_from(max_bytes).map_err(|_| {
                batch_failed(format!("{kind:?} advice size {max_bytes} exceeds usize"))
            })?;
            let plan = advice_packing_plan(
                kind,
                CommitmentMatrixShape::advice_from_max_bytes(max_bytes).total_vars(),
            )
            .map_err(batch_failed)?;
            validate_packed_object_metadata(commitment, &plan)?;
            objects.push(WhiteoutObject::Packed {
                plan,
                commitment: commitment.clone(),
            });
        }
    }
    #[cfg(feature = "field-inline")]
    {
        let layout = FieldIncLayout::new(formula_dimensions.trace.log_t());
        validate_group_commitment_metadata(
            field_inc_commitment,
            layout.layout_digest(),
            layout.num_vars(),
        )?;
        objects.push(WhiteoutObject::FieldInc {
            layout,
            commitment: field_inc_commitment.clone(),
        });
    }
    if let Some(committed) = preprocessing.program.committed() {
        let plan = committed_program_packing_plan(
            preprocessing.program.bytecode_len(),
            preprocessing.program.program_image_len_words(),
            committed.trace_order,
        )
        .map_err(batch_failed)?;
        for (plan, commitment) in [
            (plan.bytecode, &committed.bytecode_commitment),
            (plan.program_image, &committed.program_image_commitment),
        ] {
            validate_packed_object_metadata(commitment, &plan)?;
            objects.push(WhiteoutObject::Packed {
                plan,
                commitment: commitment.clone(),
            });
        }
    }
    let layout = FusedIncLayout::new(formula_dimensions.trace.log_t());
    validate_group_commitment_metadata(
        fused_inc_commitment,
        layout.layout_digest(),
        layout.num_vars(),
    )?;
    objects.push(WhiteoutObject::FusedInc {
        layout,
        commitment: fused_inc_commitment.clone(),
    });
    Ok(WhiteoutOpeningPlan { trace, objects })
}

/// One group of a Whiteout opening: its commitment, its opening point (the
/// caller's order) and one value per member, a masked form on the
/// verifier's side and its decoding on the prover's.
#[cfg(all(feature = "akita", feature = "zk"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhiteoutGroup<F, V, C> {
    pub commitment: C,
    pub point: Vec<F>,
    pub values: Vec<V>,
}

/// `Σ cᵢ · vᵢ` over a value type: a clear evaluation's or a masked form's.
#[cfg(all(feature = "akita", feature = "zk"))]
type LinearCombination<'a, F, V> = dyn Fn(&[(F, &V)]) -> V + 'a;

/// Whiteout's stage 8 over `plan`: each precommitted object's claim, then
/// the trace group's, from the committed columns' `leaves`, the RV64
/// increments' `fused_inc` leaf (and, with field-inline, the field
/// increments' `field_inc` leaf) by the clear stage 8's leaf geometry. A packed object's reduction binds its layout and point
/// and absorbs no evaluation: its values are affine forms of masked messages
/// already absorbed, under challenges that precede none of them, and Akita's
/// masked root binds them without absorbing values (spec §8, D25). The
/// prover runs it over the forms' decodings, so both roles draw the same
/// selectors; `combine` is the value type's linear combination.
///
/// # Errors
///
/// The errors of the clear stage 8's leaf geometry.
#[cfg(all(feature = "akita", feature = "zk"))]
#[cfg_attr(
    feature = "field-inline",
    expect(
        clippy::too_many_arguments,
        reason = "one argument per precommitted object kind, the field increments included"
    )
)]
fn whiteout_groups<F, V, C, Ch>(
    plan: &WhiteoutOpeningPlan<C>,
    chunk_width: usize,
    trace_commitment: &C,
    leaves: &BTreeMap<JoltCommittedPolynomial, Leaf<F, V>>,
    fused_inc: Leaf<F, V>,
    #[cfg(feature = "field-inline")] field_inc: Leaf<F, V>,
    combine: &LinearCombination<'_, F, V>,
    transcript: &mut Ch,
) -> Result<Vec<WhiteoutGroup<F, V, C>>, VerifierError>
where
    F: JoltField,
    V: Clone,
    C: Clone,
    Ch: FieldTranscript<F>,
{
    // An advice leaf and its object (its commitment) are present together.
    for polynomial in [
        JoltCommittedPolynomial::UntrustedAdvice,
        JoltCommittedPolynomial::TrustedAdvice,
    ] {
        let committed = plan.objects.iter().any(|object| {
            matches!(object, WhiteoutObject::Packed { plan, .. } if plan.packing().slot_index(&polynomial).is_some())
        });
        if leaves.contains_key(&polynomial) != committed {
            return Err(batch_failed(format!(
                "{polynomial:?} final claim and commitment must be both present or both absent"
            )));
        }
    }
    let mut groups = Vec::with_capacity(plan.objects.len().saturating_add(1));
    let mut fused_inc = Some(fused_inc);
    #[cfg(feature = "field-inline")]
    let mut field_inc = Some(field_inc);
    for object in &plan.objects {
        let (point, value) = match object {
            WhiteoutObject::Packed { plan, .. } => {
                let packed = plan
                    .packed_leaves(|id| {
                        leaves
                            .get(&id)
                            .map(|leaf| (leaf.point.as_slice(), &leaf.value))
                    })
                    .map_err(batch_failed)?;
                let selection = plan
                    .packing()
                    .select(
                        &plan.layout_digest(),
                        &packed.point,
                        PackedEvaluations::Masked {
                            count: packed.slots.len(),
                        },
                        transcript,
                    )
                    .map_err(batch_failed)?;
                let terms = selection
                    .weights
                    .iter()
                    .zip(&packed.slots)
                    .map(|(weight, (scale, value))| (*weight * *scale, *value))
                    .collect::<Vec<_>>();
                (selection.point, combine(&terms))
            }
            #[cfg(feature = "field-inline")]
            WhiteoutObject::FieldInc { layout, .. } => {
                let leaf = field_inc.take().ok_or_else(|| {
                    batch_failed("a Whiteout opening has one field-increment group")
                })?;
                (
                    layout.opening_point(&leaf.point).map_err(batch_failed)?,
                    leaf.value,
                )
            }
            WhiteoutObject::FusedInc { layout, .. } => {
                let leaf = fused_inc.take().ok_or_else(|| {
                    batch_failed("a Whiteout opening has one RV64-increment group")
                })?;
                (
                    layout.opening_point(&leaf.point).map_err(batch_failed)?,
                    leaf.value,
                )
            }
        };
        groups.push(WhiteoutGroup {
            commitment: object.commitment().clone(),
            point,
            values: vec![value],
        });
    }
    let (point, values) = one_hot_trace_group(&plan.trace, chunk_width, leaves)?;
    groups.push(WhiteoutGroup {
        commitment: trace_commitment.clone(),
        point,
        values,
    });
    Ok(groups)
}

/// Whiteout's stage 8 on the verifier's side: [`whiteout_groups`] over the
/// residual's opening forms, each column's form the masked opening of its
/// final claim, for the masked Akita opening.
///
/// # Errors
///
/// The errors of the clear stage 8's leaf geometry.
#[cfg(all(feature = "akita", feature = "zk"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the stage outputs are passed separately by the verifier driver"
)]
pub fn masked_group_claims<F, C, Ch>(
    plan: &WhiteoutOpeningPlan<C>,
    one_hot_config: JoltOneHotConfig,
    one_hot_trace_commitment: &C,
    schedule: &PrecommittedSchedule,
    stage4_staged_points: &RamValCheckStagedOpenings<Vec<F>>,
    stage6b_points: &Stage6bOutputPoints<F>,
    stage7_points: &Stage7OutputPoints<F>,
    residual: &JoltResidual<F>,
    transcript: &mut Ch,
) -> Result<Vec<MaskedGroupClaim<F, C>>, VerifierError>
where
    F: JoltField,
    C: Clone,
    Ch: FieldTranscript<F>,
{
    let leaves = leaves(
        stage7_points,
        precommitted_final_openings(schedule, stage7_points, stage6b_points, None)?,
        &mut |kind| {
            let point = match kind {
                JoltAdviceKind::Untrusted => stage4_staged_points.untrusted_advice.as_ref(),
                JoltAdviceKind::Trusted => stage4_staged_points.trusted_advice.as_ref(),
            };
            point
                .map(|point| {
                    Ok(Leaf {
                        point: point.clone(),
                        value: residual.opening_form(&val_check_advice_opening(kind).into())?,
                    })
                })
                .transpose()
        },
        &mut |polynomial, _| residual.opening_form(&final_opening_id(polynomial).into()),
    )?;
    let groups = whiteout_groups(
        plan,
        one_hot_config.committed_chunk_bits(),
        one_hot_trace_commitment,
        &leaves,
        Leaf {
            point: stage6b_points.fused_inc_opening_point().to_vec(),
            value: residual.opening_form(&fused_inc_read_raf_opening().into())?,
        },
        #[cfg(feature = "field-inline")]
        Leaf {
            point: stage6b_points.field_registers_inc_opening_point().to_vec(),
            value: residual.opening_form(&field_rd_inc_reduced().into())?,
        },
        &|terms: &[(F, &AffineRow<F>)]| {
            terms.iter().fold(
                AffineRow::constant(F::zero()),
                |sum, (coefficient, form)| sum.add_scaled(form, *coefficient),
            )
        },
        transcript,
    )?;
    Ok(groups
        .into_iter()
        .map(|group| MaskedGroupClaim {
            commitment: group.commitment,
            point: group.point,
            evaluations: group.values,
        })
        .collect())
}

/// Whiteout's stage 8 on the prover's side: [`whiteout_groups`] over the
/// clear leaf claims and increment claims, the decodings of the
/// verifier's forms, on the prover's transcript.
///
/// # Errors
///
/// The errors of the clear stage 8's leaf geometry.
#[cfg(all(feature = "akita", feature = "zk"))]
pub fn clear_group_claims<F, C, Ch>(
    plan: &WhiteoutOpeningPlan<C>,
    one_hot_config: JoltOneHotConfig,
    one_hot_trace_commitment: &C,
    leaves: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    stage6b: &Stage6bClearOutput<F>,
    transcript: &mut Ch,
) -> Result<Vec<WhiteoutGroup<F, F, C>>, VerifierError>
where
    F: JoltField,
    C: Clone,
    Ch: FieldTranscript<F>,
{
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
    whiteout_groups(
        plan,
        one_hot_config.committed_chunk_bits(),
        one_hot_trace_commitment,
        &leaves,
        Leaf {
            point: stage6b.output_points.fused_inc_opening_point().to_vec(),
            value: stage6b.output_values.bytecode_read_raf.fused_inc,
        },
        #[cfg(feature = "field-inline")]
        Leaf {
            point: stage6b
                .output_points
                .field_registers_inc_opening_point()
                .to_vec(),
            value: stage6b
                .output_values
                .field_registers_inc_claim_reduction
                .rd_inc,
        },
        &|terms: &[(F, &F)]| {
            terms.iter().fold(F::zero(), |sum, (coefficient, value)| {
                sum + *coefficient * **value
            })
        },
        transcript,
    )
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
pub(crate) struct Leaf<F, V> {
    pub(crate) point: Vec<F>,
    pub(crate) value: V,
}

/// The `OneHotTrace` group's common opening point and its columns' values in
/// the plan's canonical order: each column's point is mapped to the
/// committed row-major order, and all must agree.
pub(crate) fn one_hot_trace_group<F: JoltField, V: Clone>(
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
        &stage7.output_points,
        finals,
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
        &mut |polynomial, clear| match polynomial {
            Poly::InstructionRa(index) => indexed(&hamming.instruction_ra, index, polynomial),
            Poly::BytecodeRa(index) => indexed(&hamming.bytecode_ra, index, polynomial),
            Poly::RamRa(index) => indexed(&hamming.ram_ra, index, polynomial),
            Poly::RdInc
            | Poly::RamInc
            | Poly::Reserved4(_)
            | Poly::TrustedAdvice
            | Poly::UntrustedAdvice
            | Poly::ProgramImageInit
            | Poly::ProgramBytecode => clear.ok_or_else(|| {
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
pub(crate) fn leaves<F: JoltField, V>(
    stage7_points: &Stage7OutputPoints<F>,
    precommitted: Vec<PrecommittedFinalOpening<F>>,
    advice: &mut dyn FnMut(JoltAdviceKind) -> Result<Option<Leaf<F, V>>, VerifierError>,
    value: &mut dyn FnMut(JoltCommittedPolynomial, Option<F>) -> Result<V, VerifierError>,
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
    let families: [Family<'_, F>; 3] = [
        (&hamming.instruction_ra, Poly::InstructionRa),
        (&hamming.bytecode_ra, Poly::BytecodeRa),
        (&hamming.ram_ra, Poly::RamRa),
    ];
    for (points, polynomial) in families {
        for (index, point) in points.iter().enumerate() {
            insert(polynomial(index), point, value(polynomial(index), None)?)?;
        }
    }
    for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
        if let Some(leaf) = advice(kind)? {
            let polynomial = match kind {
                JoltAdviceKind::Trusted => Poly::TrustedAdvice,
                JoltAdviceKind::Untrusted => Poly::UntrustedAdvice,
            };
            insert(polynomial, &leaf.point, leaf.value)?;
        }
    }

    for opening in precommitted {
        let leaf_value = value(opening.polynomial, opening.opening_claim)?;
        insert(opening.polynomial, &opening.point, leaf_value)?;
    }

    Ok(leaves)
}
