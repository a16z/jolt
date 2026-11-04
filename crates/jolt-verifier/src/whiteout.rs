//! Whiteout: the zero-knowledge verifier of the Akita build
//! (`specs/whiteout-jolt-integration.md`). Transcript order (§2):
//!
//! 0–1. the statement and the commitments (`validate_and_seed_transcript`);
//! 2.   the pad commitment, on the layout [`pad_layout`] derives (NARG);
//! 3.   stages 1–7, masked: every private message is m̂ from the proof's
//!      masked tape, absorbed where the clear verifier absorbs its value;
//! 4–5. stage 8: each precommitted group's and the trace group's claims as
//!      affine forms, then Akita's masked levels through the cut (NARG);
//! 6.   the auxiliary block from the masked tape, Akita's part first, then
//!      Jolt's;
//! 7–8. the Σ-protocol and the response proof (NARG);
//! 9.   the end of the argument string and of the masked tape.

use std::borrow::BorrowMut;

use common::jolt_device::JoltDevice;
use jolt_algebra::JoltField;
use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::{
    verifier_transcript, ChallengeCodec, Encoding, FieldTranscript, Shake128, ShapeTranscript,
    VerifierInit, VerifierState,
};
use jolt_zk::masking::{
    CommittedPadsView, MaskedReader, MaskedSession, OperandShapeDomain, PadLayout,
    ResidualRelation, ResidualRows,
};
use jolt_zk::whiteout::{
    LatticePadPayload, MaskedGroupClaim, MaskedGroupOpening, MaskedOpeningShape,
};

use crate::num;
use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::stages::build_formula_dimensions;
use crate::stages::pow::{check_pow, PowSite, StagePowNonces};
use crate::stages::stage8::akita::{
    masked_group_claims, whiteout_opening_plan, WhiteoutOpeningPlan,
};
use crate::stages::stage8::{OneHotTraceCommitmentMetadata, OneHotTraceSetupMetadata};
use crate::stages::zk::residual::JoltResidual;
use crate::verifier::{
    validate_and_seed_transcript, verify_stages_1_to_7, CheckedInputs, StageOutputs, JOLT_SESSION,
};
use crate::VerifierError;

/// The masked opening seam a Whiteout PCS implements over Jolt's field.
pub trait WhiteoutScheme<F: JoltField>:
    CommitmentScheme<Field = F>
    + MaskedGroupOpening<F, Setup = Self::VerifierSetup, GroupCommitment = Self::Output>
where
    Self::Output: Clone + OneHotTraceCommitmentMetadata,
    Self::VerifierSetup: OneHotTraceSetupMetadata,
{
}

impl<F: JoltField, PCS> WhiteoutScheme<F> for PCS
where
    PCS: CommitmentScheme<Field = F>
        + MaskedGroupOpening<F, Setup = PCS::VerifierSetup, GroupCommitment = PCS::Output>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
{
}

/// The Akita state a Whiteout PCS runs its masked levels and tail on.
pub type WhiteoutState<'a, F, PCS> = VerifierState<'a, <PCS as MaskedGroupOpening<F>>::Sponge>;

/// A Whiteout proof's public frame: the stage formula dimensions, the final
/// opening's plan and the masked opening's shape, all from the statement and
/// the proof's commitments. The commitments' metadata is checked against the
/// plan before any shape is built.
pub struct WhiteoutFrame<F: JoltField, PCS: WhiteoutScheme<F>>
where
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
{
    pub formula_dimensions: JoltFormulaDimensions,
    pub plan: WhiteoutOpeningPlan<PCS::Output>,
    pub shape: <PCS as MaskedGroupOpening<F>>::Shape,
}

/// Builds [`WhiteoutFrame`] for `proof`.
///
/// # Errors
///
/// [`VerifierError`] if a required commitment is missing or its metadata
/// differs from the plan.
pub fn whiteout_frame<F, PCS, VC, ZkProof, O>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof, O>,
    checked: &CheckedInputs,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<WhiteoutFrame<F, PCS>, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
{
    let formula_dimensions = build_formula_dimensions(
        proof,
        preprocessing,
        checked,
        num::ilog2(checked.trace_length),
        JoltRelationId::InstructionReadRaf,
    )?;
    let fused_inc =
        proof
            .fused_inc_commitment
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "fused_inc_commitment",
            })?;
    #[cfg(feature = "field-inline")]
    let field_inc =
        proof
            .field_inc_commitment
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "field_inc_commitment",
            })?;
    let plan = whiteout_opening_plan::<PCS, VC>(
        &formula_dimensions,
        proof.one_hot_config,
        preprocessing,
        &proof.commitments,
        fused_inc,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
        #[cfg(feature = "field-inline")]
        field_inc,
    )?;
    let shape = PCS::shape(&preprocessing.pcs_setup, &plan.shape())?;
    Ok(WhiteoutFrame {
        formula_dimensions,
        plan,
        shape,
    })
}

/// Stage 8's masked claims over `residual`, from the stage outputs: the
/// proof of work before the packing selectors when there are any (as on the
/// clear Akita path), then each group's claims as affine forms.
#[expect(clippy::too_many_arguments, reason = "the stage outputs and the frame")]
fn stage8_claims<F, C, T>(
    checked: &CheckedInputs,
    proof_one_hot: jolt_claims::protocols::jolt::JoltOneHotConfig,
    pow_nonces: &StagePowNonces,
    trace_commitment: &C,
    plan: &WhiteoutOpeningPlan<C>,
    stages: &StageOutputs<F>,
    residual: &JoltResidual<F>,
    transcript: &mut T,
) -> Result<Vec<MaskedGroupClaim<F, C>>, VerifierError>
where
    F: JoltField,
    C: Clone,
    T: FieldTranscript<F>,
{
    if plan.has_selectors() {
        check_pow(PowSite::Stage8, pow_nonces, transcript)?;
    } else {
        pow_nonces.require_absent(PowSite::Stage8)?;
    }
    masked_group_claims(
        plan,
        proof_one_hot,
        trace_commitment,
        &checked.precommitted,
        stages.stage4.output_points(),
        stages.stage6b.output_points(),
        &stages.stage7.zk()?.output_points,
        residual,
        transcript,
    )
}

/// A pad layout [`pad_layout`] derived from a statement and its proof's
/// shape, with the masked opening's shape it was derived on. Only
/// [`pad_layout`] builds one, so a caller that reuses a derived layout (the
/// prover) cannot inject another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedPadLayout {
    layout: PadLayout,
    shape: MaskedOpeningShape,
}

impl DerivedPadLayout {
    /// The layout.
    pub fn layout(&self) -> &PadLayout {
        &self.layout
    }

    /// The masked opening's groups, from the statement.
    pub fn shape(&self) -> &MaskedOpeningShape {
        &self.shape
    }
}

/// The NARG bytes a pad-layout replay of Akita's masked levels starts with;
/// doubled on failure up to [`MAX_LAYOUT_ZEROS`].
const LAYOUT_ZEROS: usize = 1 << 20;
const MAX_LAYOUT_ZEROS: usize = 1 << 26;

/// The pad layout of a Whiteout proof, from the masked verifier's own replay
/// over `proof`'s shape: stages 1–7 read zeros from an unbounded masked tape
/// on a [`ShapeTranscript`] (which prices away every proof of work), Akita's
/// masked levels count over zero NARG bytes (their own counting replay skips
/// grinding), then Akita's checks, then Jolt's, allocate the auxiliary slots.
/// Only the statement and the proof's commitments and configuration matter;
/// prover (on its proof shell) and verifier both call it, so the schedule has
/// one owner.
///
/// # Errors
///
/// [`VerifierError`] if the replay fails.
pub fn pad_layout<F, PCS, VC, ZkProof, O>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof, O>,
    checked: &CheckedInputs,
    frame: &WhiteoutFrame<F, PCS>,
) -> Result<DerivedPadLayout, VerifierError>
where
    F: JoltField + ChallengeCodec,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
{
    let _span = tracing::info_span!("whiteout_pad_layout").entered();
    let mut transcript = ShapeTranscript::<Shake128>::new();
    let mut counter = MaskedSession::new(CommittedPadsView::counting(), MaskedReader::zeros());
    let stages = verify_stages_1_to_7::<F, PCS, VC, _, _, _, ()>(
        preprocessing,
        proof,
        checked,
        &frame.formula_dimensions,
        &mut transcript,
        Some(&mut counter),
    )?;
    let residual = stages.residual(checked, preprocessing, proof)?;
    let claims = stage8_claims(
        checked,
        proof.one_hot_config,
        &proof.stages.pow_nonces,
        &proof.commitments,
        &frame.plan,
        &stages,
        &residual,
        &mut transcript,
    )?;
    let base = counter.pads.used();
    let mut zeros = LAYOUT_ZEROS;
    let (prefix, view) = loop {
        let bytes = vec![0u8; zeros];
        let mut state = verifier_transcript::<<PCS as MaskedGroupOpening<F>>::Sponge>(
            &JOLT_SESSION,
            &0u64,
            &bytes,
        );
        let mut view = CommittedPadsView::counting();
        let _ = view.take(base)?;
        match PCS::count_masked(&frame.shape, claims.clone(), &mut view, &mut state) {
            Ok(prefix) => break (prefix, view),
            Err(_) if zeros < MAX_LAYOUT_ZEROS => zeros <<= 1,
            Err(error) => return Err(VerifierError::Masking(error)),
        }
    };
    let mut domain = OperandShapeDomain::new(view.used());
    PCS::prefix_shape(&prefix, &mut domain)?;
    Ok(DerivedPadLayout {
        layout: residual.shape(domain)?.layout()?,
        shape: frame.plan.shape(),
    })
}

/// A Whiteout proof through stage 7 and stage 8's claims: the frame, the
/// masked session (the pad commitment's view and the proof's masked tape),
/// the stage outputs' residual relation, and every group's masked claims
/// (the precommitted groups in canonical order, then the trace group).
pub struct WhiteoutStages<'m, F: JoltField, PCS: WhiteoutScheme<F>>
where
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
{
    pub frame: WhiteoutFrame<F, PCS>,
    pub session: MaskedSession<'m, F, LatticePadPayload<F>>,
    pub residual: JoltResidual<F>,
    pub claims: Vec<MaskedGroupClaim<F, PCS::Output>>,
}

/// Runs a seeded Whiteout transcript's pad commitment and stages 1–7, and
/// builds stage 8's masked claims. `layout` is the derived pad layout when
/// the caller has it; `None` derives it here.
///
/// # Errors
///
/// [`VerifierError`] if any step rejects.
pub fn whiteout_stages<'a, 'm, F, PCS, VC, T, O>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &'m JoltProof<PCS, VC, (), O>,
    checked: &CheckedInputs,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    layout: Option<&DerivedPadLayout>,
) -> Result<WhiteoutStages<'m, F, PCS>, VerifierError>
where
    F: JoltField + ChallengeCodec,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    T: FieldTranscript<F> + BorrowMut<WhiteoutState<'a, F, PCS>>,
{
    let frame = whiteout_frame(preprocessing, proof, checked, trusted_advice_commitment)?;
    let layout = match layout {
        Some(layout) => layout.clone(),
        None => pad_layout(preprocessing, proof, checked, &frame)?,
    }
    .layout;
    let pads = PCS::receive_pads(&frame.shape, layout, transcript.borrow_mut())?;
    let (tape, ()) = proof.masked_payload()?;
    let mut session = MaskedSession::new(pads, MaskedReader::new(tape));
    let stages = verify_stages_1_to_7::<F, PCS, VC, T, _, _, LatticePadPayload<F>>(
        preprocessing,
        proof,
        checked,
        &frame.formula_dimensions,
        transcript,
        Some(&mut session),
    )?;
    let residual = stages.residual(checked, preprocessing, proof)?;
    let claims = stage8_claims(
        checked,
        proof.one_hot_config,
        &proof.stages.pow_nonces,
        &proof.commitments,
        &frame.plan,
        &stages,
        &residual,
        transcript,
    )?;
    Ok(WhiteoutStages {
        frame,
        session,
        residual,
        claims,
    })
}

/// The masked levels through the cut.
///
/// # Errors
///
/// [`VerifierError`] if a masked level rejects.
pub fn whiteout_prefix<'a, F, PCS, T>(
    stages: &mut WhiteoutStages<'_, F, PCS>,
    transcript: &mut T,
) -> Result<<PCS as MaskedGroupOpening<F>>::Prefix, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    T: BorrowMut<WhiteoutState<'a, F, PCS>>,
{
    Ok(PCS::verify_masked(
        &stages.frame.shape,
        stages.claims.clone(),
        &mut stages.session.pads,
        transcript.borrow_mut(),
    )?)
}

/// The auxiliary block (Akita's slots first, then Jolt's), read from the
/// masked tape, and the residual relation it completes: Akita's masked-level
/// rows and cut row, then stages 1–7's rows and equalities. Every pad and
/// every masked message must then be used.
///
/// # Errors
///
/// [`VerifierError`] if the block is truncated, a pad or masked message is
/// left over, or a row cannot be built.
pub fn whiteout_relation<F, PCS, T>(
    stages: &mut WhiteoutStages<'_, F, PCS>,
    prefix: &<PCS as MaskedGroupOpening<F>>::Prefix,
    transcript: &mut T,
) -> Result<ResidualRelation<F>, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    T: FieldTranscript<F>,
{
    let layout = stages.session.pads.layout().clone();
    let mut akita = OperandShapeDomain::new(layout.auxiliary.start);
    PCS::prefix_shape(prefix, &mut akita)?;
    let akita_slots = akita.auxiliary_count();
    let (masked, _) = stages
        .session
        .channel(transcript)
        .receive_private(layout.auxiliary.len())?;
    stages.session.finish()?;
    let (akita_auxiliary, jolt_auxiliary) =
        masked
            .split_at_checked(akita_slots)
            .ok_or(VerifierError::ResidualConstructionFailed {
                reason: "the auxiliary block is shorter than Akita's part".to_string(),
            })?;
    // `akita_slots` is at most the block's length, so this stays inside
    // the layout.
    let jolt_first = layout.auxiliary.start.saturating_add(akita_slots);
    let mut relation = ResidualRelation::new(layout.clone(), 1);
    PCS::append_prefix_rows(prefix, akita_auxiliary, &mut relation)?;
    stages
        .residual
        .rows_from(jolt_auxiliary, jolt_first)
        .append_rows(&mut relation)?;
    Ok(relation)
}

/// Verifies a Whiteout proof.
///
/// # Errors
///
/// [`VerifierError`] if any step rejects.
pub fn verify<'a, F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &'a JoltProof<PCS, VC, (), ()>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: JoltField + ChallengeCodec,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + Encoding + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Encoding,
    T: FieldTranscript<F>
        + VerifierInit<'a>
        + BorrowMut<WhiteoutState<'a, F, PCS>>
        + Into<WhiteoutState<'a, F, PCS>>,
{
    let (checked, mut transcript) = validate_and_seed_transcript(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        |statement| T::init(&JOLT_SESSION, statement, &proof.narg),
    )?;
    verify_transcript::<F, PCS, VC, T>(
        preprocessing,
        proof,
        &checked,
        trusted_advice_commitment,
        &mut transcript,
        None,
    )?;
    let state: WhiteoutState<'a, F, PCS> = transcript.into();
    state
        .check_eof()
        .map_err(|_| VerifierError::NargNotConsumed)
}

/// [`verify`]'s body after the statement and commitments, over a seeded
/// transcript the caller owns and finishes (an audit transcript's event log
/// is the masked-site census's input). `layout` is the derived pad layout
/// when the caller has it.
///
/// # Errors
///
/// [`VerifierError`] if any step rejects.
pub fn verify_transcript<'a, F, PCS, VC, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, (), ()>,
    checked: &CheckedInputs,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    layout: Option<&DerivedPadLayout>,
) -> Result<(), VerifierError>
where
    F: JoltField + ChallengeCodec,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    T: FieldTranscript<F> + BorrowMut<WhiteoutState<'a, F, PCS>>,
{
    let mut stages = tracing::info_span!("whiteout_verify_stages").in_scope(|| {
        whiteout_stages::<F, PCS, VC, T, _>(
            preprocessing,
            proof,
            checked,
            trusted_advice_commitment,
            transcript,
            layout,
        )
    })?;
    let prefix = tracing::info_span!("whiteout_verify_prefix")
        .in_scope(|| whiteout_prefix::<F, PCS, T>(&mut stages, transcript))?;
    let relation = tracing::info_span!("whiteout_verify_relation")
        .in_scope(|| whiteout_relation(&mut stages, &prefix, transcript))?;
    PCS::verify_tail(
        &stages.frame.shape,
        &prefix,
        &relation,
        &stages.session.pads,
        transcript.borrow_mut(),
    )?;
    Ok(())
}
