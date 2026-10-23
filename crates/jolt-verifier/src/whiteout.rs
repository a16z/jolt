//! Whiteout: the zero-knowledge verifier of the Akita build
//! (`specs/whiteout-jolt-integration.md`). Transcript order (§2):
//!
//! 0–1. the preamble and the commitments (`seed_transcript`);
//! 2.   the pad commitment, on the layout [`pad_layout`] derives;
//! 3.   stages 1–7, masked;
//! 4–5. stage 8: the trace group's claims as affine forms, then Akita's
//!      masked levels through the cut;
//! 6.   the auxiliary block, Akita's part first, then Jolt's;
//! 7–8. the Σ-protocol and the response proof;
//! 9.   the end of the argument string.
//!
//! The 32-byte checkpoint after the masked levels lets the prover check that
//! its replay of this verifier reached its own transcript state.

use common::jolt_device::JoltDevice;
use jolt_algebra::JoltField;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Channel, Sponge, TranscriptError, VerifierTranscript};
use jolt_zk::masking::{
    CommittedPadsView, OperandShapeDomain, PadLayout, ResidualRelation, ResidualRows,
};
use jolt_zk::whiteout::{LatticePadPayload, MaskedGroupClaim, MaskedGroupOpening};

use crate::config::{validate_proof_config, JOLT_VERIFIER_CONFIG};
use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::sites::{PADS, STAGE8, WHITEOUT};
use crate::stages::stage8::akita::masked_trace_claim;
use crate::stages::stage8::{OneHotTraceCommitmentMetadata, OneHotTraceSetupMetadata};
use crate::stages::zk::residual::JoltResidual;
use crate::verifier::{
    jolt_protocol_id, seed_transcript, verify_stages_1_to_7, JoltSponge, SeededTranscript,
    StageOutputs, JOLT_SESSION,
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

/// A Whiteout proof through stage 7 and stage 8's claims: the masked
/// opening's shape, the pad view, the stage outputs' residual relation, and
/// the trace group's masked claims.
pub struct WhiteoutStages<F: JoltField, PCS: WhiteoutScheme<F>>
where
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
{
    pub shape: <PCS as MaskedGroupOpening<F>>::Shape,
    pub pads: CommittedPadsView<LatticePadPayload<F>>,
    pub residual: JoltResidual<F>,
    pub claim: MaskedGroupClaim<F, PCS::Output>,
}

/// Stage 8's masked claims over `residual`, from the stage outputs.
fn stage8_claim<F, PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    seeded: &SeededTranscript<PCS::Output>,
    trusted_advice_commitment: Option<&PCS::Output>,
    stages: &StageOutputs<F>,
    residual: &JoltResidual<F>,
) -> Result<MaskedGroupClaim<F, PCS::Output>, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
{
    masked_trace_claim::<F, PCS, VC>(
        &seeded.formula_dimensions,
        seeded.checked.one_hot_config,
        preprocessing,
        &seeded.commitments.one_hot_trace,
        seeded.commitments.untrusted_advice.as_ref(),
        trusted_advice_commitment,
        &seeded.checked.precommitted,
        stages.stage6b.output_points(),
        &stages.stage7.zk()?.output_points,
        residual,
    )
}

/// The pad layout of a Whiteout proof, from the masked verifier's own replay
/// over the seeded prefix followed by zero bytes: stages 1–7 and Akita's
/// masked levels count the message pads; then Akita's checks, then Jolt's,
/// allocate the auxiliary slots. Prover and verifier both call it.
///
/// # Errors
///
/// [`VerifierError`] if the seeded prefix is malformed or the replay fails.
pub fn pad_layout<F, PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    seeded_prefix: &[u8],
) -> Result<PadLayout, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
{
    // As the homomorphic build's replay: 1 MiB covers the masked messages
    // of every realistic trace, doubled up to 16 MiB.
    let mut padding = 1usize << 20;
    loop {
        let mut bytes = Vec::with_capacity(seeded_prefix.len().saturating_add(padding));
        bytes.extend_from_slice(seeded_prefix);
        bytes.resize(seeded_prefix.len().saturating_add(padding), 0);
        let mut transcript = VerifierTranscript::<JoltSponge>::new(
            &jolt_protocol_id::<JoltSponge>(),
            JOLT_SESSION,
            &bytes,
        );
        let seeded = seed_transcript(
            preprocessing,
            public_io,
            trusted_advice_commitment,
            &mut transcript,
        )?;
        let shape = PCS::shape(
            &preprocessing.pcs_setup,
            &[&seeded.commitments.one_hot_trace],
        )?;
        let mut counter = CommittedPadsView::counting();
        let replayed = verify_stages_1_to_7::<F, PCS, VC, JoltSponge, ()>(
            preprocessing,
            &seeded.checked,
            &seeded.formula_dimensions,
            &mut transcript,
            Some(&mut counter),
        )
        .and_then(|stages| {
            let residual = stages.residual(&seeded.checked, preprocessing)?;
            let claim = stage8_claim::<F, PCS, VC>(
                preprocessing,
                &seeded,
                trusted_advice_commitment,
                &stages,
                &residual,
            )?;
            let prefix = PCS::count_masked(&shape, vec![claim], &mut counter, &mut transcript)
                .map_err(VerifierError::Masking)?;
            Ok((residual, prefix))
        });
        match replayed {
            Ok((residual, prefix)) => {
                let mut domain = OperandShapeDomain::new(counter.used());
                PCS::prefix_shape(&prefix, &mut domain)?;
                residual.shape(&mut domain)?;
                return Ok(domain.layout()?);
            }
            Err(VerifierError::Transcript(TranscriptError::Truncated)) if padding < 1 << 24 => {
                padding <<= 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Runs a Whiteout proof's seeding messages, pad commitment and stages 1–7,
/// and builds stage 8's masked claims. `layout` is the derived pad layout
/// when the caller has it.
///
/// # Errors
///
/// [`VerifierError`] if any step rejects.
pub fn whiteout_stages<F, PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut VerifierTranscript<'_, H>,
    layout: Option<&PadLayout>,
) -> Result<WhiteoutStages<F, PCS>, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    let seeded = seed_transcript(
        preprocessing,
        public_io,
        trusted_advice_commitment,
        transcript,
    )?;
    let layout = match layout {
        Some(layout) => layout.clone(),
        None => pad_layout::<F, PCS, VC>(
            preprocessing,
            public_io,
            trusted_advice_commitment,
            transcript.consumed(),
        )?,
    };
    let shape = PCS::shape(
        &preprocessing.pcs_setup,
        &[&seeded.commitments.one_hot_trace],
    )?;
    transcript.site(PADS);
    let mut pads = PCS::receive_pads(&shape, layout, transcript)?;
    let stages = verify_stages_1_to_7::<F, PCS, VC, H, LatticePadPayload<F>>(
        preprocessing,
        &seeded.checked,
        &seeded.formula_dimensions,
        transcript,
        Some(&mut pads),
    )?;
    let residual = stages.residual(&seeded.checked, preprocessing)?;
    transcript.site(STAGE8);
    let claim = stage8_claim::<F, PCS, VC>(
        preprocessing,
        &seeded,
        trusted_advice_commitment,
        &stages,
        &residual,
    )?;
    Ok(WhiteoutStages {
        shape,
        pads,
        residual,
        claim,
    })
}

/// The masked levels through the cut, then the checkpoint.
///
/// # Errors
///
/// [`VerifierError`] if a masked level rejects.
pub fn whiteout_prefix<F, PCS, H>(
    stages: &mut WhiteoutStages<F, PCS>,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<(<PCS as MaskedGroupOpening<F>>::Prefix, [u8; 32]), VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    H: Sponge,
{
    let prefix = PCS::verify_masked(
        &stages.shape,
        vec![stages.claim.clone()],
        &mut stages.pads,
        transcript,
    )?;
    Ok((prefix, transcript.challenge_bytes()))
}

/// The auxiliary block (Akita's slots first, then Jolt's) and the residual
/// relation it completes: Akita's masked-level rows and cut row, then
/// stages 1–7's rows and equalities.
///
/// # Errors
///
/// [`VerifierError`] if the block is truncated or a row cannot be built.
pub fn whiteout_relation<F, PCS, H>(
    stages: &mut WhiteoutStages<F, PCS>,
    prefix: &<PCS as MaskedGroupOpening<F>>::Prefix,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<ResidualRelation<F>, VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    H: Sponge,
{
    use jolt_transcript::PrivateChannel;

    let layout = stages.pads.layout().clone();
    let mut akita = OperandShapeDomain::new(layout.auxiliary.start);
    PCS::prefix_shape(prefix, &mut akita)?;
    let akita_slots = akita.auxiliary_count();
    transcript.site(WHITEOUT);
    let mut masked = vec![F::zero(); layout.auxiliary.len()];
    let _ = stages
        .pads
        .channel(transcript)
        .exchange_private_base(&mut masked)
        .map_err(VerifierError::Masking)?;
    stages.pads.finish()?;
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
pub fn verify<F, PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + OneHotTraceCommitmentMetadata,
    PCS::VerifierSetup: OneHotTraceSetupMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    validate_proof_config(&JOLT_VERIFIER_CONFIG, proof.protocol)?;
    let mut transcript =
        VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, &proof.narg);
    let mut stages = whiteout_stages::<F, PCS, VC, H>(
        preprocessing,
        public_io,
        trusted_advice_commitment,
        &mut transcript,
        None,
    )?;
    let (prefix, _checkpoint) = whiteout_prefix(&mut stages, &mut transcript)?;
    let relation = whiteout_relation(&mut stages, &prefix, &mut transcript)?;
    PCS::verify_tail(
        &stages.shape,
        &prefix,
        &relation,
        &stages.pads,
        &mut transcript,
    )?;
    transcript.finish()?;
    Ok(())
}
