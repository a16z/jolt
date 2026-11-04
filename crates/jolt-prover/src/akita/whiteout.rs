//! The Whiteout prover: Jolt's zero-knowledge proof over Akita
//! (`specs/whiteout-jolt-integration.md`), in the verifier's transcript order
//! (`jolt_verifier::whiteout`). A fold overflow or exhausted Σ attempts rerun
//! the whole proof with fresh randomness, at most [`WHITEOUT_ATTEMPTS`]
//! times (§6).

use std::borrow::BorrowMut;

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaProverTranscript, TraceOneHotCommitment};
use jolt_algebra::{CanonicalBytes, JoltField};
use jolt_crypto::VectorCommitment;
use jolt_openings::{
    GroupCommitmentMetadata, GroupOpeningClaim, GroupSetupMetadata, TaggedGroupOpeningClaim,
    TransparentObjectSetup,
};
use jolt_transcript::{
    verifier_transcript, ChallengeCodec, Encoding, FieldTranscript, ProverInit, ProverTranscript,
    UniformBytes, VerifierState,
};
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs};
use jolt_verifier::stages::pow::{PowSite, StagePowNonces};
use jolt_verifier::stages::stage8::akita::{clear_group_claims, leaf_claims};
use jolt_verifier::whiteout::{
    pad_layout, whiteout_frame, whiteout_prefix, whiteout_relation, whiteout_stages,
    DerivedPadLayout, WhiteoutScheme,
};
use jolt_verifier::{validate_and_seed_transcript, CheckedInputs, JOLT_SESSION};
use jolt_zk::masking::MaskedMessages;
#[cfg(feature = "prover-fixtures")]
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::MaskedGroupOpening;
use rand_core::{CryptoRng, OsRng, RngCore};

use super::stage0::prove_stage0;
use super::witness::AdviceObject;
use crate::boundary::finish_stage;
use crate::stages::grind_pow;
use crate::stages::stage1::prove_stage1;
use crate::stages::stage2::prove_stage2;
use crate::stages::stage3::prove_stage3;
use crate::stages::stage4::prove_stage4;
use crate::stages::stage5::prove_stage5;
use crate::stages::stage6a::prove_stage6a;
use crate::stages::stage6b::prove_stage6b;
use crate::stages::stage7::prove_stage7;
use crate::{JoltProverPreprocessing, ProofMode, ProverConfig, ProverError};
use jolt_kernels::{JoltBackend, KernelContext};

/// The most attempts a Whiteout proof makes (§6): the first run and two
/// reruns.
const WHITEOUT_ATTEMPTS: usize = 3;

/// A Whiteout proof: the masked tape and no clear opening proof; Akita's
/// masked opening, the pads, Σ and the response proof are in the NARG string.
pub type WhiteoutProof<PCS, VC> = JoltProof<PCS, VC, (), ()>;

/// Logs the argument string's length at the end of `region`, so a profile
/// log splits the proof's bytes by region (`specs/whiteout-v6-costs.md`).
fn mark_region<T: ProverTranscript>(region: &'static str, transcript: &T, tape: usize) {
    tracing::info!(
        region,
        narg_bytes = transcript.narg_string().len(),
        masked_values = tape,
        "argument string"
    );
}

/// See [`super::prove`].
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, T>(
    mut backend: KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
) -> Result<WhiteoutProof<PCS, VC>, ProverError<F>>
where
    F: JoltField + CanonicalBytes + Encoding + ChallengeCodec,
    PCS: WhiteoutScheme<F> + TraceOneHotCommitment + TransparentObjectSetup,
    <PCS as MaskedGroupOpening<F>>::Sponge: UniformBytes,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + Encoding + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Encoding,
    T: FieldTranscript<F> + ProverInit + BorrowMut<AkitaProverTranscript>,
{
    let mut attempt = 1;
    loop {
        let result = prove_once::<F, PCS, VC, T, _>(
            &mut backend,
            preprocessing,
            config,
            trusted_advice,
            public_io,
            &mut OsRng,
        );
        match &result {
            Err(ProverError::Zk(error)) if error.is_rerunnable() && attempt < WHITEOUT_ATTEMPTS => {
                tracing::info!(attempt, %error, "Whiteout attempt aborted; rerunning");
                attempt += 1;
                backend.restart();
            }
            Ok(_) | Err(_) => {
                if result.is_ok() {
                    tracing::info!(attempt, "Whiteout attempt accepted");
                }
                backend.finish();
                return result;
            }
        }
    }
}

/// The proof assembled so far: this attempt's commitments, configuration,
/// stage nonces and masked tape, and its argument string `narg`. The
/// verifier replays exactly this.
#[expect(clippy::too_many_arguments, reason = "one argument per proof field")]
fn shell<PCS, VC>(
    checked: &CheckedInputs,
    config: &ProverConfig,
    commitment: &PCS::Output,
    fused_inc: &PCS::Output,
    untrusted_advice: Option<&PCS::Output>,
    #[cfg(feature = "field-inline")] field_inc: &PCS::Output,
    pow_nonces: StagePowNonces,
    tape: MaskedMessages<PCS::Field>,
    narg: Vec<u8>,
) -> WhiteoutProof<PCS, VC>
where
    PCS: jolt_openings::CommitmentScheme,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = PCS::Field>,
{
    JoltProof {
        protocol: JoltProtocolConfig::for_zk(checked.zk),
        commitments: commitment.clone(),
        fused_inc_commitment: Some(fused_inc.clone()),
        #[cfg(feature = "field-inline")]
        field_inc_commitment: Some(field_inc.clone()),
        stages: JoltStageProofs::masked(pow_nonces),
        joint_opening_proof: (),
        untrusted_advice_commitment: untrusted_advice.cloned(),
        claims: JoltProofClaims::Zk {
            masked: tape,
            zk_proof: (),
        },
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
        narg,
    }
}

/// One Whiteout attempt from stage 0. An aborted attempt's transcript is
/// dropped with it.
#[expect(
    clippy::too_many_lines,
    reason = "the proof's rows in transcript order"
)]
fn prove_once<F, PCS, VC, T, R>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
    rng: &mut R,
) -> Result<WhiteoutProof<PCS, VC>, ProverError<F>>
where
    F: JoltField + CanonicalBytes + Encoding + ChallengeCodec,
    PCS: WhiteoutScheme<F> + TraceOneHotCommitment + TransparentObjectSetup,
    <PCS as MaskedGroupOpening<F>>::Sponge: UniformBytes,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + Encoding + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Encoding,
    T: FieldTranscript<F> + ProverInit + BorrowMut<AkitaProverTranscript>,
    R: RngCore + CryptoRng + Send,
{
    let trusted_commitment = trusted_advice.map(|advice| &advice.commitment);
    let stage0 =
        prove_stage0::<F, PCS, VC, T>(backend, preprocessing, config, trusted_advice, public_io)?;
    let log_t = config.trace_length.ilog2() as usize;
    finish_stage("stage0", log_t, backend.session(), &());
    let checked = stage0.checked;
    let mut transcript = stage0.transcript;
    let shell_of = |pow_nonces: StagePowNonces, tape: MaskedMessages<F>, narg: Vec<u8>| {
        shell::<PCS, VC>(
            &checked,
            config,
            &stage0.commitment,
            &stage0.fused_inc.commitment,
            stage0
                .untrusted_advice
                .as_ref()
                .map(|object| &object.commitment),
            #[cfg(feature = "field-inline")]
            &stage0.field_inc.commitment,
            pow_nonces,
            tape,
            narg,
        )
    };
    mark_region("stage0", &transcript, 0);

    let empty = shell_of(
        StagePowNonces::default(),
        MaskedMessages::default(),
        Vec::new(),
    );
    let frame = whiteout_frame(
        &preprocessing.verifier,
        &empty,
        &checked,
        trusted_commitment,
    )?;
    let layout: DerivedPadLayout = tracing::info_span!("pad_layout")
        .in_scope(|| pad_layout(&preprocessing.verifier, &empty, &checked, &frame))?;
    let pads = tracing::info_span!("commit_pads").in_scope(|| {
        PCS::whiteout_commit_pads(
            &preprocessing.pcs_setup,
            layout.shape(),
            layout.layout().clone(),
            transcript.borrow_mut(),
            rng,
        )
    })?;
    mark_region("pads", &transcript, 0);
    let mut mode = ProofMode::<VC>::masked(pads);

    let stage1 = prove_stage1::<F, PCS, VC, T>(backend, &mut mode, log_t, &mut transcript)?;
    finish_stage("stage1", log_t, backend.session(), &stage1.clear_output);
    let stage2 = prove_stage2::<F, PCS, VC, T>(
        backend,
        &mut mode,
        config,
        public_io,
        &stage1.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage2", log_t, backend.session(), &stage2.clear_output);
    let stage3 = prove_stage3::<F, PCS, VC, T>(
        backend,
        &mut mode,
        config,
        &stage1.clear_output,
        &stage2.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage3", log_t, backend.session(), &stage3.clear_output);
    let stage4 = prove_stage4::<F, PCS, VC, T>(
        backend,
        &mut mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage3.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage4", log_t, backend.session(), &stage4.clear_output);
    let stage5 = prove_stage5::<F, PCS, VC, T>(
        backend,
        &mut mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage4.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage5", log_t, backend.session(), &stage5.clear_output);
    let stage6a = prove_stage6a::<F, PCS, VC, T>(
        backend,
        &mut mode,
        &checked,
        config,
        preprocessing,
        &stage1.clear_output,
        &stage2.clear_output,
        &stage3.clear_output,
        &stage4.clear_output,
        &stage5.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage6a", log_t, backend.session(), &stage6a.clear_output);
    let stage6b = prove_stage6b::<F, PCS, VC, T>(
        backend,
        &mut mode,
        &checked,
        config,
        preprocessing,
        &stage1.clear_output,
        &stage2.clear_output,
        &stage3.clear_output,
        &stage4.clear_output,
        &stage5.clear_output,
        &stage6a.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage6b", log_t, backend.session(), &stage6b.clear_output);
    let stage7 = prove_stage7::<F, PCS, VC, T>(
        backend,
        &mut mode,
        &checked,
        config,
        preprocessing,
        &stage4.clear_output,
        &stage6b.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage7", log_t, backend.session(), &stage7.clear_output);
    let (mut pads, mut tape) = mode.into_parts();
    mark_region("stage7", &transcript, tape.values.len());

    // Whiteout's opening replaces the clear stage 8; it keeps the stage's
    // span so the per-stage profile rollup stays aligned.
    let stage8_span = tracing::info_span!("prove_stage8").entered();
    // The packing selectors are short challenges: a proof of work precedes
    // them, as on the clear Akita path.
    let stage8_pow_nonce = if frame.plan.has_selectors() {
        grind_pow::<F, _>(PowSite::Stage8, &mut transcript)?
    } else {
        None
    };
    let pow_nonces = StagePowNonces::from_iter([
        (PowSite::Stage1Tau, stage1.tau_pow_nonce),
        (PowSite::Stage1, stage1.pow_nonce),
        (PowSite::Stage2TauHigh, stage2.tau_high_pow_nonce),
        (PowSite::Stage2, stage2.pow_nonce),
        (PowSite::Stage3, stage3.pow_nonce),
        (PowSite::Stage4, stage4.pow_nonce),
        (PowSite::Stage5, stage5.pow_nonce),
        (PowSite::Stage6a, stage6a.pow_nonce),
        (PowSite::Stage6b, stage6b.pow_nonce),
        (PowSite::Stage7, stage7.pow_nonce),
        (PowSite::Stage8, stage8_pow_nonce),
    ]);

    // Stage 8's claims are affine forms of the masked messages, which only
    // the verifier's residual states: replay its stages 1–7 to obtain them.
    let replay_shell = shell_of(
        pow_nonces,
        tape.clone(),
        transcript.borrow().narg_string().to_vec(),
    );
    let stages = tracing::info_span!("replay_stages").in_scope(|| {
        let (replayed, mut replay) =
            seed_replay::<F, PCS, VC>(preprocessing, public_io, trusted_commitment, &replay_shell)?;
        Ok::<_, ProverError<F>>(whiteout_stages::<F, PCS, VC, _, _>(
            &preprocessing.verifier,
            &replay_shell,
            &replayed,
            trusted_commitment,
            &mut replay,
            Some(&layout),
        )?)
    })?;
    let leaves = leaf_claims(
        &checked.precommitted,
        &stage4.clear_output,
        &stage6b.clear_output,
        &stage7.clear_output,
    )?;
    let mut groups = clear_group_claims(
        &stages.frame.plan,
        checked_one_hot(config),
        &stage0.commitment,
        &leaves,
        &stage6b.clear_output,
        &mut transcript,
    )?;
    if groups.len() != stages.claims.len()
        || groups.iter().zip(&stages.claims).any(|(group, claim)| {
            group.point != claim.point || group.values.len() != claim.evaluations.len()
        })
    {
        return Err(ProverError::InvariantViolation {
            reason: "the opening's groups disagree with the verifier's masked claims",
        });
    }
    let trace = groups.pop().ok_or(ProverError::InvariantViolation {
        reason: "a Whiteout opening has a trace group",
    })?;
    // The precommitted hints in the plan's canonical order: advice, then the
    // field increments or the committed program, then the RV64 increments.
    let hints = stage0
        .untrusted_advice
        .as_ref()
        .map(|object| &object.hint)
        .into_iter()
        .chain(trusted_advice.map(|object| &object.hint));
    #[cfg(feature = "field-inline")]
    let hints = hints.chain(std::iter::once(&stage0.field_inc.hint));
    let hints = hints
        .chain(
            preprocessing
                .committed_program
                .iter()
                .flat_map(|data| data.direct_program.objects().map(|object| &object.hint)),
        )
        .chain(std::iter::once(&stage0.fused_inc.hint))
        .collect::<Vec<_>>();
    if hints.len() != groups.len() {
        return Err(ProverError::InvariantViolation {
            reason: "the precommitted hints disagree with the opening's plan",
        });
    }
    let precommitted = groups
        .into_iter()
        .zip(&stages.frame.plan.objects)
        .zip(hints)
        .map(|((group, object), hint)| {
            (
                TaggedGroupOpeningClaim::new(
                    object.role(),
                    GroupOpeningClaim::new(group.commitment, group.point, group.values),
                ),
                hint.clone(),
            )
        })
        .collect();
    let forms = stages
        .claims
        .iter()
        .flat_map(|claim| claim.evaluations.iter().cloned())
        .collect::<Vec<_>>();
    let (residual, mut auxiliary) =
        tracing::info_span!("whiteout_prove_masked").in_scope(|| {
            PCS::whiteout_prove_masked(
                &preprocessing.pcs_setup,
                layout.shape(),
                precommitted,
                GroupOpeningClaim::new(trace.commitment, trace.point, trace.values),
                stage0.hint,
                &forms,
                &mut pads,
                transcript.borrow_mut(),
            )
        })?;
    mark_region("akita_masked_prefix", &transcript, tape.values.len());
    #[cfg(feature = "prover-fixtures")]
    if faults::abort(transcript.borrow().narg_string()) {
        return Err(ProverError::Zk(ZkError::FoldOverflow { level: 0 }));
    }

    auxiliary.extend(
        stages
            .residual
            .auxiliary_values(|operand| pads.decode(operand.masked, operand.pad))?,
    );
    let _ = pads.send(&auxiliary, &mut tape, &mut transcript)?;
    pads.finish()?;
    mark_region("auxiliary", &transcript, tape.values.len());

    // The verifier's own relation over the proof so far; the replay must land
    // on the prover's transcript state.
    let relation_shell = shell_of(
        pow_nonces,
        tape.clone(),
        transcript.borrow().narg_string().to_vec(),
    );
    let relation = tracing::info_span!("replay_relation").in_scope(|| {
        let (replayed, mut replay) = seed_replay::<F, PCS, VC>(
            preprocessing,
            public_io,
            trusted_commitment,
            &relation_shell,
        )?;
        let mut stages = whiteout_stages::<F, PCS, VC, _, _>(
            &preprocessing.verifier,
            &relation_shell,
            &replayed,
            trusted_commitment,
            &mut replay,
            Some(&layout),
        )?;
        let prefix = whiteout_prefix::<F, PCS, _>(&mut stages, &mut replay)?;
        let relation = whiteout_relation(&mut stages, &prefix, &mut replay)?;
        // A 32-byte compare guarding the seam the design rests on: the replay
        // landing on the prover's forward sponge state.
        let mut forward = [0u8; 32];
        transcript.verifier_message_oracle()(&[], &mut forward);
        let replayed: [u8; 32] = replay.verifier_message_as(32, |bytes| {
            let mut out = [0u8; 32];
            out.copy_from_slice(bytes);
            out
        });
        if replayed != forward {
            return Err(ProverError::InvariantViolation {
                reason: "the verifier replay diverged from the prover's forward transcript",
            });
        }
        Ok(relation)
    })?;
    #[cfg(feature = "prover-fixtures")]
    secrets::record(pads.pads());
    tracing::info_span!("whiteout_prove_tail").in_scope(|| {
        PCS::whiteout_prove_tail(
            &preprocessing.pcs_setup,
            layout.shape(),
            &relation,
            &residual,
            &pads,
            transcript.borrow_mut(),
            rng,
        )
    })?;
    mark_region("sigma_response", &transcript, tape.values.len());
    drop(stage8_span);
    // The response proof's tables are sized by `e`, not the trace, so they
    // are released whatever the trace length `finish_stage` gates on.
    let _ = jolt_kernels::mem::release_retained_memory();
    finish_stage("stage8", log_t, backend.session(), &());
    Ok(shell_of(
        pow_nonces,
        tape,
        transcript.borrow().narg_string().to_vec(),
    ))
}

/// The one-hot configuration stage 0 wrote to the proof.
fn checked_one_hot(config: &ProverConfig) -> jolt_claims::protocols::jolt::JoltOneHotConfig {
    config.one_hot_config
}

/// The checked statement and a verifier transcript seeded over a shell.
type SeededReplay<'a, F, PCS> = (
    CheckedInputs,
    VerifierState<'a, <PCS as MaskedGroupOpening<F>>::Sponge>,
);

/// A verifier transcript over `shell`'s argument string, seeded as the
/// verifier seeds it, for the prover's replays of the verifier.
fn seed_replay<'a, F, PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_commitment: Option<&PCS::Output>,
    shell: &'a WhiteoutProof<PCS, VC>,
) -> Result<SeededReplay<'a, F, PCS>, ProverError<F>>
where
    F: JoltField,
    PCS: WhiteoutScheme<F>,
    PCS::Output: Clone + Encoding + GroupCommitmentMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Encoding,
{
    Ok(validate_and_seed_transcript(
        &preprocessing.verifier,
        public_io,
        shell,
        trusted_commitment,
        |statement| {
            verifier_transcript::<<PCS as MaskedGroupOpening<F>>::Sponge>(
                &JOLT_SESSION,
                statement,
                &shell.narg,
            )
        },
    )?)
}

/// Test-only seam (`prover-fixtures`): the pads of the last Whiteout proof
/// proved on this thread, for soundness probes that rebuild the residual
/// relation as a cheating prover would. The copy is kept as canonical bytes
/// in a buffer zeroed when it is dropped.
#[cfg(feature = "prover-fixtures")]
pub mod secrets {
    use std::cell::RefCell;

    use jolt_algebra::CanonicalEncoding;
    use zeroize::Zeroizing;

    thread_local! {
        static LAST: RefCell<Option<Zeroizing<Vec<u8>>>> = const { RefCell::new(None) };
    }

    pub(super) fn record<F: CanonicalEncoding>(pads: &[F]) {
        let mut bytes = Zeroizing::new(vec![0u8; pads.len() * F::NUM_BYTES]);
        for (pad, out) in pads.iter().zip(bytes.chunks_exact_mut(F::NUM_BYTES)) {
            pad.to_bytes_le(out);
        }
        LAST.with(|last| *last.borrow_mut() = Some(bytes));
    }

    /// Takes the recorded pads, decoded as `F`.
    pub fn take<F: CanonicalEncoding>() -> Option<Vec<F>> {
        let bytes = LAST.with(|last| last.borrow_mut().take())?;
        Some(
            bytes
                .chunks_exact(F::NUM_BYTES)
                .map(F::from_bytes_le_reduced)
                .collect(),
        )
    }
}

/// Test-only seam (`prover-fixtures`): forces the next attempt on this
/// thread to abort as a fold overflow right after its masked levels, and
/// keeps the aborted argument string, so HT3 can check that the rerun
/// reuses nothing from it (§6).
#[cfg(feature = "prover-fixtures")]
pub mod faults {
    use std::cell::RefCell;

    thread_local! {
        static ARMED: RefCell<bool> = const { RefCell::new(false) };
        static ABORTED: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
    }

    /// Aborts the next attempt on this thread after its masked levels.
    pub fn abort_next_attempt() {
        ARMED.with(|armed| *armed.borrow_mut() = true);
    }

    /// The argument string of the attempt [`abort_next_attempt`] aborted.
    pub fn take_aborted() -> Option<Vec<u8>> {
        ABORTED.with(|aborted| aborted.borrow_mut().take())
    }

    /// Whether to abort this attempt, keeping its argument string if so.
    pub(super) fn abort(narg: &[u8]) -> bool {
        let armed = ARMED.with(|armed| std::mem::take(&mut *armed.borrow_mut()));
        if armed {
            ABORTED.with(|aborted| *aborted.borrow_mut() = Some(narg.to_vec()));
        }
        armed
    }
}
