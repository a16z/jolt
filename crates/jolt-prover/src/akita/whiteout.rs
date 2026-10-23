//! The Whiteout prover: Jolt's zero-knowledge proof over Akita
//! (`specs/whiteout-jolt-integration.md`), in the verifier's transcript order
//! (`jolt_verifier::whiteout`). A fold overflow or exhausted Σ attempts rerun
//! the whole proof with fresh randomness, at most [`WHITEOUT_ATTEMPTS`]
//! times (§6).

use common::jolt_device::JoltDevice;
use jolt_akita::TraceOneHotCommitment;
use jolt_algebra::JoltField;
use jolt_crypto::VectorCommitment;
use jolt_openings::{
    GroupCommitmentMetadata, GroupOpeningClaim, GroupSetupMetadata, TaggedGroupOpeningClaim,
    TransparentObjectSetup,
};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};
use jolt_verifier::config::{JoltProtocolConfig, ZkConfig};
use jolt_verifier::proof::JoltProof;
use jolt_verifier::sites::{PADS, STAGE8, WHITEOUT, WHITEOUT_SIGMA};
use jolt_verifier::stages::stage8::akita::{clear_group_claims, leaf_claims};
use jolt_verifier::whiteout::{
    pad_layout, whiteout_prefix, whiteout_relation, whiteout_stages, WhiteoutScheme,
};
use jolt_verifier::{jolt_protocol_id, JOLT_SESSION};
use rand_core::{CryptoRng, OsRng, RngCore};

use super::spine::prove_stages_1_to_7;
use super::stage0::prove_stage0;
use super::witness::AdviceObject;
use crate::boundary::finish_stage;
use crate::{JoltProverPreprocessing, ProofMode, ProverConfig, ProverError};
use jolt_kernels::{JoltBackend, KernelContext};

/// The most attempts a Whiteout proof makes (§6): the first run and two
/// reruns.
const WHITEOUT_ATTEMPTS: usize = 3;

/// See [`super::prove`].
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, H>(
    mut backend: KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
) -> Result<JoltProof, ProverError<F>>
where
    F: JoltField,
    PCS: WhiteoutScheme<F> + TraceOneHotCommitment + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    let mut attempt = 1;
    loop {
        let result = prove_once::<F, PCS, VC, H, _>(
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
                backend.finish();
                return result;
            }
        }
    }
}

/// One Whiteout attempt from stage 0. An aborted attempt's transcript is
/// dropped with it.
fn prove_once<F, PCS, VC, H, R>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
    rng: &mut R,
) -> Result<JoltProof, ProverError<F>>
where
    F: JoltField,
    PCS: WhiteoutScheme<F> + TraceOneHotCommitment + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
    R: RngCore + CryptoRng + Send,
{
    let trusted_commitment = trusted_advice.map(|advice| &advice.commitment);
    let stage0 =
        prove_stage0::<F, PCS, VC, H>(backend, preprocessing, config, trusted_advice, public_io)?;
    let log_t = config.trace_length.ilog2() as usize;
    finish_stage("stage0", log_t, backend.session(), &());
    let mut transcript = stage0.transcript;
    let layout = pad_layout::<F, PCS, VC, H>(
        &preprocessing.verifier,
        public_io,
        trusted_commitment,
        transcript.narg(),
    )?;
    transcript.site(PADS);
    let pads = PCS::whiteout_commit_pads(
        &preprocessing.pcs_setup,
        layout.shape(),
        layout.layout().clone(),
        &mut transcript,
        rng,
    )?;
    let mut mode = ProofMode::<VC>::masked(pads);
    let (stage4, stage6b, stage7) = prove_stages_1_to_7::<F, PCS, VC, H>(
        backend,
        &mut mode,
        &stage0.checked,
        config,
        preprocessing,
        public_io,
        &mut transcript,
    )?;
    let mut pads = mode.into_pads();

    // Stage 8's claims are affine forms of the masked messages, which only
    // the verifier's residual states: replay stages 1–7 to obtain them.
    let stages = {
        let mut replay =
            VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, transcript.narg());
        whiteout_stages::<F, PCS, VC, H>(
            &preprocessing.verifier,
            public_io,
            trusted_commitment,
            &mut replay,
            Some(&layout),
        )?
    };
    transcript.site(STAGE8);
    let leaves = leaf_claims(&stage0.checked.precommitted, &stage4, &stage6b, &stage7)?;
    let mut groups = clear_group_claims(
        &stages.plan,
        stage0.checked.one_hot_config,
        &stage0.commitment,
        &leaves,
        #[cfg(feature = "field-inline")]
        &stage6b,
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
    // field increments or the committed program.
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
        .collect::<Vec<_>>();
    if hints.len() != groups.len() {
        return Err(ProverError::InvariantViolation {
            reason: "the precommitted hints disagree with the opening's plan",
        });
    }
    let precommitted = groups
        .into_iter()
        .zip(&stages.plan.objects)
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
    let (residual, mut auxiliary) = PCS::whiteout_prove_masked(
        &preprocessing.pcs_setup,
        layout.shape(),
        precommitted,
        GroupOpeningClaim::new(trace.commitment, trace.point, trace.values),
        stage0.hint,
        &forms,
        &mut pads,
        &mut transcript,
    )?;
    #[cfg(feature = "prover-fixtures")]
    if faults::abort(transcript.narg()) {
        return Err(ProverError::Zk(jolt_zk::masking::ZkError::FoldOverflow {
            level: 0,
        }));
    }
    let checkpoint = transcript.challenge_bytes::<32>();

    transcript.site(WHITEOUT);
    auxiliary.extend(
        stages
            .residual
            .auxiliary_values(|operand| pads.decode(operand.masked, operand.pad))?,
    );
    let _ = pads.send(&auxiliary, &mut transcript)?;
    pads.finish()?;

    // The verifier's own relation over the whole argument string so far; the
    // replay must land on the prover's transcript state.
    let relation = {
        let mut replay =
            VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, transcript.narg());
        let mut stages = whiteout_stages::<F, PCS, VC, H>(
            &preprocessing.verifier,
            public_io,
            trusted_commitment,
            &mut replay,
            Some(&layout),
        )?;
        let (prefix, replayed) = whiteout_prefix(&mut stages, &mut replay)?;
        if replayed != checkpoint {
            return Err(ProverError::InvariantViolation {
                reason: "the verifier replay diverged from the prover's forward transcript",
            });
        }
        let relation = whiteout_relation(&mut stages, &prefix, &mut replay)?;
        if replay.remaining() != 0 {
            return Err(ProverError::InvariantViolation {
                reason: "the verifier replay stopped short of the auxiliary block's end",
            });
        }
        relation
    };
    #[cfg(feature = "prover-fixtures")]
    secrets::record(pads.pads());
    transcript.site(WHITEOUT_SIGMA);
    PCS::whiteout_prove_tail(
        &preprocessing.pcs_setup,
        layout.shape(),
        &relation,
        &residual,
        &pads,
        &mut transcript,
        rng,
    )?;
    finish_stage("stage8", log_t, backend.session(), &());
    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(ZkConfig::Whiteout),
        narg: transcript.finish(),
    })
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
