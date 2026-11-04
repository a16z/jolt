//! The ZK proof: pads committed before the first challenge, and the BlindFold
//! proof of the residual relation over them after stage 8.
//!
//! The prover does not mirror the verifier's residual lowering — it *runs*
//! it, strictly through `jolt-verifier`'s public verification surface. The
//! pad layout comes from [`pad_layout`], the masked verifier's own replay over
//! the proof's shape. After stage 8 the prover replays its proof shell through
//! [`verify_stages_on`] on a fresh transcript that must land exactly on the
//! prover's own, so the relation it proves is the one the verifier builds —
//! and the replay doubles as a full self-check of the assembled stages. The
//! auxiliary values come from the same residual lowering, over the values the
//! prover's pads decode.

use common::jolt_device::JoltDevice;
use jolt_algebra::JoltField;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{ChallengeCodec, Encoding, FieldTranscript, PadSlots, ProverInit};
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs};
use jolt_verifier::stages::pow::StagePowNonces;
use jolt_verifier::{
    pad_layout, validate_and_seed_transcript, verify_stages_on, BlindFoldZkProof, CheckedInputs,
    VerifiedStages, VerifierError, JOLT_SESSION,
};
use jolt_zk::blindfold::{BlindFold, BlindFoldResidualWitness};
use jolt_zk::masking::{MaskedMessages, PadLayout, ZkBackend};
use rand_core::OsRng;

#[cfg(feature = "prover-fixtures")]
use self::secrets::ZkSecrets;
use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// The stage-8 hiding-opening secrets: the joint evaluation committed inside
/// the PCS's hiding evaluation commitment and its blind.
pub(crate) struct ZkFinalOpening<F> {
    pub joint_evaluation: F,
    pub evaluation_blind: F,
}

/// The pads committed after stage 0 and what the proof keeps of them.
pub(crate) struct CommittedMode<VC: VectorCommitment> {
    pub mode: ProofMode<VC>,
    pub layout: PadLayout,
    pub pad_commitment: Vec<VC::Output>,
}

/// Samples and commits the pads at the end of stage 0, before the first
/// challenge, on the layout the masked verifier derives from the proof's
/// shape: a shell with this proof's configuration and commitments and no
/// stage data (`JoltStageProofs::masked`).
#[tracing::instrument(skip_all, name = "commit_pads")]
pub(crate) fn commit_pads<F, PCS, VC, T>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    checked: &CheckedInputs,
    commitments: &jolt_verifier::proof::ProofCommitments<PCS>,
    untrusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
) -> Result<CommittedMode<VC>, ProverError<F>>
where
    F: JoltField + ChallengeCodec,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + Encoding,
    T: FieldTranscript<F>,
{
    let shell: JoltProof<PCS, VC, (), ()> = JoltProof {
        protocol: JoltProtocolConfig::for_zk(checked.zk),
        commitments: commitments.clone(),
        stages: JoltStageProofs::masked(StagePowNonces::default()),
        joint_opening_proof: (),
        untrusted_advice_commitment: untrusted_advice_commitment.cloned(),
        claims: JoltProofClaims::Zk {
            masked: MaskedMessages::default(),
            zk_proof: (),
        },
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
        narg: Vec::new(),
    };
    let layout = tracing::info_span!("pad_layout")
        .in_scope(|| pad_layout::<F, PCS, VC, _, _>(&preprocessing.verifier, &shell, checked))?;
    let vc_setup = preprocessing
        .verifier
        .vc_setup
        .as_ref()
        .ok_or(ProverError::Verifier(
            VerifierError::MissingVectorCommitmentSetup,
        ))?;
    let (pads, pad_commitment) =
        BlindFold::<VC>::commit_pads(vc_setup, &layout, transcript, &mut OsRng)?;
    Ok(CommittedMode {
        mode: ProofMode::masked(pads),
        layout,
        pad_commitment,
    })
}

/// Proves the ZK tail onto `transcript`, which holds the proof through stage
/// 8: the masked auxiliary block, then BlindFold over the residual relation.
/// The verifier's stage spine is replayed over `shell` (the assembled proof
/// with the masked tape so far) to obtain the relation, and the replay must
/// land on the forward transcript's exact state. Returns the finished masked
/// tape and the BlindFold payload.
#[tracing::instrument(skip_all, name = "prove_blindfold")]
#[expect(
    clippy::too_many_arguments,
    reason = "the tail threads the shell, the pads and their commitment, and the opening secrets once"
)]
pub(crate) fn prove_blindfold<F, PCS, VC, T>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    shell: &JoltProof<PCS, VC, ()>,
    committed: CommittedMode<VC>,
    final_opening: &ZkFinalOpening<F>,
    transcript: &mut T,
) -> Result<(MaskedMessages<F>, BlindFoldZkProof<F, VC::Output>), ProverError<F>>
where
    F: JoltField + Encoding + ChallengeCodec,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Encoding + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + Encoding,
    T: FieldTranscript<F> + ProverInit,
{
    let CommittedMode {
        mode,
        layout,
        pad_commitment,
    } = committed;
    let (checked, mut replay) = validate_and_seed_transcript(
        &preprocessing.verifier,
        public_io,
        shell,
        trusted_advice_commitment,
        |statement| T::init(&JOLT_SESSION, statement),
    )?;
    let VerifiedStages::Zk(masked) = tracing::info_span!("replay_stages").in_scope(|| {
        verify_stages_on::<F, PCS, VC, T, ()>(
            &preprocessing.verifier,
            shell,
            trusted_advice_commitment,
            &checked,
            &mut replay,
            Some(pad_commitment.clone()),
            Some(&layout),
        )
    })?
    else {
        return Err(ProverError::InvariantViolation {
            reason: "the verifier replay of a ZK proof produced no residual relation",
        });
    };
    // A 32-byte compare guarding the seam the design rests on: the replay
    // landing on the prover's forward sponge state, compared by what copies of
    // the two sponges squeeze next. Hard error (not debug-only) so release
    // provers diagnose drift here rather than as a downstream rejection.
    let next_bytes = |transcript: &T| {
        let mut out = [0u8; 32];
        transcript.verifier_message_oracle()(&[], &mut out);
        out
    };
    if next_bytes(&replay) != next_bytes(transcript) {
        return Err(ProverError::InvariantViolation {
            reason: "the verifier replay diverged from the prover's forward transcript",
        });
    }

    let (mut pads, mut tape) = mode.into_parts();
    let auxiliary = masked
        .residual
        .auxiliary_values(|operand| pads.decode(operand.masked, operand.pad))?;
    let start = tape.values.len();
    let _: PadSlots = pads.send(&auxiliary, &mut tape, transcript)?;
    let masked_auxiliary = tape.values.get(start..).unwrap_or_default().to_vec();
    let relation = masked.relation(&masked_auxiliary)?;

    #[cfg(feature = "prover-fixtures")]
    secrets::record(ZkSecrets {
        pads: pads.pads().to_vec(),
        joint_evaluation: final_opening.joint_evaluation,
    });

    let vc_setup = preprocessing
        .verifier
        .vc_setup
        .as_ref()
        .ok_or(ProverError::Verifier(
            VerifierError::MissingVectorCommitmentSetup,
        ))?;
    let blindfold = tracing::info_span!("BlindFold::prove").in_scope(|| {
        BlindFold::<VC>::prove(
            vc_setup,
            &relation,
            pads,
            &masked.residual_commitment,
            &BlindFoldResidualWitness {
                evaluation: final_opening.joint_evaluation,
                blind: final_opening.evaluation_blind,
            },
            transcript,
            &mut OsRng,
        )
    })?;
    Ok((
        tape,
        BlindFoldZkProof {
            pad_commitment,
            blindfold,
        },
    ))
}

/// Test-only seam (`prover-fixtures`): the secrets of the last ZK proof
/// proved on this thread, for soundness probes that rebuild the residual
/// relation as a cheating prover would.
#[cfg(feature = "prover-fixtures")]
pub mod secrets {
    use std::any::Any;
    use std::cell::RefCell;

    /// The pads and the stage-8 joint evaluation (the residual witness).
    pub struct ZkSecrets<F> {
        pub pads: Vec<F>,
        pub joint_evaluation: F,
    }

    thread_local! {
        static LAST: RefCell<Option<Box<dyn Any>>> = const { RefCell::new(None) };
    }

    pub(crate) fn record<F: 'static>(secrets: ZkSecrets<F>) {
        LAST.with(|last| *last.borrow_mut() = Some(Box::new(secrets)));
    }

    /// Takes the recorded secrets, if the last ZK proof on this thread was
    /// over `F`.
    pub fn take<F: 'static>() -> Option<ZkSecrets<F>> {
        LAST.with(|last| last.borrow_mut().take())?
            .downcast()
            .ok()
            .map(|secrets| *secrets)
    }
}
