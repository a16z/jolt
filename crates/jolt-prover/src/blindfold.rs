//! The ZK proof: pads committed before the first challenge, and the BlindFold
//! proof of the residual relation over them after stage 8.
//!
//! The prover does not mirror the verifier's residual lowering — it *runs*
//! it, strictly through `jolt-verifier`'s public verification surface. The
//! pad layout comes from [`pad_layout`], the masked verifier's own replay of
//! the seeded prefix. After stage 8 the prover replays its argument string
//! through [`verify_stages`] on a verifier transcript that must land exactly
//! on the prover's own, so the relation it proves is the one the verifier
//! builds — and the replay doubles as a full self-check of the assembled
//! proof. The auxiliary values come from the same residual lowering, over the
//! values the prover's pads decode.

use common::jolt_device::JoltDevice;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{CanonicalDecode, JoltField};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use jolt_verifier::sites::{BLINDFOLD, PADS};
use jolt_verifier::{
    jolt_protocol_id, pad_layout, verify_stages, VerifiedStages, VerifierError, JOLT_SESSION,
};
use jolt_zk::blindfold::{BlindFold, BlindFoldResidualWitness};
use jolt_zk::masking::ZkBackend;
use rand_core::OsRng;

use crate::recorder::{ModePads, ProofMode};
use crate::{JoltProverPreprocessing, ProverError};

/// The stage-8 hiding-opening secrets: the joint evaluation committed inside
/// the PCS's hiding evaluation commitment and its blind.
pub(crate) struct ZkFinalOpening<F> {
    pub joint_evaluation: F,
    pub evaluation_blind: F,
}

/// Samples and commits the pads at the end of stage 0, before the first
/// challenge, on the layout the masked verifier derives from the seeded
/// prefix.
pub(crate) fn commit_pads<F, PCS, VC, H>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut ProverTranscript<H>,
) -> Result<ProofMode<VC>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalDecode,
    H: Sponge,
{
    let layout = pad_layout::<F, PCS, VC>(
        &preprocessing.verifier,
        public_io,
        trusted_advice_commitment,
        transcript.narg(),
    )?;
    let vc_setup = preprocessing
        .verifier
        .vc_setup
        .as_ref()
        .ok_or(ProverError::Verifier(
            VerifierError::MissingVectorCommitmentSetup,
        ))?;
    transcript.site(PADS);
    let pads = BlindFold::<VC>::commit_pads(vc_setup, &layout, transcript, &mut OsRng)?;
    Ok(ProofMode::masked(pads))
}

/// Proves the ZK tail onto `transcript`, which holds the argument string
/// through stage 8: the masked auxiliary block, then BlindFold over the
/// residual relation. The verifier's stage spine is replayed over that prefix
/// to obtain the relation, and the replay must land on the forward
/// transcript's exact state.
pub(crate) fn prove_blindfold<F, PCS, VC, H>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    mut pads: ModePads<VC>,
    final_opening: &ZkFinalOpening<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalDecode,
    H: Sponge,
{
    let mut replay =
        VerifierTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION, transcript.narg());
    let VerifiedStages::Zk(masked) = verify_stages::<F, PCS, VC, H>(
        &preprocessing.verifier,
        public_io,
        trusted_advice_commitment,
        &mut replay,
    )?
    else {
        return Err(ProverError::InvariantViolation {
            reason: "the verifier replay of a ZK proof produced no residual relation",
        });
    };
    // The prover draws the same checkpoint the verifier draws after the stage
    // spine; equal checkpoints over the whole argument string mean the replay
    // reached the prover's sponge state. Hard error (not debug-only) so release
    // provers diagnose drift here rather than as a downstream rejection.
    if replay.remaining() != 0 || transcript.challenge_bytes::<32>() != masked.checkpoint {
        return Err(ProverError::InvariantViolation {
            reason: "the verifier replay diverged from the prover's forward transcript",
        });
    }

    transcript.site(BLINDFOLD);
    let auxiliary = masked
        .residual
        .auxiliary_values(|operand| pads.decode(operand.masked, operand.pad))?;
    let slots = pads.send(&auxiliary, transcript)?;
    let masked_auxiliary = auxiliary
        .iter()
        .zip(pads.pads().get(slots.start..).unwrap_or_default())
        .map(|(value, pad)| *value + *pad)
        .collect::<Vec<_>>();
    let relation = masked.relation(&masked_auxiliary)?;

    let vc_setup = preprocessing
        .verifier
        .vc_setup
        .as_ref()
        .ok_or(ProverError::Verifier(
            VerifierError::MissingVectorCommitmentSetup,
        ))?;
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
    .map_err(ProverError::from)
}
