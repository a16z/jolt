//! The ZK proof tail: BlindFold over the committed stage proofs.
//!
//! The prover does not mirror the verifier's protocol lowering — it *runs*
//! it, strictly through `jolt-verifier`'s existing public verification
//! surface. After stage 8 it assembles a shell proof (every wire field real,
//! the claims slot a unit placeholder) and replays it through the verifier's
//! own stage functions — `validate_and_seed_transcript`, `stage1::verify` …
//! `stage8::verify` — to obtain the per-stage ZK outputs and a transcript
//! positioned exactly where the verifier's will be, then lowers them with
//! the verifier's own `stages::zk::blindfold::build`. The `BlindFoldProtocol`
//! the prover proves against is therefore the same code path the verifier
//! executes — a claim-formula change that updates the verifier's lowering is
//! picked up here automatically — and the replay doubles as a full
//! self-check of the assembled proof.
//!
//! The witness rows come from the recorder-retained per-stage secrets via
//! [`BlindFoldProtocol::assign_witness`], which needs only the protocol's
//! public parts plus the stage domains — protocol constants this crate's own
//! stage recipes prove over.

use common::jolt_device::JoltDevice;
use jolt_blindfold::{BlindFoldProof, BlindFoldProtocol, BlindFoldWitness};
use jolt_claims::protocols::jolt::geometry::dimensions::{
    OUTER_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_DOMAIN_SIZE,
};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_sumcheck::{CommittedSumcheckWitness, SumcheckDomainSpec};
use jolt_transcript::{AppendToTranscript, Label, Transcript};
use jolt_verifier::proof::JoltProof;
use jolt_verifier::stages::zk::{blindfold, inputs::BlindFoldInputs};
use jolt_verifier::stages::{
    stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7, stage8,
};
use jolt_verifier::VerifierError;

use crate::{JoltProverPreprocessing, ProverError};

/// The recorder-retained committed sumcheck witnesses, one per BlindFold
/// stage, named to pin the protocol stage order (`blindfold::build` inserts
/// each stage's uni-skip before its remainder batch).
pub(crate) struct ZkStageWitnesses<F> {
    pub stage1_uniskip: CommittedSumcheckWitness<F>,
    pub stage1: CommittedSumcheckWitness<F>,
    pub stage2_uniskip: CommittedSumcheckWitness<F>,
    pub stage2: CommittedSumcheckWitness<F>,
    pub stage3: CommittedSumcheckWitness<F>,
    pub stage4: CommittedSumcheckWitness<F>,
    pub stage5: CommittedSumcheckWitness<F>,
    pub stage6a: CommittedSumcheckWitness<F>,
    pub stage6b: CommittedSumcheckWitness<F>,
    pub stage7: CommittedSumcheckWitness<F>,
}

impl<F> ZkStageWitnesses<F> {
    fn in_protocol_order(&self) -> [&CommittedSumcheckWitness<F>; 10] {
        [
            &self.stage1_uniskip,
            &self.stage1,
            &self.stage2_uniskip,
            &self.stage2,
            &self.stage3,
            &self.stage4,
            &self.stage5,
            &self.stage6a,
            &self.stage6b,
            &self.stage7,
        ]
    }
}

/// The BlindFold stage domains in the same order: the two uni-skips run over
/// their centered integer domains, every batch over the Boolean hypercube —
/// the constants the stage recipes themselves prove over.
const STAGE_DOMAINS: [SumcheckDomainSpec; 10] = [
    SumcheckDomainSpec::centered_integer(OUTER_UNISKIP_DOMAIN_SIZE),
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::centered_integer(PRODUCT_UNISKIP_DOMAIN_SIZE),
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
    SumcheckDomainSpec::BooleanHypercube,
];

/// The stage-8 hiding-opening secrets: the joint evaluation committed inside
/// the PCS's hiding evaluation commitment and its blind.
pub(crate) struct ZkFinalOpening<F> {
    pub joint_evaluation: F,
    pub evaluation_blind: F,
}

/// Prove the BlindFold tail for `shell` (the assembled proof with a unit
/// claims placeholder). `forward_state` is the prover's own transcript state
/// at the stage-8 boundary — the replay must land on the same bytes.
pub(crate) fn prove_blindfold<F, PCS, VC, T>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&PCS::Output>,
    shell: &JoltProof<PCS, VC, ()>,
    witnesses: &ZkStageWitnesses<F>,
    final_opening: &ZkFinalOpening<F>,
    forward_state: [u8; 32],
) -> Result<BlindFoldProof<F, VC::Output>, ProverError<F>>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let (protocol, mut transcript) = replay_stages::<F, PCS, VC, T>(
        &preprocessing.verifier,
        public_io,
        shell,
        trusted_advice_commitment,
    )?;
    debug_assert_eq!(
        transcript.state(),
        forward_state,
        "the verifier replay diverged from the prover's forward transcript",
    );

    let assigned = protocol.assign_witness(
        &STAGE_DOMAINS,
        &witnesses.in_protocol_order(),
        &[final_opening.joint_evaluation],
        &[final_opening.evaluation_blind],
        &mut rand_core::OsRng,
    )?;

    let vc_setup = preprocessing
        .verifier
        .vc_setup
        .as_ref()
        .ok_or(ProverError::Verifier(
            VerifierError::MissingVectorCommitmentSetup,
        ))?;
    transcript.append(&Label(b"BlindFold"));
    let proof = jolt_blindfold::prove::<F, VC, T, _>(
        vc_setup,
        &protocol,
        &mut transcript,
        BlindFoldWitness {
            rows: &assigned.rows,
            blindings: &assigned.blindings,
            eval_outputs: &[final_opening.joint_evaluation],
            eval_blindings: &[final_opening.evaluation_blind],
        },
        &mut rand_core::OsRng,
    )?;
    Ok(proof)
}

/// Replay the shell through the verifier's public stage spine and lower the
/// ZK outputs into the BlindFold protocol — the same call sequence
/// `jolt_verifier::verify` runs before its BlindFold tail, expressed against
/// the same public surface its ZK audit harness uses.
#[expect(
    clippy::type_complexity,
    reason = "the pair is the protocol plus the transcript it was lowered on"
)]
fn replay_stages<F, PCS, VC, T>(
    preprocessing: &jolt_verifier::JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    shell: &JoltProof<PCS, VC, ()>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(BlindFoldProtocol<F, VC::Output>, T), ProverError<F>>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let (checked, mut transcript) = jolt_verifier::validate_and_seed_transcript::<PCS, VC, T, ()>(
        preprocessing,
        public_io,
        shell,
        trusted_advice_commitment,
    )?;
    let formula_dimensions = jolt_verifier::stages::build_formula_dimensions(
        shell,
        preprocessing,
        &checked,
        checked.trace_length.ilog2() as usize,
        JoltRelationId::InstructionReadRaf,
    )?;

    let stage1 = stage1::verify(&checked, shell, &mut transcript)?;
    let stage2 = stage2::verify(&checked, shell, &mut transcript, &stage1)?;
    let stage3 = stage3::verify(&checked, shell, &mut transcript, &stage1, &stage2)?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        shell,
        &mut transcript,
        &stage2,
        &stage3,
    )?;
    let stage5 = stage5::verify(
        &checked,
        shell,
        &formula_dimensions,
        &mut transcript,
        &stage2,
        &stage4,
    )?;
    let stage6a = stage6a::verify(
        &checked,
        preprocessing,
        shell,
        &formula_dimensions,
        &mut transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
    )?;
    let stage6b = stage6b::verify(
        &checked,
        preprocessing,
        shell,
        &formula_dimensions,
        &mut transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
        &stage6a,
    )?;
    let stage7 = stage7::verify(
        &checked,
        shell,
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6b,
    )?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        shell,
        &formula_dimensions,
        trusted_advice_commitment,
        &mut transcript,
        &stage6b,
        &stage7,
    )?;

    let protocol = blindfold::build(BlindFoldInputs {
        checked: &checked,
        preprocessing,
        proof: shell,
        stage1: stage1.zk()?,
        stage2: stage2.zk()?,
        stage3: stage3.zk()?,
        stage4: stage4.zk()?,
        stage5: stage5.zk()?,
        stage6a: stage6a.zk()?,
        stage6b: stage6b.zk()?,
        stage7: stage7.zk()?,
        stage8: stage8.zk()?,
    })?;

    Ok((protocol, transcript))
}
