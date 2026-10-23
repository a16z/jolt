//! Stages 1–7 of the Akita prover, shared by the clear prover and (until
//! W6d wires Whiteout's tail) the masked prover-fixtures seam.

use common::jolt_device::JoltDevice;
use jolt_algebra::JoltField;
use jolt_crypto::VectorCommitment;
use jolt_openings::{
    CommitmentScheme, GroupCommitmentMetadata, GroupSetupMetadata, TransparentObjectSetup,
};
use jolt_transcript::{ProverTranscript, Sponge};
#[cfg(feature = "zk")]
use jolt_verifier::pad_layout;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_verifier::CheckedInputs;
#[cfg(feature = "zk")]
use jolt_zk::masking::CommittedPads;
#[cfg(feature = "zk")]
use rand_core::OsRng;

#[cfg(feature = "zk")]
use super::stage0::prove_stage0;
#[cfg(feature = "zk")]
use super::witness::AdviceObject;
use crate::boundary::finish_stage;
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

/// A Whiteout proof's stages 0–7, for tests of the masked stage spine
/// before W6d's tail exists: the argument string and the pads.
#[cfg(feature = "zk")]
pub struct MaskedStagesProof<F> {
    pub narg: Vec<u8>,
    pub pads: Vec<F>,
}

/// Proves stages 0–7 of a Whiteout proof over uncommitted pads drawn from
/// the operating system, on the layout [`pad_layout`] derives: no pad
/// commitment is sent, so the result is not a proof. Test support only.
///
/// # Errors
///
/// [`ProverError`] if a stage fails.
#[cfg(feature = "zk")]
pub fn prove_masked_stages<F, PCS, VC, H>(
    mut backend: KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
) -> Result<MaskedStagesProof<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    let stage0 = prove_stage0::<F, PCS, VC, H>(
        &mut backend,
        preprocessing,
        config,
        trusted_advice,
        public_io,
    )?;
    let mut transcript = stage0.transcript;
    let layout = pad_layout::<F, PCS, VC>(
        &preprocessing.verifier,
        public_io,
        trusted_advice.map(|advice| &advice.commitment),
        transcript.narg(),
    )?;
    let pads = (0..layout.num_pads)
        .map(|_| F::random(&mut OsRng))
        .collect();
    let mut mode = ProofMode::<VC>::masked(CommittedPads::uncommitted(layout, pads)?);
    let _ = prove_stages_1_to_7::<F, PCS, VC, H>(
        &mut backend,
        &mut mode,
        &stage0.checked,
        config,
        preprocessing,
        public_io,
        &mut transcript,
    )?;
    backend.finish();
    Ok(MaskedStagesProof {
        narg: transcript.finish(),
        pads: mode.into_pads().pads().to_vec(),
    })
}

/// Stages 1–7 in `mode`; returns the clear carriers stage 8 reads.
#[expect(clippy::type_complexity, reason = "the three carriers stage 8 reads")]
pub(super) fn prove_stages_1_to_7<F, PCS, VC, H>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    mode: &mut ProofMode<VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    transcript: &mut ProverTranscript<H>,
) -> Result<
    (
        Stage4ClearOutput<F>,
        Stage6bClearOutput<F>,
        Stage7ClearOutput<F>,
    ),
    ProverError<F>,
>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    let log_t = config.trace_length.ilog2() as usize;
    let stage1 = prove_stage1::<F, PCS, VC, H>(backend, mode, log_t, transcript)?;
    finish_stage("stage1", log_t, backend.session(), &stage1.clear_output);
    let stage2 = prove_stage2::<F, PCS, VC, H>(
        backend,
        mode,
        config,
        public_io,
        &stage1.clear_output,
        transcript,
    )?;
    finish_stage("stage2", log_t, backend.session(), &stage2.clear_output);
    let stage3 = prove_stage3::<F, PCS, VC, H>(
        backend,
        mode,
        config,
        &stage1.clear_output,
        &stage2.clear_output,
        transcript,
    )?;
    finish_stage("stage3", log_t, backend.session(), &stage3.clear_output);
    let stage4 = prove_stage4::<F, PCS, VC, H>(
        backend,
        mode,
        checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage3.clear_output,
        transcript,
    )?;
    finish_stage("stage4", log_t, backend.session(), &stage4.clear_output);
    let stage5 = prove_stage5::<F, PCS, VC, H>(
        backend,
        mode,
        checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage4.clear_output,
        transcript,
    )?;
    finish_stage("stage5", log_t, backend.session(), &stage5.clear_output);
    let stage6a = prove_stage6a::<F, PCS, VC, H>(
        backend,
        mode,
        checked,
        config,
        preprocessing,
        &stage1.clear_output,
        &stage2.clear_output,
        &stage3.clear_output,
        &stage4.clear_output,
        &stage5.clear_output,
        transcript,
    )?;
    finish_stage("stage6a", log_t, backend.session(), &stage6a.clear_output);
    let stage6b = prove_stage6b::<F, PCS, VC, H>(
        backend,
        mode,
        checked,
        config,
        preprocessing,
        &stage1.clear_output,
        &stage2.clear_output,
        &stage3.clear_output,
        &stage4.clear_output,
        &stage5.clear_output,
        &stage6a.clear_output,
        transcript,
    )?;
    finish_stage("stage6b", log_t, backend.session(), &stage6b.clear_output);
    let stage7 = prove_stage7::<F, PCS, VC, H>(
        backend,
        mode,
        checked,
        config,
        preprocessing,
        &stage4.clear_output,
        &stage6b.clear_output,
        transcript,
    )?;
    finish_stage("stage7", log_t, backend.session(), &stage7.clear_output);
    Ok((
        stage4.clear_output,
        stage6b.clear_output,
        stage7.clear_output,
    ))
}
