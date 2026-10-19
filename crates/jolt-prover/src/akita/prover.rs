//! The Akita top-level prover: the stage recipes run in protocol order on
//! one transcript and one backend session, and the transcript's argument
//! string is the [`JoltProof`]; Akita's opening messages are its tail.

use common::jolt_device::JoltDevice;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{
    CommitmentScheme, GroupCommitmentMetadata, GroupSetupMetadata, TransparentObjectSetup,
};
use jolt_transcript::Sponge;
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::JoltProof;

use super::stage0::prove_stage0;
use super::stage8::prove_stage8;
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
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    // The Akita path is transparent-only (`akita` and `zk` are mutually
    // exclusive), so the mode context carries nothing; the shared stage
    // recipes still thread it to mint their clear recorders.
    let mode = ProofMode::<VC>::new(None)?;
    let stage0 = prove_stage0::<F, PCS, VC, H>(
        &mut backend,
        preprocessing,
        config,
        trusted_advice,
        public_io,
    )?;
    let log_t = config.trace_length.ilog2() as usize;
    finish_stage("stage0", log_t, backend.session(), &());
    let checked = stage0.checked;
    let mut transcript = stage0.transcript;

    let stage1 = prove_stage1::<F, PCS, VC, H>(&mut backend, &mode, log_t, &mut transcript)?;
    finish_stage("stage1", log_t, backend.session(), &stage1.clear_output);
    let stage2 = prove_stage2::<F, PCS, VC, H>(
        &mut backend,
        &mode,
        config,
        public_io,
        &stage1.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage2", log_t, backend.session(), &stage2.clear_output);
    let stage3 = prove_stage3::<F, PCS, VC, H>(
        &mut backend,
        &mode,
        config,
        &stage1.clear_output,
        &stage2.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage3", log_t, backend.session(), &stage3.clear_output);
    let stage4 = prove_stage4::<F, PCS, VC, H>(
        &mut backend,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage3.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage4", log_t, backend.session(), &stage4.clear_output);
    let stage5 = prove_stage5::<F, PCS, VC, H>(
        &mut backend,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage4.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage5", log_t, backend.session(), &stage5.clear_output);
    let stage6a = prove_stage6a::<F, PCS, VC, H>(
        &mut backend,
        &mode,
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
    let stage6b = prove_stage6b::<F, PCS, VC, H>(
        &mut backend,
        &mode,
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
    let stage7 = prove_stage7::<F, PCS, VC, H>(
        &mut backend,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage4.clear_output,
        &stage6b.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage7", log_t, backend.session(), &stage7.clear_output);
    prove_stage8::<F, PCS, VC, H>(
        &checked,
        config,
        preprocessing,
        &stage0.commitment,
        stage0.hint,
        stage0.untrusted_advice.as_ref(),
        trusted_advice,
        #[cfg(feature = "field-inline")]
        &stage0.field_inc,
        preprocessing
            .committed_program
            .as_ref()
            .map(|data| &data.direct_program),
        &stage4.clear_output,
        &stage6b.clear_output,
        &stage7.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage8", log_t, backend.session(), &());
    backend.finish();

    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(false),
        narg: transcript.finish(),
    })
}
