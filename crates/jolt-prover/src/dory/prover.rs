//! The top-level prover: the stage recipes run in protocol order on one
//! transcript and one backend session, and the transcript's argument string
//! is the [`JoltProof`].

use common::jolt_device::JoltDevice;
use jolt_algebra::CanonicalDecode;
use jolt_algebra::{Accumulator, JoltField, WithAccumulator};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_kernels::{JoltBackend, KernelContext};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::Sponge;
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::JoltProof;

use crate::boundary::finish_stage;
use crate::dory::stages::stage0::{prove_stage0, TrustedAdviceCommitment};
use crate::dory::stages::stage8::prove_stage8;
use crate::recorder::ProofMode;
use crate::stages::stage1::prove_stage1;
use crate::stages::stage2::prove_stage2;
use crate::stages::stage3::prove_stage3;
use crate::stages::stage4::prove_stage4;
use crate::stages::stage5::prove_stage5;
use crate::stages::stage6a::prove_stage6a;
use crate::stages::stage6b::prove_stage6b;
use crate::stages::stage7::prove_stage7;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Prove one execution: run stages 0 through 8 on a fresh transcript and
/// backend session in the compiled proof mode — clear claims without the `zk`
/// feature, the BlindFold tail with it — and return the argument string.
///
/// `config` is the derived proof shape (sent as the proof header), `backend` binds the witness used by the kernels,
/// and `public_io` the Fiat-Shamir preamble's program I/O.
///
/// `trusted_advice` is the externally supplied (preprocessing-time)
/// trusted-advice commitment and opening hint; pass it exactly when the guest
/// consumes trusted advice. Untrusted advice needs no extra input — its
/// polynomial is committed at prove time from the witness when
/// `public_io.untrusted_advice` is non-empty.
///
/// Supported envelope: either trace layout,
/// with or without trusted/untrusted advice (non-dominant: the advice grid
/// must not exceed the main commitment grid) and with or without
/// committed-program preprocessing (which requires
/// `preprocessing.committed_program` — the prover-retained full program and
/// bytecode/image hints). Dominant advice returns
/// [`ProverError::Unsupported`] at stage 0.
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, H>(
    mut backend: KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    public_io: &JoltDevice,
) -> Result<JoltProof, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output, Blind = F>,
    PCS::Output: HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalDecode,
    H: Sponge,
    <F as WithAccumulator>::Accumulator: Accumulator<Element = F>,
{
    let mode = ProofMode::<VC>::new(preprocessing.verifier.vc_setup.as_ref())?;
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
    let stage8 = prove_stage8::<F, PCS, VC, H>(
        &mut backend,
        &checked,
        config,
        preprocessing,
        &stage0.commitments,
        trusted_advice.map(|trusted| &trusted.commitment),
        stage0.hints,
        #[cfg(feature = "field-inline")]
        stage0.field_inline_hints,
        &stage6b.clear_output,
        &stage7.clear_output,
        &mut transcript,
    )?;
    finish_stage("stage8", log_t, backend.session(), &());
    backend.finish();

    #[cfg(not(feature = "zk"))]
    let _ = stage8;
    #[cfg(feature = "zk")]
    {
        use crate::blindfold::{self, ZkFinalOpening, ZkStageWitnesses};

        let witnesses = ZkStageWitnesses {
            stage1_uniskip: stage1.uniskip_witness,
            stage1: stage1.committed_witness,
            stage2_uniskip: stage2.uniskip_witness,
            stage2: stage2.committed_witness,
            stage3: stage3.committed_witness,
            stage4: stage4.committed_witness,
            stage5: stage5.committed_witness,
            stage6a: stage6a.committed_witness,
            stage6b: stage6b.committed_witness,
            stage7: stage7.committed_witness,
        };
        let final_opening = ZkFinalOpening {
            joint_evaluation: stage8.joint_evaluation,
            evaluation_blind: stage8.evaluation_blind,
        };
        blindfold::prove_blindfold::<F, PCS, VC, H>(
            preprocessing,
            public_io,
            trusted_advice.map(|trusted| &trusted.commitment),
            &witnesses,
            &final_opening,
            &mut transcript,
        )?;
    }

    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(cfg!(feature = "zk")),
        narg: transcript.finish(),
    })
}
