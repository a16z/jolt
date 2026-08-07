//! The packed top-level prover: the stage recipes run in protocol order on
//! one transcript and one backend session, and their wire outputs assemble
//! into the packed-envelope [`JoltProof`].

use common::jolt_device::JoltDevice;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalBytes, Field};
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::{ClearProofClaims, JoltProof, JoltProofClaims, JoltStageProofs};
use jolt_witness::JoltWitnessPlane;

use super::reconstruction::prove_reconstruction;
use super::stage0::prove_stage0;
use super::stage8::prove_stage8;
use super::witness::{commit_advice_one_hot, commit_program_one_hot};
use super::JoltAkitaBackend;
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

/// See [`super::prove`].
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltAkitaBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&PCS::Output>,
    program_one_hot: Option<&PCS::Output>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: Field + CanonicalBytes + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
{
    // The packed path is transparent-only (`akita` and `zk` are mutually
    // exclusive), so the mode context carries nothing; the shared stage
    // recipes still thread it to mint their clear recorders.
    let mode = ProofMode::<VC>::new(None)?;
    let mut session = backend.begin_proof();
    let stage0 = prove_stage0::<F, PCS, VC, T, W>(
        preprocessing,
        config,
        trusted_advice,
        program_one_hot,
        witness,
        public_io,
    )?;
    let checked = stage0.checked;
    let mut transcript = stage0.transcript;
    let log_t = config.trace_length.ilog2() as usize;

    let stage1 = prove_stage1::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        log_t,
        witness,
        &mut transcript,
    )?;
    let stage2 = prove_stage2::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        config,
        public_io,
        &stage1.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage3 = prove_stage3::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        config,
        &stage1.clear_output,
        &stage2.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage4 = prove_stage4::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage3.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage5 = prove_stage5::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage4.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage6a = prove_stage6a::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage1.clear_output,
        &stage2.clear_output,
        &stage3.clear_output,
        &stage4.clear_output,
        &stage5.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage6b = prove_stage6b::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
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
        witness,
        &mut transcript,
    )?;
    let stage7 = prove_stage7::<F, PCS, VC, T>(
        &backend.base,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage4.clear_output,
        &stage6b.clear_output,
        witness,
        &mut transcript,
    )?;
    let reconstruction = prove_reconstruction::<F, PCS, VC::Output, T>(
        backend,
        &mut session,
        &checked,
        &stage6b.clear_output,
        &stage7.clear_output,
        witness,
        &mut transcript,
    )?;

    // The auxiliary objects' opening material, transparently re-derived from
    // the public shapes and cross-checked against the passed precommitted
    // commitments (a divergence means the caller's object was built from
    // different data and its opening could never verify).
    let trusted_object = trusted_advice
        .map(|commitment| {
            let object = commit_advice_one_hot::<PCS>(
                &public_io.trusted_advice,
                public_io.memory_layout.max_trusted_advice_size as usize,
            )?;
            if object.commitment != *commitment {
                return Err(ProverError::Unsupported {
                    reason: "the trusted-advice commitment does not match the public advice bytes",
                });
            }
            Ok(object)
        })
        .transpose()?;
    let program_object = program_one_hot
        .map(|commitment| {
            let chunk_count = checked
                .precommitted
                .bytecode
                .as_ref()
                .map(|layout| layout.chunk_count())
                .ok_or(ProverError::InvariantViolation {
                    reason: "committed-program mode without a bytecode schedule",
                })?;
            let object =
                commit_program_one_hot::<PCS>(witness.program_preprocessing(), chunk_count)?;
            if object.commitment != *commitment {
                return Err(ProverError::Unsupported {
                    reason: "the ProgramOneHot commitment does not match the retained program",
                });
            }
            Ok(object)
        })
        .transpose()?;

    let joint_opening_proof = prove_stage8::<F, PCS, VC, T>(
        &checked,
        config,
        preprocessing,
        stage0.hint,
        stage0.untrusted_advice.as_ref(),
        trusted_object.as_ref(),
        program_object.as_ref(),
        &stage7.clear_output,
        &reconstruction.clear_output,
        &mut transcript,
    )?;

    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(false),
        commitments: stage0.commitment,
        stages: JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
            stage1_sumcheck_proof: stage1.sumcheck_proof,
            stage2_uni_skip_first_round_proof: stage2.uniskip_proof,
            stage2_sumcheck_proof: stage2.sumcheck_proof,
            stage3_sumcheck_proof: stage3.sumcheck_proof,
            stage4_sumcheck_proof: stage4.sumcheck_proof,
            stage5_sumcheck_proof: stage5.sumcheck_proof,
            stage6a_sumcheck_proof: stage6a.sumcheck_proof,
            stage6b_sumcheck_proof: stage6b.sumcheck_proof,
            stage7_sumcheck_proof: stage7.sumcheck_proof,
            reconstruction_sumcheck_proof: reconstruction.sumcheck_proof,
        },
        joint_opening_proof,
        untrusted_advice_commitment: stage0
            .untrusted_advice
            .map(|object| object.commitment.clone()),
        claims: JoltProofClaims::Clear(ClearProofClaims {
            stage1: stage1.claims,
            stage2: stage2.claims,
            stage3: stage3.claims,
            stage4: stage4.claims,
            stage5: stage5.claims,
            stage6a: stage6a.claims,
            stage6b: stage6b.claims,
            stage7: stage7.claims,
            reconstruction: reconstruction.claims,
        }),
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
    })
}
