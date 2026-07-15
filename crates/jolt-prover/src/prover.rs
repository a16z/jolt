//! The top-level prover: the stage recipes run in protocol order on one
//! transcript and one backend session, and their wire outputs assemble into
//! the complete [`JoltProof`].

use common::jolt_device::JoltDevice;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_kernels::JoltBackend;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::{ClearProofClaims, JoltProof, JoltProofClaims, JoltStageProofs};
use jolt_witness::{JoltVmStage5InstructionReadRafRows, JoltVmStage6Rows, JoltWitnessOracle};

use crate::stages::stage0::{prove_stage0, TrustedAdviceCommitment};
use crate::stages::stage1::prove_stage1;
use crate::stages::stage2::prove_stage2;
use crate::stages::stage3::prove_stage3;
use crate::stages::stage4::prove_stage4;
use crate::stages::stage5::prove_stage5;
use crate::stages::stage6a::prove_stage6a;
use crate::stages::stage6b::prove_stage6b;
use crate::stages::stage7::prove_stage7;
use crate::stages::stage8::prove_stage8;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Prove one execution: run stages 0 through 8 on a fresh transcript and
/// backend session, and assemble the clear-mode [`JoltProof`].
///
/// `config` is the derived proof shape (its five wire fields are copied into
/// the proof verbatim), `witness` the trace-backed provider the kernels read,
/// and `public_io` the Fiat-Shamir preamble's program I/O.
///
/// `trusted_advice` is the externally supplied (preprocessing-time)
/// trusted-advice commitment and opening hint; pass it exactly when the guest
/// consumes trusted advice. Untrusted advice needs no extra input — its
/// polynomial is committed at prove time from the witness when
/// `public_io.untrusted_advice` is non-empty.
///
/// Supported envelope: transparent (clear) proofs in either trace layout,
/// with or without trusted/untrusted advice (non-dominant: the advice grid
/// must not exceed the main commitment grid) and with or without
/// committed-program preprocessing (which requires
/// `preprocessing.committed_program` — the prover-retained full program and
/// chunk/image hints). Dominant advice returns
/// [`ProverError::Unsupported`] at stage 0.
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessOracle<F> + JoltVmStage5InstructionReadRafRows + JoltVmStage6Rows,
{
    let mut session = backend.begin_proof();
    let stage0 = prove_stage0::<F, PCS, VC, T>(
        backend,
        &mut session,
        preprocessing,
        config,
        trusted_advice,
        witness,
        public_io,
    )?;
    let checked = stage0.checked;
    let mut transcript = stage0.transcript;
    let log_t = config.trace_length.ilog2() as usize;

    let stage1 = prove_stage1::<F, PCS, VC::Output, T>(
        backend,
        &mut session,
        log_t,
        witness,
        &mut transcript,
    )?;
    let stage2 = prove_stage2::<F, PCS, VC::Output, T>(
        backend,
        &mut session,
        config,
        public_io,
        &stage1.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage3 = prove_stage3::<F, PCS, VC::Output, T>(
        backend,
        &mut session,
        config,
        &stage1.clear_output,
        &stage2.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage4 = prove_stage4::<F, PCS, VC, VC::Output, T>(
        backend,
        &mut session,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage3.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage5 = prove_stage5::<F, PCS, VC, VC::Output, T, W>(
        backend,
        &mut session,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage4.clear_output,
        witness,
        &mut transcript,
    )?;
    let stage6a = prove_stage6a::<F, PCS, VC, VC::Output, T, W>(
        backend,
        &mut session,
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
    let stage6b = prove_stage6b::<F, PCS, VC, VC::Output, T>(
        backend,
        &mut session,
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
    let stage7 = prove_stage7::<F, PCS, VC, VC::Output, T>(
        backend,
        &mut session,
        &checked,
        config,
        preprocessing,
        &stage4.clear_output,
        &stage6b.clear_output,
        stage6b.trusted_advice_member,
        stage6b.untrusted_advice_member,
        stage6b.bytecode_reduction_member,
        stage6b.program_image_member,
        witness,
        &mut transcript,
    )?;
    let stage8 = prove_stage8::<F, PCS, VC, T>(
        backend,
        &mut session,
        &checked,
        config,
        preprocessing,
        &stage0.commitments,
        stage0.untrusted_advice_commitment.as_ref(),
        trusted_advice.map(|trusted| &trusted.commitment),
        &stage0.hints,
        &stage6b.clear_output,
        &stage7.clear_output,
        witness,
        &mut transcript,
    )?;

    Ok(JoltProof {
        protocol: JoltProtocolConfig::for_zk(false),
        commitments: stage0.commitments,
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
        },
        joint_opening_proof: stage8.joint_opening_proof,
        untrusted_advice_commitment: stage0.untrusted_advice_commitment,
        claims: JoltProofClaims::Clear(ClearProofClaims {
            stage1: stage1.claims,
            stage2: stage2.claims,
            stage3: stage3.claims,
            stage4: stage4.claims,
            stage5: stage5.claims,
            stage6a: stage6a.claims,
            stage6b: stage6b.claims,
            stage7: stage7.claims,
        }),
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
    })
}
