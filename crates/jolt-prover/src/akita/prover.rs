//! The packed top-level prover: the stage recipes run in protocol order on
//! one transcript and one backend session, and their wire outputs assemble
//! into the packed-envelope [`JoltProof`].

use common::jolt_device::JoltDevice;
use jolt_akita::TraceOneHotCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::{CanonicalBytes, JoltField};
use jolt_openings::{
    CommitmentScheme, GroupCommitmentMetadata, GroupSetupMetadata, TransparentObjectSetup,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::config::JoltProtocolConfig;
use jolt_verifier::proof::{ClearProofClaims, JoltProof, JoltProofClaims, JoltStageProofs};
use jolt_witness::JoltWitnessPlane;

use super::stage0::prove_stage0;
use super::stage8::prove_stage8;
use super::witness::AdviceObject;
use super::JoltAkitaBackend;
use crate::stages::stage1::prove_stage1;
use crate::stages::stage2::prove_stage2;
use crate::stages::stage3::prove_stage3;
use crate::stages::stage4::prove_stage4;
use crate::stages::stage5::prove_stage5;
use crate::stages::stage6a::prove_stage6a;
use crate::stages::stage6b::prove_stage6b;
use crate::stages::stage7::prove_stage7;
use crate::{JoltProverPreprocessing, ProofMode, ProverConfig, ProverError};

/// See [`super::prove`].
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltAkitaBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: JoltField + CanonicalBytes + AppendToTranscript,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup + TraceOneHotCommitment,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone + PartialEq + AppendToTranscript + GroupCommitmentMetadata,
    VC: VectorCommitment<Field = F>,
    VC::Output: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F> + Sync,
{
    // The packed path is transparent-only (`akita` and `zk` are mutually
    // exclusive), so the mode context carries nothing; the shared stage
    // recipes still thread it to mint their clear recorders.
    let mode = ProofMode::<VC>::new(None)?;
    let mut session = backend.begin_proof();
    let log_t = config.trace_length.ilog2() as usize;
    // Backend witness preparation only reads the witness and parks owners in
    // the session, which stage 0 never touches, so it runs on its own thread
    // under the trace commitment. It starts once the one-hot rows are
    // assembled: assembly is CPU-bound, the commit leaves the CPU mostly idle.
    let (rows_ready_sender, rows_ready_receiver) = std::sync::mpsc::channel::<()>();
    let (stage0, witness_prepare) = std::thread::scope(|scope| {
        let prepare_kernel = backend.base.spartan_outer_uniskip.as_ref();
        let prepare_session = &mut session;
        let prepare = scope.spawn(move || -> Result<(), ProverError<F>> {
            // A dropped sender means stage 0 failed before assembling rows;
            // the stage 0 error is reported first, so just run to completion.
            let _ = rows_ready_receiver.recv();
            let span = tracing::info_span!(
                "jolt_prover::backend_witness_prepare_async",
                log_t,
                cycles = 1usize << log_t,
                complete = tracing::field::Empty,
            );
            let _entered = span.enter();
            let result = prepare_kernel.prepare_witness(prepare_session, log_t, witness);
            let _ = span.record("complete", result.is_ok());
            result.map_err(ProverError::from)
        });
        let stage0 = prove_stage0::<F, PCS, VC, T, W>(
            backend,
            preprocessing,
            config,
            trusted_advice,
            witness,
            public_io,
            Some(&rows_ready_sender),
        );
        drop(rows_ready_sender);
        let completed_before_join = prepare.is_finished();
        let _span = tracing::info_span!(
            "jolt_prover::backend_witness_prepare",
            completed_before_join
        )
        .entered();
        let witness_prepare = prepare
            .join()
            .map_err(|_| ProverError::InvariantViolation {
                reason: "asynchronous backend witness preparation panicked",
            })
            .and_then(|result| result);
        (stage0, witness_prepare)
    });
    let stage0 = stage0?;
    witness_prepare?;
    let checked = stage0.checked;
    let mut transcript = stage0.transcript;

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
    let joint_opening_proof = prove_stage8::<F, PCS, VC, T>(
        &checked,
        config,
        preprocessing,
        &stage0.commitment,
        stage0.hint,
        stage0.untrusted_advice.as_ref(),
        trusted_advice,
        preprocessing
            .committed_program
            .as_ref()
            .map(|data| &data.direct_program),
        &stage6b.clear_output,
        &stage7.clear_output,
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
        }),
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
    })
}
