//! The top-level prover: the stage recipes run in protocol order on one
//! transcript and one backend session, and their wire outputs assemble into
//! the complete [`JoltProof`].

use core::any::Any;
use std::sync::Arc;

#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::JoltDevice;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::config::JoltProtocolConfig;
#[cfg(not(feature = "zk"))]
use jolt_verifier::proof::ClearProofClaims;
use jolt_verifier::proof::{JoltProof, JoltProofClaims, JoltStageProofs};
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage1::outputs::Stage1ClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage6a::outputs::Stage6aClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_witness::JoltWitnessPlane;

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

/// Per-stage heap snapshots for the profile harness: inert unless the
/// harness opted in via `jolt_profiling::set_flamegraph_prefix`. The stage's
/// clear-output carrier is recovered by downcast to the concrete BN254
/// field (the only production field), so `prove` needs no `Allocative`
/// bound on `F`; the proof session is visited shallowly (its carries are
/// `Box<dyn Any>` — see the `ProofSession` impl in `jolt-kernels`).
#[cfg(feature = "allocative")]
fn stage_flamegraph(stage: &str, session: &ProofSession, output: &dyn Any) {
    use jolt_field::Fr;

    let Some(prefix) = jolt_profiling::flamegraph_prefix() else {
        return;
    };
    let mut flamegraph = FlameGraphBuilder::default();
    macro_rules! visit_downcast {
        ($($ty:ty),+ $(,)?) => {$(
            if let Some(concrete) = output.downcast_ref::<$ty>() {
                flamegraph.visit_root(concrete);
            }
        )+};
    }
    visit_downcast!(
        Stage1ClearOutput<Fr>,
        Stage2ClearOutput<Fr>,
        Stage3ClearOutput<Fr>,
        Stage4ClearOutput<Fr>,
        Stage5ClearOutput<Fr>,
        Stage6aClearOutput<Fr>,
        Stage6bClearOutput<Fr>,
        Stage7ClearOutput<Fr>,
    );
    flamegraph.visit_root(session);
    jolt_profiling::write_flamegraph_folded(flamegraph, format!("{prefix}{stage}.folded"));
}

#[cfg(not(feature = "allocative"))]
fn stage_flamegraph(_stage: &str, _session: &ProofSession, _output: &dyn Any) {}

/// Prove one execution: run stages 0 through 8 on a fresh transcript and
/// backend session, and assemble the [`JoltProof`] in the compiled proof
/// mode — clear claims without the `zk` feature, the BlindFold tail with it.
///
/// `config` is the derived proof shape (its five wire fields are copied into
/// the proof verbatim), `witness` the owned trace-backed provider the kernels
/// read and retain for deferred extraction, and `public_io` the Fiat-Shamir
/// preamble's program I/O.
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
/// chunk/image hints). Dominant advice returns
/// [`ProverError::Unsupported`] at stage 0.
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    witness: Arc<W>,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output, Blind = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F> + 'static,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let mode = ProofMode::<VC>::new(preprocessing.verifier.vc_setup.as_ref())?;
    let mut session = backend.begin_proof();
    let session_witness: Arc<dyn JoltWitnessPlane<F>> = witness.clone();
    session.set_witness(session_witness);
    let stage0 = prove_stage0::<F, PCS, VC, T, W>(
        backend,
        &mut session,
        preprocessing,
        config,
        trusted_advice,
        witness.as_ref(),
        public_io,
    )?;
    stage_flamegraph("stage0", &session, &());
    let checked = stage0.checked;
    let mut transcript = stage0.transcript;
    let log_t = config.trace_length.ilog2() as usize;

    let stage1 = prove_stage1::<F, PCS, VC, T>(
        backend,
        &mut session,
        &mode,
        log_t,
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage1", &session, &stage1.clear_output);
    let stage2 = prove_stage2::<F, PCS, VC, T>(
        backend,
        &mut session,
        &mode,
        config,
        public_io,
        &stage1.clear_output,
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage2", &session, &stage2.clear_output);
    let stage3 = prove_stage3::<F, PCS, VC, T>(
        backend,
        &mut session,
        &mode,
        config,
        &stage1.clear_output,
        &stage2.clear_output,
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage3", &session, &stage3.clear_output);
    let stage4 = prove_stage4::<F, PCS, VC, T>(
        backend,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage3.clear_output,
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage4", &session, &stage4.clear_output);
    let stage5 = prove_stage5::<F, PCS, VC, T>(
        backend,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage2.clear_output,
        &stage4.clear_output,
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage5", &session, &stage5.clear_output);
    let stage6a = prove_stage6a::<F, PCS, VC, T>(
        backend,
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
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage6a", &session, &stage6a.clear_output);
    let stage6b = prove_stage6b::<F, PCS, VC, T>(
        backend,
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
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage6b", &session, &stage6b.clear_output);
    let stage7 = prove_stage7::<F, PCS, VC, T>(
        backend,
        &mut session,
        &mode,
        &checked,
        config,
        preprocessing,
        &stage4.clear_output,
        &stage6b.clear_output,
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage7", &session, &stage7.clear_output);
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
        witness.as_ref(),
        &mut transcript,
    )?;
    stage_flamegraph("stage8", &session, &());

    let stages = JoltStageProofs {
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
    };

    #[cfg(not(feature = "zk"))]
    {
        Ok(JoltProof {
            protocol: JoltProtocolConfig::for_zk(false),
            commitments: stage0.commitments,
            stages,
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
    #[cfg(feature = "zk")]
    {
        use crate::blindfold::{self, ZkFinalOpening, ZkStageWitnesses};

        // The shell: every wire field real, the claims slot a unit
        // placeholder the stage replay never reads (claims are not absorbed
        // in ZK — the BlindFold proof replaces them after the tail).
        let shell = JoltProof::<PCS, VC, ()> {
            protocol: JoltProtocolConfig::for_zk(true),
            commitments: stage0.commitments,
            stages,
            joint_opening_proof: stage8.joint_opening_proof,
            untrusted_advice_commitment: stage0.untrusted_advice_commitment,
            claims: JoltProofClaims::Zk {
                blindfold_proof: (),
            },
            trace_length: config.trace_length,
            ram_K: config.ram_K,
            rw_config: config.rw_config,
            one_hot_config: config.one_hot_config,
            trace_polynomial_order: config.trace_polynomial_order,
        };
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
        let blindfold_proof = blindfold::prove_blindfold::<F, PCS, VC, T>(
            preprocessing,
            public_io,
            trusted_advice.map(|trusted| &trusted.commitment),
            &shell,
            &witnesses,
            &final_opening,
            transcript.state(),
        )?;

        Ok(shell.with_claims(JoltProofClaims::Zk { blindfold_proof }))
    }
}
