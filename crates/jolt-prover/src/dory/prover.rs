//! The top-level prover: the stage recipes run in protocol order on one
//! transcript and one backend session, and their wire outputs assemble into
//! the complete [`JoltProof`].

use core::any::Any;

#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Accumulator, JoltField, WithAccumulator};
use jolt_kernels::optimized::booleanity::spawn_booleanity_address_masses;
use jolt_kernels::optimized::bytecode_read_raf::spawn_bytecode_stage_pushforwards;
use jolt_kernels::optimized::trace_record::spawn_shared_record_collect;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::config::{BooleanityAnchor, JoltProtocolConfig};
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
use jolt_verifier::stages::stage6a::bytecode_read_raf::bytecode_early_stage_points;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage6a::outputs::Stage6aClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
#[cfg(feature = "allocative")]
use jolt_verifier::stages::stage7::outputs::Stage7ClearOutput;
use jolt_witness::JoltWitnessPlane;

use crate::dory::stages::stage0::{prove_stage0, TrustedAdviceCommitment};
use crate::dory::stages::stage8::{prove_stage8, stage8_materialization_plan, Stage8Prepared};
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
/// the proof verbatim), `witness` the trace-backed provider the kernels read,
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
/// chunk/image hints). Dominant advice returns
/// [`ProverError::Unsupported`] at stage 0.
#[tracing::instrument(skip_all, name = "jolt_prover::prove", fields(trace_length = config.trace_length))]
pub fn prove<F, PCS, VC, T, W>(
    backend: &JoltBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<JoltProof<PCS, VC>, ProverError<F>>
where
    F: JoltField + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output, Blind = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
    <F as WithAccumulator>::Accumulator: Accumulator<Element = F>,
{
    let mode = ProofMode::<VC>::new(preprocessing.verifier.vc_setup.as_ref())?;
    let mut session = backend.begin_proof();
    let log_t = config.trace_length.ilog2() as usize;
    // The stage sequence runs inside one thread scope so the shared
    // trace-record walk — challenge-independent, witness-only — can build on
    // a capped background pool overlapping stage 0's device-heavy commit
    // window. Stage 1's first record consumer joins it (or rebuilds inline —
    // identical values either way).
    std::thread::scope(|scope| {
        spawn_shared_record_collect(
            &mut session,
            witness as &dyn JoltWitnessPlane<F>,
            log_t,
            scope,
        );
        let stage0 = prove_stage0::<F, PCS, VC, T, W>(
            backend,
            &mut session,
            preprocessing,
            config,
            trusted_advice,
            witness,
            public_io,
        )?;
        stage_flamegraph("stage0", &session, &());
        let checked = stage0.checked;
        let mut transcript = stage0.transcript;

        let stage1 = prove_stage1::<F, PCS, VC, T>(
            backend,
            &mut session,
            &mode,
            log_t,
            witness,
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
            witness,
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
            witness,
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
            witness,
            &mut transcript,
        )?;
        stage_flamegraph("stage4", &session, &stage4.clear_output);
        // Stage-6a prepare work whose inputs exist by the end of stage 4 builds
        // on capped background pools overlapping stage 5's device-heavy window;
        // the 6a prepares join (or rebuild inline on any mismatch — identical
        // values either way).
        //
        // Booleanity pushforward: only under the stage-1 anchor (`Stage1CycleV1`)
        // — the legacy anchor point does not exist until stage 5 ends.
        if config.booleanity_anchor == BooleanityAnchor::Stage1CycleV1 {
            let formula_dimensions = crate::stages::formula_dimensions(
                &checked,
                config,
                preprocessing.verifier.program.bytecode_len(),
                JoltRelationId::Booleanity,
            )?;
            let anchor: Vec<F> = stage1
                .clear_output
                .cycle_binding_checked(JoltRelationId::Booleanity)?
                .iter()
                .rev()
                .copied()
                .collect();
            spawn_booleanity_address_masses(
                &mut session,
                witness,
                BooleanityDimensions::new(
                    formula_dimensions.ra_layout,
                    formula_dimensions.trace.log_t(),
                    config.one_hot_config.committed_chunk_bits(),
                ),
                anchor,
            )?;
        }
        // Bytecode read-RAF stage-1..4 pushforwards: anchor-independent (the four
        // early points are protocol-fixed); the stage-5 table's walk stays on the
        // 6a critical path.
        {
            let formula_dimensions = crate::stages::formula_dimensions(
                &checked,
                config,
                preprocessing.verifier.program.bytecode_len(),
                JoltRelationId::BytecodeReadRaf,
            )?;
            let stage1_cycle_binding = stage1
                .clear_output
                .cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
            let early_points = bytecode_early_stage_points(
                &stage1_cycle_binding,
                &stage2.clear_output.output_points,
                &stage3.clear_output.output_points,
                &stage4.clear_output.output_points,
            )?;
            spawn_bytecode_stage_pushforwards(
                &mut session,
                witness,
                formula_dimensions.trace.log_t(),
                1usize << formula_dimensions.bytecode_read_raf.log_k(),
                early_points,
            )?;
        }
        let stage5 = prove_stage5::<F, PCS, VC, T>(
            backend,
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
            witness,
            &mut transcript,
        )?;
        stage_flamegraph("stage6a", &session, &stage6a.clear_output);
        let stage8_plan =
            stage8_materialization_plan::<F, PCS, VC>(&checked, config, preprocessing)?;
        let mut prefetch_session = backend.joint_opening.prefetch_session(&mut session);
        let (stage6b, stage7, stage8_polynomials) = std::thread::scope(|scope| {
            let prefetch = scope.spawn(|| {
                // Backend-neutral kernel-seam span at the call boundary, so every
                // `JointOpeningPolynomials` implementation inherits it — see the
                // taxonomy's kernel-seam contract.
                tracing::info_span!(
                    "JointOpeningPolynomials::prepare",
                    polynomials = stage8_plan.order.len(),
                    total_vars = stage8_plan.grid.total_vars
                )
                .in_scope(|| {
                    backend.joint_opening.prepare(
                        &mut prefetch_session,
                        witness,
                        &stage8_plan.order,
                        &stage8_plan.precommitted_tables,
                        stage8_plan.grid,
                    )
                })
            });
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
                witness,
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
                witness,
                &mut transcript,
            )?;
            stage_flamegraph("stage7", &session, &stage7.clear_output);
            let polynomials = prefetch
                .join()
                .map_err(|_| ProverError::InvariantViolation {
                    reason: "stage-8 materialization prefetch panicked",
                })??;
            Ok::<_, ProverError<F>>((stage6b, stage7, polynomials))
        })?;
        let stage8_prepared = Stage8Prepared::new(stage8_plan, stage8_polynomials);
        let stage8 = prove_stage8::<F, PCS, VC, T>(
            &checked,
            config,
            preprocessing,
            &stage0.commitments,
            stage0.untrusted_advice_commitment.as_ref(),
            trusted_advice.map(|trusted| &trusted.commitment),
            &stage0.hints,
            &stage6b.clear_output,
            &stage7.clear_output,
            stage8_prepared,
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
                protocol: JoltProtocolConfig {
                    booleanity_anchor: config.booleanity_anchor,
                    ..JoltProtocolConfig::for_zk(false)
                },
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
    })
}
