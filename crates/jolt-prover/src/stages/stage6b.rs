//! Stage 6b: the cycle-phase batch — bytecode read+RAF and booleanity cycle
//! phases, RAM Hamming booleanity, both RA virtualizations, the increment
//! claim reduction, and the present precommitted claim-reduction cycle
//! phases (advice, committed bytecode, program image — head-aligned
//! members). A precommitted member with active address-phase rounds stages
//! its intermediate claim here and its kernel object is carried to stage 7
//! for the address phase.
//!
//! Pure orchestration mirroring `stage6b::verify`: the bytecode gamma is
//! carried from stage 6a's squeeze (no draw here), the instruction-RA and
//! increment gammas are drawn post-6a, the batch is built by the verifier's
//! own promoted `Stage6bSumchecks::build_from_parts` over the clear
//! carriers, the challenges aggregate is hand-assembled (the batch
//! suppresses the generated draw), and the driver's curation hook supplies
//! the verifier's promoted `stage6b_opening_values` — the curated order with
//! the runtime dedup of booleanity's `BytecodeRa` claims against the
//! bytecode read-RAF points (which fires when the bytecode address width is
//! a multiple of the committed chunk width).

use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltRelationId, PrecommittedReductionLayout};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::precommitted_reduction::PrecommittedReductionProver;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
use jolt_verifier::stages::stage6a::outputs::Stage6aClearOutput;
use jolt_verifier::stages::stage6b::batch::Stage6bBuildParts;
use jolt_verifier::stages::stage6b::booleanity::BooleanityCyclePhaseChallenges;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCyclePhaseCommittedChallenges;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhaseChallenges, BytecodeReductionCyclePhaseOutputClaims,
    ProgramImageReductionCyclePhaseOutputClaims, TrustedAdviceCyclePhaseOutputClaims,
    UntrustedAdviceCyclePhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReductionChallenges;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualizationChallenges;
use jolt_verifier::stages::stage6b::outputs::{
    Stage6bChallenges, Stage6bClearOutput, Stage6bExternalMembers, Stage6bOutputClaims,
    Stage6bSumchecks,
};
use jolt_verifier::stages::stage6b::{
    stage6b_input_points_from_upstream, stage6b_input_values_from_upstream, stage6b_opening_values,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use super::precommitted::{scalar_phase_adapter, PrecommittedKernelAdapter};
use crate::{BackendPreparer, JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 6b's outputs: the wire proof, the wire claims, the verifier-typed
/// cross-stage carrier stage 7 consumes, and the still-bound advice reduction
/// kernels (present when scheduled) that span into stage 7's address phase.
pub struct Stage6bProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6bOutputClaims<F>,
    pub clear_output: Stage6bClearOutput<F>,
    pub trusted_advice_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
    pub untrusted_advice_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
    pub bytecode_reduction_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
    pub program_image_member: Option<Box<dyn PrecommittedReductionProver<F>>>,
}

/// Prove stage 6b on `transcript` (positioned at the stage-6a boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage6b<F, PCS, VC, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    stage6a: &Stage6aClearOutput<F>,
    witness: &dyn JoltVmWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage6bProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let precommitted = &checked.precommitted;
    let formula_dimensions = JoltFormulaDimensions::try_from(config.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.verifier.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let committed_program = precommitted.bytecode.is_some();

    // The bytecode gamma shares stage 6a's squeeze; the post-6a draws follow
    // the verifier's order.
    let carried = &stage6a.challenges;
    let instruction_ra_gamma: F = transcript.challenge_scalar();
    let inc_gamma: F = transcript.challenge_scalar();
    // The bytecode claim-reduction eta, drawn exactly when the bytecode
    // layout is committed (the verifier's draw position).
    let eta: Option<F> = precommitted
        .bytecode
        .as_ref()
        .map(|_| transcript.challenge_scalar());

    // The batch, through the verifier's own promoted constructor over the
    // clear carriers.
    let program = preprocessing
        .program()
        .ok_or(ProverError::InvariantViolation {
            reason: "full bytecode preprocessing is unavailable",
        })?;
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index()
        .ok_or(ProverError::InvariantViolation {
            reason: "entry address was not found in bytecode preprocessing",
        })?;
    let stage1_cycle_binding =
        stage1
            .cycle_binding()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 1 remainder point is empty".to_string(),
            })?;
    // The staged advice RAM address points from stage 4's RAM value-check —
    // the clear-only references the advice `FinalScale` terms read.
    let advice_reference = |kind| {
        stage4
            .ram_val_check_init
            .advice_contribution(kind)
            .map(|contribution| contribution.opening_point.clone())
    };
    let sumchecks = Stage6bSumchecks::build_from_parts(Stage6bBuildParts {
        formula_dimensions: &formula_dimensions,
        ram_log_k: log_k,
        committed_chunk_bits: chunk_bits,
        precommitted,
        entry_bytecode_index,
        bytecode_table_rows: (!committed_program).then_some(&program.bytecode.bytecode),
        carried,
        eta,
        stage1_cycle_binding: stage1_cycle_binding.clone(),
        stage2_points: &stage2.output_points,
        stage3_points: &stage3.output_points,
        stage4_points: &stage4.output_points,
        stage5_points: &stage5.output_points,
        stage5_instruction_address: stage5.instruction_r_address.clone(),
        stage6a_points: &stage6a.output_points,
        address_val_stages: stage6a.output_values.bytecode_read_raf.val_stages.clone(),
        trusted_advice_reference_point: advice_reference(JoltAdviceKind::Trusted),
        untrusted_advice_reference_point: advice_reference(JoltAdviceKind::Untrusted),
    })?;

    // Hand-assembled (the generated draw is suppressed): the bytecode gamma
    // rides from 6a, the booleanity gamma was drawn pre-6a.
    let cycle_challenges = Stage6bChallenges {
        bytecode_read_raf: BytecodeReadRafCyclePhaseCommittedChallenges {
            gamma: carried.bytecode_read_raf.gamma,
        },
        booleanity: BooleanityCyclePhaseChallenges {
            gamma: carried.booleanity_gamma,
        },
        ram_hamming_booleanity: NoChallenges::default(),
        ram_ra_virtualization: NoChallenges::default(),
        instruction_ra_virtualization: InstructionRaVirtualizationChallenges {
            gamma: instruction_ra_gamma,
        },
        inc_claim_reduction: IncClaimReductionChallenges { gamma: inc_gamma },
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| NoChallenges::default()),
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| NoChallenges::default()),
        bytecode_reduction: sumchecks
            .bytecode_reduction
            .as_ref()
            .zip(eta)
            .map(|(_, eta)| BytecodeReductionCyclePhaseChallenges { eta }),
        program_image_reduction: sumchecks
            .program_image_reduction
            .as_ref()
            .map(|_| NoChallenges::default()),
    };

    let inputs = stage6b_input_values_from_upstream(
        &sumchecks,
        &stage6a.output_values,
        &stage2.output_values,
        stage4,
        &stage5.output_values,
    )?;
    let input_points = stage6b_input_points_from_upstream(
        &sumchecks,
        &stage2.output_points,
        &stage4.output_points,
        &stage5.output_points,
    );

    // The committed-program weights: read back off the batch member (the
    // `build_from_parts` fold), for the bytecode reduction kernel and the
    // clear carrier stage 7 consumes.
    let bytecode_weights = sumchecks
        .bytecode_reduction
        .as_ref()
        .map(|member| member.weights().clone());

    let prepare_advice =
        |session: &mut ProofSession,
         kind: JoltAdviceKind,
         layout: Option<&jolt_claims::protocols::jolt::AdviceClaimReductionLayout>|
         -> Result<Option<Box<dyn PrecommittedReductionProver<F>>>, ProverError<F>> {
            let Some(layout) = layout else {
                return Ok(None);
            };
            let reference = advice_reference(kind).ok_or(ProverError::InvariantViolation {
                reason: "stage 4 staged no advice opening for a scheduled advice reduction",
            })?;
            Ok(Some(
                backend
                    .advice_claim_reduction
                    .prepare(session, kind, layout, &reference, witness)?,
            ))
        };
    let mut trusted_advice_member = prepare_advice(
        session,
        JoltAdviceKind::Trusted,
        precommitted.trusted_advice.as_ref(),
    )?;
    let mut untrusted_advice_member = prepare_advice(
        session,
        JoltAdviceKind::Untrusted,
        precommitted.untrusted_advice.as_ref(),
    )?;
    let mut bytecode_reduction_member = precommitted
        .bytecode
        .as_ref()
        .zip(bytecode_weights.as_ref())
        .map(|(layout, weights)| {
            backend.bytecode_claim_reduction.prepare(
                session,
                layout,
                weights,
                &program.bytecode.bytecode,
            )
        })
        .transpose()?;
    let mut program_image_member = precommitted
        .program_image
        .as_ref()
        .map(|layout| {
            let (point, _) = stage4
                .ram_val_check_init
                .program_image_contribution
                .as_ref()
                .ok_or(ProverError::InvariantViolation {
                    reason: "stage 4 staged no program-image contribution",
                })?;
            backend
                .program_image_claim_reduction
                .prepare(
                    session,
                    layout,
                    point,
                    layout.start_index(),
                    &program.ram.bytecode_words,
                )
                .map_err(ProverError::from)
        })
        .transpose()?;

    // The external adapters: each precommitted member's wire claim is the
    // intermediate handoff claim when its schedule continues into the stage-7
    // address phase, else the final (fully bound) opening.
    let mut trusted_adapter = trusted_advice_member
        .as_mut()
        .zip(precommitted.trusted_advice.as_ref())
        .map(|(member, layout)| {
            scalar_phase_adapter(
                &mut **member,
                layout.dimensions().has_address_phase(),
                |trusted| TrustedAdviceCyclePhaseOutputClaims { trusted },
            )
        });
    let mut untrusted_adapter = untrusted_advice_member
        .as_mut()
        .zip(precommitted.untrusted_advice.as_ref())
        .map(|(member, layout)| {
            scalar_phase_adapter(
                &mut **member,
                layout.dimensions().has_address_phase(),
                |untrusted| UntrustedAdviceCyclePhaseOutputClaims { untrusted },
            )
        });
    let mut bytecode_adapter = bytecode_reduction_member
        .as_mut()
        .zip(precommitted.bytecode.as_ref())
        .map(|(member, layout)| {
            let has_address_phase = layout.dimensions().has_address_phase();
            PrecommittedKernelAdapter::new(
                &mut **member,
                move |member: &dyn PrecommittedReductionProver<F>| {
                    Ok(if has_address_phase {
                        BytecodeReductionCyclePhaseOutputClaims {
                            intermediate: Some(member.cycle_intermediate_claim()),
                            chunks: Vec::new(),
                        }
                    } else {
                        BytecodeReductionCyclePhaseOutputClaims {
                            intermediate: None,
                            chunks: member.final_aux_claims()?,
                        }
                    })
                },
            )
        });
    let mut program_image_adapter = program_image_member
        .as_mut()
        .zip(precommitted.program_image.as_ref())
        .map(|(member, layout)| {
            scalar_phase_adapter(
                &mut **member,
                layout.dimensions().has_address_phase(),
                |program_image| ProgramImageReductionCyclePhaseOutputClaims { program_image },
            )
        });

    let mut preparer = BackendPreparer {
        backend,
        session,
        witness,
        context: (),
    };
    // The curation hook supplies the promoted verifier helper's canonical
    // order, including the runtime booleanity-vs-bytecode point dedup.
    let proved = sumchecks.prove_clear(
        &mut preparer,
        &inputs,
        &input_points,
        &cycle_challenges,
        Stage6bExternalMembers {
            trusted_advice: trusted_adapter.as_mut().map(|adapter| adapter as _),
            untrusted_advice: untrusted_adapter.as_mut().map(|adapter| adapter as _),
            bytecode_reduction: bytecode_adapter.as_mut().map(|adapter| adapter as _),
            program_image_reduction: program_image_adapter.as_mut().map(|adapter| adapter as _),
        },
        |claims, output_points| {
            let booleanity_opening_point =
                output_points.booleanity_opening_point().ok_or_else(|| {
                    VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::Booleanity,
                        reason: "stage-6b booleanity produced no opening point".to_string(),
                    }
                })?;
            Ok(stage6b_opening_values(
                claims,
                &output_points.bytecode_read_raf.bytecode_ra,
                booleanity_opening_point,
            ))
        },
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    // The boxed extraction closures carry the member borrows' lifetime in
    // their type, so the adapters must be dropped before the kernels move
    // into the output (dropck would otherwise extend the borrows to the end
    // of scope).
    drop(trusted_adapter);
    drop(untrusted_adapter);
    drop(program_image_adapter);

    Ok(Stage6bProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage6bClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            bytecode_reduction_weights: bytecode_weights,
        },
        trusted_advice_member,
        untrusted_advice_member,
        bytecode_reduction_member,
        program_image_member,
    })
}
