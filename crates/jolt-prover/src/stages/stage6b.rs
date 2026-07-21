//! Stage 6b: the cycle-phase batch — bytecode read+RAF and booleanity cycle
//! phases, RAM Hamming booleanity, both RA virtualizations, the increment
//! claim reduction, and the present precommitted claim-reduction cycle
//! phases (advice, committed bytecode, program image — head-aligned
//! members). A precommitted member's cycle kernel parks its shared two-phase
//! state in the proof session; when the schedule has active address-phase
//! rounds it stages its intermediate claim here and stage 7's address-phase
//! member reclaims the carry.
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

use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltRelationId};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
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
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::BytecodeReductionCyclePhaseChallenges;
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReductionChallenges;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualizationChallenges;
use jolt_verifier::stages::stage6b::outputs::{
    Stage6bChallenges, Stage6bClearOutput, Stage6bOutputClaims, Stage6bSumchecks,
};
use jolt_verifier::stages::stage6b::{
    stage6b_input_points_from_upstream, stage6b_input_values_from_upstream,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 6b's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 7 consumes. The precommitted reduction kernels
/// that span into stage 7's address phase travel as `ProofSession` carries,
/// not output fields.
pub struct Stage6bProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6bOutputClaims<F>,
    pub clear_output: Stage6bClearOutput<F>,
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
    let log_k = checked.ram_K.ilog2() as usize;
    let precommitted = &checked.precommitted;
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::BytecodeReadRaf,
    )?;
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
    // `build_from_parts` fold), for the clear carrier stage 7 consumes (the
    // bytecode reduction kernel reads them off its relation).
    let bytecode_weights = sumchecks
        .bytecode_reduction
        .as_ref()
        .map(|member| member.weights().clone());

    // The absorb order is the stage's curation override at its
    // `impl_stage_prover` invocation site (the promoted verifier helper's
    // canonical order, including the runtime booleanity-vs-bytecode point
    // dedup).
    let proved = sumchecks.prove(
        backend,
        session,
        witness,
        &inputs,
        &input_points,
        &cycle_challenges,
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    Ok(Stage6bProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage6bClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            bytecode_reduction_weights: bytecode_weights,
        },
    })
}
