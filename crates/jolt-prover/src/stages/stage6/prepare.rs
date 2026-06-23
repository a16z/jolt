use common::jolt_device::JoltDevice;
use jolt_backends::{
    stage6_bytecode_pc_indices, stage6_hamming_weight, stage6_inc_rows, stage6_ra_rows,
    Stage6RegularBatchSumcheckBackend, SumcheckBooleanityStateRequest,
    SumcheckBytecodeReadRafStateRequest, SumcheckIncClaimReductionStateRequest,
    SumcheckInstructionRaVirtualizationStateRequest, SumcheckRamHammingBooleanityStateRequest,
    SumcheckRamRaVirtualizationStateRequest,
};
use jolt_claims::protocols::jolt::{
    formulas::{booleanity::BooleanityDimensions, bytecode},
    AdviceClaimReductionLayout, JoltAdviceKind, JoltFormulaDimensions,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::SumcheckInstance;
use jolt_verifier::stages::stage6::outputs::Stage6AddressPhaseClaims;
use jolt_verifier::stages::stage6::{
    stage6_advice_cycle_phase_reference, stage6_bytecode_cycle_points,
    stage6_bytecode_register_points, stage6_inc_claim_reduction_cycle_points,
    stage6_instruction_read_raf_point, stage6_post_address_transcript_challenges,
    stage6_pre_address_transcript_challenges, stage6_stage1_cycle_binding,
    stage6_stage5_ram_reduced_opening_point, AdviceCyclePhase, AdviceCyclePhaseInputClaims,
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims, IncClaimReduction,
    IncClaimReductionInputClaims, InstructionRaVirtualization,
    InstructionRaVirtualizationInputClaims, RamHammingBooleanity, RamHammingBooleanityInputClaims,
    RamRaVirtualization, RamRaVirtualizationInputClaims, Stage6BatchInputClaims,
    Stage6PreAddressChallenges, Stage6TranscriptChallenges,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmStage6Rows};
use jolt_witness::WitnessProvider;

use super::io::Stage6RegularBatchPrefixOutput;
use super::Stage6ProverConfig;
use crate::stages::advice::advice_layouts;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

pub(super) const STAGE6_REGULAR_BATCH_OPT_IDS: &[&str] = &["cpu_stage6_regular_batch_sumcheck"];

pub(super) struct Stage6BackendStates<B, F>
where
    F: Field,
    B: Stage6RegularBatchSumcheckBackend<F>,
{
    pub(super) bytecode_read_raf: B::BytecodeReadRafState,
    pub(super) booleanity: B::BooleanityState,
    pub(super) ram_hamming_booleanity: B::RamHammingBooleanityState,
    pub(super) ram_ra_virtualization: B::RamRaVirtualizationState,
    pub(super) instruction_ra_virtualization: B::InstructionRaVirtualizationState,
    pub(super) inc_claim_reduction: B::IncClaimReductionState,
}

pub(crate) fn prover_config<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof_parameters: ProverConfig,
    formula_dimensions: JoltFormulaDimensions,
) -> Result<Stage6ProverConfig, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = proof_parameters.trace_length.trailing_zeros() as usize;
    let log_k = proof_parameters.ram_k.trailing_zeros() as usize;
    let committed_chunk_bits = proof_parameters.one_hot_config.committed_chunk_bits();
    let (trusted_advice_layout, untrusted_advice_layout) =
        advice_layouts(public_io, proof_parameters, log_t, committed_chunk_bits)?;
    let full_program = preprocessing.verifier.program.as_full().ok_or_else(|| {
        ProverError::InvalidProverConfig {
            reason: "Stage 6 requires full program preprocessing (committed-program mode is not supported by the prover)".to_owned(),
        }
    })?;
    let entry_bytecode_index = full_program
        .bytecode
        .entry_bytecode_index()
        .ok_or_else(|| ProverError::InvalidProverConfig {
            reason: "entry address was not found in bytecode preprocessing".to_owned(),
        })?;

    Ok(Stage6ProverConfig::new(
        log_t,
        log_k,
        committed_chunk_bits,
        formula_dimensions.bytecode_read_raf,
        BooleanityDimensions::new(formula_dimensions.ra_layout, log_t, committed_chunk_bits),
        formula_dimensions.ram_ra_virtualization,
        formula_dimensions.instruction_ra_virtualization,
        trusted_advice_layout,
        untrusted_advice_layout,
    )
    .with_bytecode_context(full_program.bytecode.bytecode.clone(), entry_bytecode_index))
}

pub(super) fn derive_stage6_pre_address_challenges<F, T>(
    config: &Stage6ProverConfig,
    stage5: &Stage5ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage6PreAddressChallenges<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let instruction_read_raf =
        jolt_verifier::stages::stage6::stage6_instruction_read_raf_point(stage5);
    Ok(stage6_pre_address_transcript_challenges(
        instruction_read_raf.address,
        instruction_read_raf.cycle,
        config.committed_chunk_bits,
        stage5
            .output_claims
            .instruction_read_raf
            .lookup_table_flags
            .len(),
        transcript,
    ))
}

/// Computes the bytecode read-RAF address-phase input claim (the stage-folding
/// RLC).
pub(super) fn bytecode_read_raf_address_input<F>(
    config: &Stage6ProverConfig,
    pre: &Stage6PreAddressChallenges<F>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<F, ProverError>
where
    F: Field,
{
    let inputs = BytecodeReadRafAddressPhaseInputClaims::from_upstream(
        stage1, stage2, stage3, stage4, stage5,
    )
    .map_err(invalid_stage_request)?;
    // `num_val_stages` only affects produced opening points, not the input claim.
    BytecodeReadRafAddressPhase::new(
        config.bytecode_read_raf_dimensions,
        pre.bytecode_gamma_powers[1],
        [
            pre.stage1_gammas[1],
            pre.stage2_gammas[1],
            pre.stage3_gammas[1],
            pre.stage4_gammas[1],
            pre.stage5_gammas[1],
        ],
        0,
    )
    .input_claim(&inputs)
    .map_err(invalid_stage_request)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 prefix completion consumes every prior clear stage output."
)]
pub(super) fn complete_stage6_prefix<F, T>(
    config: &Stage6ProverConfig,
    pre: Stage6PreAddressChallenges<F>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    address_phase: &Stage6AddressPhaseClaims<F>,
    transcript: &mut T,
) -> Result<Stage6RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let post = stage6_post_address_transcript_challenges(
        config.instruction_ra_virtualization_dimensions,
        transcript,
    );
    let transcript_challenges = Stage6TranscriptChallenges::from_address_phases(pre, post);
    let input_claims = stage6_cycle_input_claims(
        config,
        &transcript_challenges,
        address_phase,
        stage1,
        stage2,
        stage4,
        stage5,
    )?;

    Ok(Stage6RegularBatchPrefixOutput::new(
        input_claims,
        transcript_challenges,
    ))
}

/// The stage-6b cycle-batch input claims, single-sourced through the same relation
/// objects the verifier uses (`relation.input_claim`). The bytecode/booleanity
/// cycle phases consume the stage-6a address-phase openings directly; the remaining
/// relations evaluate their input `Expr` against the upstream openings. Input
/// claims are point-independent, but the relations are built with their real
/// construction points so the bundle's algebra is reproduced verbatim.
fn stage6_cycle_input_claims<F: Field>(
    config: &Stage6ProverConfig,
    challenges: &Stage6TranscriptChallenges<F>,
    address_phase: &Stage6AddressPhaseClaims<F>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<Stage6BatchInputClaims<F>, ProverError> {
    let trace_dimensions = config.trace_dimensions();
    let committed_chunk_bits = config.committed_chunk_bits;

    let ram_hamming = RamHammingBooleanity::new(
        trace_dimensions,
        stage6_stage1_cycle_binding(stage1)
            .map_err(invalid_stage_request)?
            .to_vec(),
    );
    let ram_hamming_booleanity = ram_hamming
        .input_claim(&RamHammingBooleanityInputClaims::from_upstream())
        .map_err(invalid_stage_request)?;

    let ram_reduced = stage6_stage5_ram_reduced_opening_point(stage5, config.log_k, config.log_t)
        .map_err(invalid_stage_request)?;
    let ram_ra = RamRaVirtualization::new(
        config.ram_ra_virtualization_dimensions,
        ram_reduced.address.to_vec(),
        ram_reduced.cycle.to_vec(),
        committed_chunk_bits,
    );
    let ram_ra_virtualization = ram_ra
        .input_claim(&RamRaVirtualizationInputClaims::from_upstream(stage5))
        .map_err(invalid_stage_request)?;

    let instruction_read_raf = stage6_instruction_read_raf_point(stage5);
    let instruction_ra = InstructionRaVirtualization::new(
        config.instruction_ra_virtualization_dimensions,
        challenges.instruction_ra_gamma,
        instruction_read_raf.address.to_vec(),
        instruction_read_raf.cycle.to_vec(),
        committed_chunk_bits,
    );
    let instruction_ra_virtualization = instruction_ra
        .input_claim(&InstructionRaVirtualizationInputClaims::from_upstream(
            stage5,
        ))
        .map_err(invalid_stage_request)?;

    let inc_cycles = stage6_inc_claim_reduction_cycle_points(stage2, stage4, stage5, config.log_k)
        .map_err(invalid_stage_request)?;
    let inc = IncClaimReduction::new(
        trace_dimensions,
        challenges.inc_gamma,
        inc_cycles.ram_read_write_cycle.to_vec(),
        inc_cycles.ram_val_check_cycle.to_vec(),
        inc_cycles.registers_read_write_cycle.to_vec(),
        inc_cycles.registers_val_evaluation_cycle.to_vec(),
    );
    let inc_claim_reduction = inc
        .input_claim(&IncClaimReductionInputClaims::from_upstream(
            stage2, stage4, stage5,
        ))
        .map_err(invalid_stage_request)?;

    let advice_inputs = AdviceCyclePhaseInputClaims::from_upstream(stage4);
    let advice_input = |kind: JoltAdviceKind,
                        layout: Option<&AdviceClaimReductionLayout>|
     -> Result<Option<F>, ProverError> {
        layout
            .map(|layout| {
                let reference = stage6_advice_cycle_phase_reference(stage4, kind)
                    .map_err(invalid_stage_request)?;
                AdviceCyclePhase::new(kind, layout, reference.opening_point.to_vec())
                    .input_claim(&advice_inputs)
                    .map_err(invalid_stage_request)
            })
            .transpose()
    };

    Ok(Stage6BatchInputClaims {
        bytecode_read_raf: address_phase.bytecode_read_raf,
        booleanity: address_phase.booleanity,
        ram_hamming_booleanity,
        ram_ra_virtualization,
        instruction_ra_virtualization,
        inc_claim_reduction,
        trusted_advice_cycle_phase: advice_input(
            JoltAdviceKind::Trusted,
            config.trusted_advice_layout.as_ref(),
        )?,
        untrusted_advice_cycle_phase: advice_input(
            JoltAdviceKind::Untrusted,
            config.untrusted_advice_layout.as_ref(),
        )?,
    })
}

pub(super) fn bytecode_stage_values<F>(
    config: &Stage6ProverConfig,
    pre: &Stage6PreAddressChallenges<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<Vec<[F; 5]>, ProverError>
where
    F: Field,
{
    let bytecode_context = config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;
    let register_points =
        stage6_bytecode_register_points(stage4, stage5).map_err(invalid_stage_request)?;
    let bytecode_rows = bytecode_context.rows.as_slice();
    let stage_values = bytecode::read_raf_stage_values(bytecode::BytecodeReadRafStageValueInputs {
        bytecode: bytecode_rows,
        register_read_write_point: register_points.register_read_write_address,
        register_val_evaluation_point: register_points.register_val_evaluation_address,
        stage1_gammas: &pre.stage1_gammas,
        stage2_gammas: &pre.stage2_gammas,
        stage3_gammas: &pre.stage3_gammas,
        stage4_gammas: &pre.stage4_gammas,
        stage5_gammas: &pre.stage5_gammas,
    });
    Ok(stage_values)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 materialization consumes every prior clear stage output."
)]
pub(super) fn materialize_address_phase_states<F, W, B>(
    config: &Stage6ProverConfig,
    witness: &W,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    pre: &Stage6PreAddressChallenges<F>,
    bytecode_read_raf_input: F,
    backend: &mut B,
) -> Result<(B::BytecodeReadRafState, B::BooleanityState), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: Stage6RegularBatchSumcheckBackend<F>,
{
    let stage6_rows = witness.stage6_rows()?;
    let bytecode_context = config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;

    let bytecode_request = SumcheckBytecodeReadRafStateRequest::new(
        "Stage 6 bytecode read-RAF",
        bytecode_stage_values(config, pre, stage4, stage5)?,
        stage6_bytecode_pc_indices(&stage6_rows),
        stage6_bytecode_cycle_points(stage1, stage2, stage3, stage4, stage5)
            .map_err(invalid_stage_request)?,
        pre.bytecode_gamma_powers.clone(),
        bytecode_context.entry_bytecode_index,
        bytecode_read_raf_input,
        config.log_t,
        config.bytecode_read_raf_dimensions.log_k(),
        config.committed_chunk_bits,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let bytecode_read_raf =
        backend.materialize_sumcheck_bytecode_read_raf_state(&bytecode_request)?;

    let booleanity_layout = config.booleanity_dimensions.layout;
    let booleanity_request = SumcheckBooleanityStateRequest::new(
        "Stage 6 booleanity",
        stage6_ra_rows(&stage6_rows),
        pre.booleanity_reference.address.clone(),
        pre.booleanity_reference.cycle.clone(),
        pre.booleanity_gamma,
        F::zero(),
        config.log_t,
        config.committed_chunk_bits,
        booleanity_layout.instruction(),
        booleanity_layout.bytecode(),
        booleanity_layout.ram(),
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let booleanity = backend.materialize_sumcheck_booleanity_state(&booleanity_request)?;

    Ok((bytecode_read_raf, booleanity))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 materialization consumes every prior clear stage output."
)]
pub(super) fn materialize_cycle_phase_states<F, W, B>(
    config: &Stage6ProverConfig,
    witness: &W,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    bytecode_read_raf: B::BytecodeReadRafState,
    booleanity: B::BooleanityState,
    backend: &mut B,
) -> Result<Stage6BackendStates<B, F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: Stage6RegularBatchSumcheckBackend<F>,
{
    let stage6_rows = witness.stage6_rows()?;
    let (ra_rows, inc_rows) = (stage6_ra_rows(&stage6_rows), stage6_inc_rows(&stage6_rows));

    let hamming_request = SumcheckRamHammingBooleanityStateRequest::new(
        "Stage 6 RAM hamming booleanity",
        stage6_hamming_weight(&stage6_rows),
        stage6_stage1_cycle_binding(stage1)
            .map_err(invalid_stage_request)?
            .to_vec(),
        prefix.input_claims.ram_hamming_booleanity,
        config.log_t,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let ram_hamming_booleanity =
        backend.materialize_sumcheck_ram_hamming_booleanity_state(&hamming_request)?;

    let ram_reduced = stage6_stage5_ram_reduced_opening_point(stage5, config.log_k, config.log_t)
        .map_err(invalid_stage_request)?;
    let ram_ra_request = SumcheckRamRaVirtualizationStateRequest::new(
        "Stage 6 RAM RA virtualization",
        ra_rows.clone(),
        ram_reduced.committed_address_chunks(config.committed_chunk_bits),
        ram_reduced.cycle.to_vec(),
        prefix.input_claims.ram_ra_virtualization,
        config.log_t,
        config.committed_chunk_bits,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let ram_ra_virtualization =
        backend.materialize_sumcheck_ram_ra_virtualization_state(&ram_ra_request)?;

    let instruction_ra_dimensions = config.instruction_ra_virtualization_dimensions;
    let instruction_read_raf = stage6_instruction_read_raf_point(stage5);
    let instruction_ra_request = SumcheckInstructionRaVirtualizationStateRequest::new(
        "Stage 6 instruction RA virtualization",
        ra_rows,
        instruction_read_raf.committed_address_chunks(config.committed_chunk_bits),
        instruction_read_raf.cycle.to_vec(),
        prefix.challenges.instruction_ra_gamma_powers.clone(),
        prefix.input_claims.instruction_ra_virtualization,
        config.log_t,
        config.committed_chunk_bits,
        instruction_ra_dimensions.num_virtual_ra_polys(),
        instruction_ra_dimensions.num_committed_per_virtual(),
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let instruction_ra_virtualization = backend
        .materialize_sumcheck_instruction_ra_virtualization_state(&instruction_ra_request)?;

    let [ram_read_write_cycle, ram_val_check_cycle, registers_read_write_cycle, registers_val_evaluation_cycle] =
        stage6_inc_claim_reduction_cycle_points(stage2, stage4, stage5, config.log_k)
            .map_err(invalid_stage_request)?
            .reversed_cycles();
    let inc_request = SumcheckIncClaimReductionStateRequest::new(
        "Stage 6 increment claim-reduction",
        inc_rows,
        ram_read_write_cycle,
        ram_val_check_cycle,
        registers_read_write_cycle,
        registers_val_evaluation_cycle,
        prefix.challenges.inc_gamma,
        prefix.input_claims.inc_claim_reduction,
        config.log_t,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let inc_claim_reduction =
        backend.materialize_sumcheck_inc_claim_reduction_state(&inc_request)?;

    Ok(Stage6BackendStates {
        bytecode_read_raf,
        booleanity,
        ram_hamming_booleanity,
        ram_ra_virtualization,
        instruction_ra_virtualization,
        inc_claim_reduction,
    })
}

fn invalid_stage_request(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}
