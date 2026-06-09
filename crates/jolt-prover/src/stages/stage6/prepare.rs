use common::jolt_device::JoltDevice;
#[cfg(feature = "field-inline")]
use jolt_backends::SumcheckBytecodeReadRafExtraStageValues;
#[cfg(feature = "field-inline")]
use jolt_backends::SumcheckFieldRegistersIncClaimReductionStateRequest;
use jolt_backends::{
    stage6_bytecode_pc_indices, stage6_hamming_weight, stage6_inc_rows, stage6_ra_rows,
    Stage6RegularBatchSumcheckBackend, SumcheckBooleanityStateRequest,
    SumcheckBytecodeReadRafStateRequest, SumcheckFieldRegistersReadWriteRow,
    SumcheckIncClaimReductionStateRequest, SumcheckInstructionRaVirtualizationStateRequest,
    SumcheckRamHammingBooleanityStateRequest, SumcheckRamRaVirtualizationStateRequest,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::bytecode as field_bytecode;
use jolt_claims::protocols::jolt::{
    formulas::{booleanity::BooleanityDimensions, bytecode},
    AdviceClaimReductionLayout, JoltFormulaDimensions,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6::{
    stage6_batch_input_claims, stage6_bytecode_cycle_points, stage6_bytecode_register_points,
    stage6_inc_claim_reduction_cycle_points, stage6_instruction_read_raf_point,
    stage6_stage1_cycle_binding, stage6_transcript_challenges, stage6_validate_dependencies,
};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6::{
    stage6_field_inline_bytecode_register_points,
    stage6_field_registers_inc_claim_reduction_cycle_points,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmStage6Rows};
use jolt_witness::WitnessProvider;

use super::batch::Stage6BatchContext;
use super::io::Stage6RegularBatchPrefixOutput;
use super::Stage6ProverConfig;
use crate::{JoltProverPreprocessing, ProofParameters, ProverError};

pub(super) const STAGE6_REGULAR_BATCH_OPT_IDS: &[&str] = &["cpu_stage6_regular_batch_sumcheck"];
#[cfg(feature = "field-inline")]
pub(super) const STAGE6_FIELD_INLINE_INC_OPT_IDS: &[&str] =
    &["cpu_field_inline_stage6_registers_inc_claim_reduction"];

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
    #[cfg(feature = "field-inline")]
    pub(super) field_registers_inc_claim_reduction: B::FieldRegistersIncClaimReductionState,
}

pub(crate) fn prover_config<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof_parameters: ProofParameters,
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
        advice_layouts(public_io, proof_parameters, log_t, committed_chunk_bits);
    let entry_bytecode_index = preprocessing
        .verifier
        .program
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
    .with_bytecode_context(
        preprocessing.verifier.program.bytecode.bytecode.clone(),
        entry_bytecode_index,
    ))
}

fn advice_layouts(
    public_io: &JoltDevice,
    proof_parameters: ProofParameters,
    log_t: usize,
    committed_chunk_bits: usize,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<AdviceClaimReductionLayout>,
) {
    let trusted = (!public_io.trusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_parameters.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted = (!public_io.untrusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_parameters.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });

    (trusted, untrusted)
}

pub(super) fn derive_regular_batch_prefix<F, T>(
    config: &Stage6ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage6RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    stage6_validate_dependencies(stage3).map_err(invalid_stage_request)?;

    let instruction_read_raf =
        jolt_verifier::stages::stage6::stage6_instruction_read_raf_point(stage5);
    let transcript_challenges = stage6_transcript_challenges(
        config.instruction_ra_virtualization_dimensions,
        instruction_read_raf.address,
        instruction_read_raf.cycle,
        config.committed_chunk_bits,
        stage5
            .output_claims
            .instruction_read_raf
            .lookup_table_flags
            .len(),
        transcript,
    );
    let input_claims = stage6_batch_input_claims(
        config.trace_dimensions(),
        config.bytecode_read_raf_dimensions,
        config.ram_ra_virtualization_dimensions,
        config.instruction_ra_virtualization_dimensions,
        config.trusted_advice_layout.as_ref(),
        config.untrusted_advice_layout.as_ref(),
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        transcript_challenges.input_claim_challenge_values(),
    )?;

    Ok(Stage6RegularBatchPrefixOutput::new(
        input_claims,
        transcript_challenges,
    ))
}

pub(super) fn bytecode_stage_values<F>(
    config: &Stage6ProverConfig,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
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
    #[cfg(feature = "field-inline")]
    let base_bytecode_rows = bytecode_context
        .rows
        .iter()
        .map(field_bytecode::base_jolt_bytecode_row)
        .collect::<Vec<_>>();
    #[cfg(feature = "field-inline")]
    let bytecode_rows = base_bytecode_rows.as_slice();
    #[cfg(not(feature = "field-inline"))]
    let bytecode_rows = bytecode_context.rows.as_slice();
    let stage_values = bytecode::read_raf_stage_values(bytecode::BytecodeReadRafStageValueInputs {
        bytecode: bytecode_rows,
        register_read_write_point: register_points.register_read_write_address,
        register_val_evaluation_point: register_points.register_val_evaluation_address,
        stage1_gammas: &prefix.challenges.stage1_gammas,
        stage2_gammas: &prefix.challenges.stage2_gammas,
        stage3_gammas: &prefix.challenges.stage3_gammas,
        stage4_gammas: &prefix.challenges.stage4_gammas,
        stage5_gammas: &prefix.challenges.stage5_gammas,
    });
    Ok(stage_values)
}

#[cfg(feature = "field-inline")]
pub(super) fn field_inline_bytecode_extra_stage_values<F>(
    config: &Stage6ProverConfig,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    stage1: &Stage1ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<Vec<SumcheckBytecodeReadRafExtraStageValues<F>>, ProverError>
where
    F: Field,
{
    let bytecode_context = config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;
    let field_log_k = config.field_inline.field_register_log_k;
    let field_register_points =
        stage6_field_inline_bytecode_register_points(stage4, stage5, field_log_k, config.log_t)
            .map_err(invalid_stage_request)?;
    let field_inline_bytecode = bytecode_context
        .rows
        .iter()
        .map(field_bytecode::field_inline_bytecode_row)
        .collect::<Vec<_>>();
    let field_stage_values = field_bytecode::read_raf_stage_values(
        field_bytecode::FieldInlineBytecodeReadRafStageValueInputs {
            bytecode: &field_inline_bytecode,
            field_register_read_write_point: field_register_points.read_write_address,
            field_register_val_evaluation_point: field_register_points.val_evaluation_address,
            stage1_gammas: &prefix.challenges.stage1_gammas,
            stage4_gammas: &prefix.challenges.stage4_gammas,
            stage5_gammas: &prefix.challenges.stage5_gammas,
        },
    );

    let mut stage1_values = Vec::with_capacity(bytecode_context.rows.len());
    let mut stage4_values = Vec::with_capacity(bytecode_context.rows.len());
    let mut stage5_values = Vec::with_capacity(bytecode_context.rows.len());
    for values in field_stage_values {
        stage1_values.push(values[0]);
        stage4_values.push(values[3]);
        stage5_values.push(values[4]);
    }
    let stage1_cycle = stage6_stage1_cycle_binding(stage1)
        .map_err(invalid_stage_request)?
        .iter()
        .rev()
        .copied()
        .collect();
    Ok(vec![
        SumcheckBytecodeReadRafExtraStageValues::new(0, stage1_values, stage1_cycle),
        SumcheckBytecodeReadRafExtraStageValues::new(
            3,
            stage4_values,
            field_register_points.read_write_cycle.to_vec(),
        ),
        SumcheckBytecodeReadRafExtraStageValues::new(
            4,
            stage5_values,
            field_register_points.val_evaluation_cycle.to_vec(),
        ),
    ])
}

pub(super) fn materialize_backend_states<F, W, B>(
    context: &Stage6BatchContext<'_, F, W>,
    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            unused_variables,
            reason = "field-inline rows are only used under field-inline"
        )
    )]
    field_register_rows: Option<Vec<SumcheckFieldRegistersReadWriteRow<F>>>,
    backend: &mut B,
) -> Result<Stage6BackendStates<B, F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: Stage6RegularBatchSumcheckBackend<F>,
{
    let stage6_rows = context.witness.stage6_rows()?;
    let (ra_rows, inc_rows) = (stage6_ra_rows(&stage6_rows), stage6_inc_rows(&stage6_rows));
    let bytecode_context = context.config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;

    let bytecode_request = SumcheckBytecodeReadRafStateRequest::new(
        "Stage 6 bytecode read-RAF",
        bytecode_stage_values(
            &context.config,
            context.prefix,
            context.stage4,
            context.stage5,
        )?,
        stage6_bytecode_pc_indices(&stage6_rows),
        stage6_bytecode_cycle_points(
            context.stage1,
            context.stage2,
            context.stage3,
            context.stage4,
            context.stage5,
        )
        .map_err(invalid_stage_request)?,
        context.prefix.challenges.bytecode_gamma_powers.clone(),
        bytecode_context.entry_bytecode_index,
        context.prefix.input_claims.bytecode_read_raf,
        context.config.log_t,
        context.config.bytecode_read_raf_dimensions.log_k(),
        context.config.committed_chunk_bits,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    #[cfg(feature = "field-inline")]
    let bytecode_request =
        bytecode_request.with_extra_stage_values(field_inline_bytecode_extra_stage_values(
            &context.config,
            context.prefix,
            context.stage1,
            context.stage4,
            context.stage5,
        )?);
    let bytecode_read_raf =
        backend.materialize_sumcheck_bytecode_read_raf_state(&bytecode_request)?;

    let booleanity_layout = context.config.booleanity_dimensions.layout;
    let booleanity_request = SumcheckBooleanityStateRequest::new(
        "Stage 6 booleanity",
        ra_rows.clone(),
        context
            .prefix
            .challenges
            .booleanity_reference
            .address
            .clone(),
        context.prefix.challenges.booleanity_reference.cycle.clone(),
        context.prefix.challenges.booleanity_gamma,
        context.prefix.input_claims.booleanity,
        context.config.log_t,
        context.config.committed_chunk_bits,
        booleanity_layout.instruction(),
        booleanity_layout.bytecode(),
        booleanity_layout.ram(),
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let booleanity = backend.materialize_sumcheck_booleanity_state(&booleanity_request)?;

    let hamming_request = SumcheckRamHammingBooleanityStateRequest::new(
        "Stage 6 RAM hamming booleanity",
        stage6_hamming_weight(&stage6_rows),
        stage6_stage1_cycle_binding(context.stage1)
            .map_err(invalid_stage_request)?
            .to_vec(),
        context.prefix.input_claims.ram_hamming_booleanity,
        context.config.log_t,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let ram_hamming_booleanity =
        backend.materialize_sumcheck_ram_hamming_booleanity_state(&hamming_request)?;

    let ram_reduced = context.ram_reduced_opening_point()?;
    let ram_ra_request = SumcheckRamRaVirtualizationStateRequest::new(
        "Stage 6 RAM RA virtualization",
        ra_rows.clone(),
        ram_reduced.committed_address_chunks(context.config.committed_chunk_bits),
        ram_reduced.cycle.to_vec(),
        context.prefix.input_claims.ram_ra_virtualization,
        context.config.log_t,
        context.config.committed_chunk_bits,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let ram_ra_virtualization =
        backend.materialize_sumcheck_ram_ra_virtualization_state(&ram_ra_request)?;

    let instruction_ra_dimensions = context.config.instruction_ra_virtualization_dimensions;
    let instruction_read_raf = stage6_instruction_read_raf_point(context.stage5);
    let instruction_ra_request = SumcheckInstructionRaVirtualizationStateRequest::new(
        "Stage 6 instruction RA virtualization",
        ra_rows,
        instruction_read_raf.committed_address_chunks(context.config.committed_chunk_bits),
        instruction_read_raf.cycle.to_vec(),
        context
            .prefix
            .challenges
            .instruction_ra_gamma_powers
            .clone(),
        context.prefix.input_claims.instruction_ra_virtualization,
        context.config.log_t,
        context.config.committed_chunk_bits,
        instruction_ra_dimensions.num_virtual_ra_polys(),
        instruction_ra_dimensions.num_committed_per_virtual(),
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let instruction_ra_virtualization = backend
        .materialize_sumcheck_instruction_ra_virtualization_state(&instruction_ra_request)?;

    let [ram_read_write_cycle, ram_val_check_cycle, registers_read_write_cycle, registers_val_evaluation_cycle] =
        stage6_inc_claim_reduction_cycle_points(
            context.stage2,
            context.stage4,
            context.stage5,
            context.config.log_k,
        )
        .map_err(invalid_stage_request)?
        .reversed_cycles();
    let inc_request = SumcheckIncClaimReductionStateRequest::new(
        "Stage 6 increment claim-reduction",
        inc_rows,
        ram_read_write_cycle,
        ram_val_check_cycle,
        registers_read_write_cycle,
        registers_val_evaluation_cycle,
        context.prefix.challenges.inc_gamma,
        context.prefix.input_claims.inc_claim_reduction,
        context.config.log_t,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let inc_claim_reduction =
        backend.materialize_sumcheck_inc_claim_reduction_state(&inc_request)?;

    #[cfg(feature = "field-inline")]
    let field_registers_inc_claim_reduction = {
        let field_register_rows = field_register_rows.ok_or_else(|| {
            invalid_stage_request("Stage 6 field-register rows are required under field-inline")
        })?;
        let field_log_k = context.config.field_inline.field_register_log_k;
        let [read_write_cycle, val_evaluation_cycle] =
            stage6_field_registers_inc_claim_reduction_cycle_points(
                context.stage4,
                context.stage5,
                field_log_k,
                context.config.log_t,
            )
            .map_err(invalid_stage_request)?
            .reversed_cycles();
        let request = SumcheckFieldRegistersIncClaimReductionStateRequest::new(
            "Stage 6 field-register increment claim-reduction",
            field_register_rows,
            read_write_cycle,
            val_evaluation_cycle,
            context.prefix.challenges.field_inc_gamma,
            context
                .prefix
                .input_claims
                .field_registers_inc_claim_reduction,
            context.config.log_t,
        )
        .with_optimization_ids(STAGE6_FIELD_INLINE_INC_OPT_IDS);
        backend.materialize_sumcheck_field_registers_inc_claim_reduction_state(&request)?
    };

    Ok(Stage6BackendStates {
        bytecode_read_raf,
        booleanity,
        ram_hamming_booleanity,
        ram_ra_virtualization,
        instruction_ra_virtualization,
        inc_claim_reduction,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction,
    })
}

fn invalid_stage_request(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}
