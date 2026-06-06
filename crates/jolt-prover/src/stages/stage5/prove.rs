use common::constants::INSTRUCTION_PHASES_THRESHOLD_LOG_T;
use jolt_backends::{
    Stage5ValueEvaluationSumcheckBackend, SumcheckBackend, SumcheckInstructionReadRafRow,
    SumcheckInstructionReadRafStateRequest, SumcheckRamRaClaimReductionStateRequest,
    SumcheckRamReadWriteRow, SumcheckRegisterRead, SumcheckRegisterWrite,
    SumcheckRegistersReadWriteRow, SumcheckRegistersValEvaluationStateRequest,
};
#[cfg(feature = "field-inline")]
use jolt_backends::{
    SumcheckFieldRegisterRead, SumcheckFieldRegisterWrite, SumcheckFieldRegistersReadWriteRow,
    SumcheckFieldRegistersValEvaluationStateRequest,
};
use jolt_claims::protocols::jolt::formulas::dimensions::REGISTER_ADDRESS_BITS;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::{
    try_eq_mle, IdentityPolynomial, LtPolynomial, MultilinearEvaluation, OperandPolynomial,
    OperandSide, Point, UnivariatePoly,
};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage5::inputs::{
    FieldInlineStage5Claims, FieldRegistersValEvaluationOutputOpeningClaims,
};
use jolt_verifier::stages::stage5::inputs::{
    InstructionReadRafOutputOpeningClaims, RamRaClaimReductionOutputOpeningClaims,
    RegistersValEvaluationOutputOpeningClaims,
};
use jolt_verifier::stages::stage5::outputs::{
    Stage5ClearOutput, Stage5PublicOutput, VerifiedInstructionReadRafSumcheck, VerifiedStage5Batch,
    VerifiedStage5Sumcheck,
};
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage4::Stage4ClearOutput};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
};
use jolt_witness::protocols::jolt_vm::{
    JoltVmRegisterReadWriteRow, JoltVmRegisterReadWriteRows, JoltVmStage2Rows,
    JoltVmStage2TraceRow, JoltVmStage5InstructionReadRafRows,
};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckBuilder;
use crate::ProverError;

use super::{
    input::Stage5ProverConfig,
    output::{
        stage5_output_openings_from_evaluations, Stage5RegularBatchInputClaims,
        Stage5RegularBatchOutputOpeningClaims, Stage5RegularBatchPrefixOutput,
    },
    request::build_stage5_output_opening_evaluation_request,
};

use super::input::Stage5ProverInput;
#[cfg(feature = "zk")]
use super::output::Stage5CommittedBoundaryOutput;
use super::output::{
    Stage5ProverOutput, Stage5RegularBatchExpectedOutputs, Stage5RegularBatchProofOutput,
};
#[cfg(not(feature = "field-inline"))]
const STAGE5_BATCH_COEFFICIENTS: usize = 3;
#[cfg(feature = "field-inline")]
const STAGE5_BATCH_COEFFICIENTS: usize = 4;

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
fn timed_stage5<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(all(not(feature = "frontier-harness"), not(feature = "field-inline")))]
fn timed_stage5<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
fn timed_stage5_value<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(all(not(feature = "frontier-harness"), not(feature = "field-inline")))]
fn timed_stage5_value<T>(_label: &'static str, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
fn timed_stage5_accumulate<T>(accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    *accumulator += start.elapsed().as_secs_f64() * 1000.0;
    result
}

#[cfg(all(not(feature = "frontier-harness"), not(feature = "field-inline")))]
fn timed_stage5_accumulate<T>(_accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
fn record_stage5_accumulated(label: &'static str, time_ms: f64) {
    crate::timing::record_stage_timing(label, time_ms);
}

#[cfg(all(not(feature = "frontier-harness"), not(feature = "field-inline")))]
fn record_stage5_accumulated(_label: &'static str, _time_ms: f64) {}

/// Canonical Stage 5 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage5/verify.rs` in prover order: derive
/// the instruction/RAM gammas, prove the instruction read-RAF, RAM-RA reduction,
/// register value-evaluation, and optional field-register value-evaluation
/// batched sumcheck, then assemble the verifier-owned Stage 5 proof, claims, and
/// clear output for Stage 6 and later stages. ZK Stage 5 prover assembly is a
/// separate committed-boundary path.
#[cfg(not(feature = "field-inline"))]
pub fn prove<F, W, B, T, C>(
    input: Stage5ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage5ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix =
        derive_stage5_regular_batch_prefix(input.config, input.stage2, input.stage4, transcript)?;
    let proof_output = prove_stage5_transparent_sumchecks::<F, W, B, T, C>(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage4,
        &prefix,
        transcript,
    )?;

    let claims = proof_output.output_openings.clone();
    let public = Stage5PublicOutput {
        challenges: proof_output.sumcheck_point.clone(),
        batching_coefficients: proof_output.batching_coefficients.clone(),
        instruction_gamma: prefix.instruction_gamma,
        ram_gamma: prefix.ram_gamma,
    };
    let verifier_output = Stage5ClearOutput {
        public,
        output_claims: claims.clone(),
        batch: VerifiedStage5Batch {
            batching_coefficients: proof_output.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(proof_output.sumcheck_point.clone()),
            sumcheck_final_claim: proof_output.sumcheck_final_claim,
            expected_final_claim: proof_output.expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: prefix.input_claims.instruction_read_raf,
                sumcheck_point: proof_output.instruction_read_raf_sumcheck_point.clone(),
                r_address: proof_output.instruction_read_raf_r_address.clone(),
                r_cycle: proof_output.instruction_read_raf_r_cycle.clone(),
                full_opening_point: proof_output.instruction_read_raf_full_opening_point.clone(),
                lookup_table_flag_opening_point: proof_output
                    .instruction_lookup_table_flag_opening_point
                    .clone(),
                instruction_ra_opening_points: proof_output.instruction_ra_opening_points.clone(),
                instruction_raf_flag_opening_point: proof_output
                    .instruction_raf_flag_opening_point
                    .clone(),
                expected_output_claim: proof_output.expected_outputs.instruction_read_raf,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.ram_ra_claim_reduction,
                sumcheck_point: proof_output.ram_ra_claim_reduction_sumcheck_point.clone(),
                opening_point: proof_output.ram_ra_claim_reduction_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.ram_ra_claim_reduction,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.registers_val_evaluation,
                sumcheck_point: proof_output.registers_val_evaluation_sumcheck_point.clone(),
                opening_point: proof_output.registers_val_evaluation_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.registers_val_evaluation,
            },
        },
    };

    Ok(Stage5ProverOutput {
        stage5_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
pub fn prove_committed_boundary<F, W, B, T, VC>(
    input: Stage5ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage5CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage5_committed_checked(input.config, input.checked)?;
    let prefix =
        derive_stage5_regular_batch_prefix(input.config, input.stage2, input.stage4, transcript)?;
    prove_stage5_committed_specialized_regular_batch_sumcheck::<F, W, B, T, VC>(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage4,
        &prefix,
        transcript,
        vc_setup,
    )
}

#[cfg(feature = "field-inline")]
pub fn prove<F, W, FI, B, T, C>(
    input: Stage5ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage5ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix =
        derive_stage5_regular_batch_prefix(input.config, input.stage2, input.stage4, transcript)?;
    let proof_output = prove_stage5_transparent_sumchecks::<F, W, FI, B, T, C>(
        input.config,
        input.witness,
        input.field_inline_witness,
        backend,
        input.stage2,
        input.stage4,
        &prefix,
        transcript,
    )?;

    let claims = proof_output.output_openings.clone();
    let public = Stage5PublicOutput {
        challenges: proof_output.sumcheck_point.clone(),
        batching_coefficients: proof_output.batching_coefficients.clone(),
        instruction_gamma: prefix.instruction_gamma,
        ram_gamma: prefix.ram_gamma,
    };
    let verifier_output = Stage5ClearOutput {
        public,
        output_claims: claims.clone(),
        batch: VerifiedStage5Batch {
            batching_coefficients: proof_output.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(proof_output.sumcheck_point.clone()),
            sumcheck_final_claim: proof_output.sumcheck_final_claim,
            expected_final_claim: proof_output.expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: prefix.input_claims.instruction_read_raf,
                sumcheck_point: proof_output.instruction_read_raf_sumcheck_point.clone(),
                r_address: proof_output.instruction_read_raf_r_address.clone(),
                r_cycle: proof_output.instruction_read_raf_r_cycle.clone(),
                full_opening_point: proof_output.instruction_read_raf_full_opening_point.clone(),
                lookup_table_flag_opening_point: proof_output
                    .instruction_lookup_table_flag_opening_point
                    .clone(),
                instruction_ra_opening_points: proof_output.instruction_ra_opening_points.clone(),
                instruction_raf_flag_opening_point: proof_output
                    .instruction_raf_flag_opening_point
                    .clone(),
                expected_output_claim: proof_output.expected_outputs.instruction_read_raf,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.ram_ra_claim_reduction,
                sumcheck_point: proof_output.ram_ra_claim_reduction_sumcheck_point.clone(),
                opening_point: proof_output.ram_ra_claim_reduction_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.ram_ra_claim_reduction,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.registers_val_evaluation,
                sumcheck_point: proof_output.registers_val_evaluation_sumcheck_point.clone(),
                opening_point: proof_output.registers_val_evaluation_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.registers_val_evaluation,
            },
            field_registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.field_registers_val_evaluation,
                sumcheck_point: proof_output
                    .field_registers_val_evaluation_sumcheck_point
                    .clone(),
                opening_point: proof_output
                    .field_registers_val_evaluation_opening_point
                    .clone(),
                expected_output_claim: proof_output.expected_outputs.field_registers_val_evaluation,
            },
        },
    };

    Ok(Stage5ProverOutput {
        stage5_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
pub fn prove_committed_boundary<F, W, FI, B, T, VC>(
    input: Stage5ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage5CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage5_committed_checked(input.config, input.checked)?;
    let prefix =
        derive_stage5_regular_batch_prefix(input.config, input.stage2, input.stage4, transcript)?;
    prove_stage5_committed_specialized_regular_batch_sumcheck::<F, W, FI, B, T, VC>(
        input.config,
        input.witness,
        input.field_inline_witness,
        backend,
        input.stage2,
        input.stage4,
        &prefix,
        transcript,
        vc_setup,
    )
}

pub fn derive_stage5_regular_batch_prefix<F, T>(
    config: Stage5ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    validate_stage5_dependencies(config, stage2, stage4)?;

    let instruction_gamma = transcript.challenge_scalar();
    let instruction_gamma2 = instruction_gamma * instruction_gamma;
    let ram_gamma = transcript.challenge_scalar();
    let ram_gamma2 = ram_gamma * ram_gamma;

    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output;
    let reduced_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);

    Ok(Stage5RegularBatchPrefixOutput {
        input_claims: Stage5RegularBatchInputClaims {
            instruction_read_raf: reduced_lookup_output
                + instruction_gamma
                    * stage2
                        .output_claims
                        .instruction_claim_reduction
                        .left_lookup_operand
                + instruction_gamma2
                    * stage2
                        .output_claims
                        .instruction_claim_reduction
                        .right_lookup_operand,
            ram_ra_claim_reduction: stage2.output_claims.ram_raf_evaluation
                + ram_gamma * stage2.output_claims.ram_read_write.ra
                + ram_gamma2 * stage4.output_claims.ram_val_check.ram_ra,
            registers_val_evaluation: stage4.output_claims.registers_read_write.registers_val,
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation: stage4
                .output_claims
                .field_inline
                .field_registers_read_write
                .field_registers_val,
        },
        instruction_gamma,
        ram_gamma,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 5 output openings have distinct verifier-derived points."
)]
pub fn evaluate_stage5_output_openings<F, W, B>(
    config: Stage5ProverConfig,
    witness: &W,
    backend: &mut B,
    instruction_lookup_table_flag_opening_point: Vec<F>,
    instruction_ra_opening_points: Vec<Vec<F>>,
    instruction_raf_flag_opening_point: Vec<F>,
    ram_ra_claim_reduction_opening_point: Vec<F>,
    registers_val_evaluation_opening_point: Vec<F>,
) -> Result<Stage5RegularBatchOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let request = build_stage5_output_opening_evaluation_request(
        config,
        witness,
        instruction_lookup_table_flag_opening_point,
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_opening_point,
        registers_val_evaluation_opening_point,
    )?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    stage5_output_openings_from_evaluations(&request, evaluations)
}

#[cfg(not(feature = "field-inline"))]
pub fn prove_stage5_transparent_sumchecks<F, W, B, T, C>(
    config: Stage5ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let proof_output = prove_stage5_specialized_regular_batch_sumcheck::<F, W, B, T, C>(
        config, witness, backend, stage2, stage4, prefix, transcript,
    )?;
    append_stage5_opening_claims(transcript, &proof_output.output_openings);
    Ok(proof_output)
}

#[cfg(feature = "field-inline")]
#[expect(
    clippy::too_many_arguments,
    reason = "Stage 5 field-inline sumcheck inputs mirror the verifier dependency boundary."
)]
pub fn prove_stage5_transparent_sumchecks<F, W, FI, B, T, C>(
    config: Stage5ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let proof_output = prove_stage5_specialized_regular_batch_sumcheck::<F, W, FI, B, T, C>(
        config,
        witness,
        field_inline_witness,
        backend,
        stage2,
        stage4,
        prefix,
        transcript,
    )?;
    append_stage5_opening_claims(transcript, &proof_output.output_openings);
    Ok(proof_output)
}

#[cfg(not(feature = "field-inline"))]
fn prove_stage5_specialized_regular_batch_sumcheck<F, W, B, T, C>(
    config: Stage5ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    timed_stage5("stage5.validate", || {
        validate_stage5_dependencies(config, stage2, stage4)
    })?;

    let instruction_address_bits = config
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    let instruction_ra_count = config
        .instruction_read_raf_dimensions
        .num_virtual_ra_polys();
    let instruction_ra_chunk_bits = instruction_address_bits
        .checked_div(instruction_ra_count)
        .ok_or_else(|| {
            invalid_sumcheck_output(
                "Stage 5 instruction read-RAF config has no virtual RA polynomials",
            )
        })?;
    if instruction_ra_chunk_bits * instruction_ra_count != instruction_address_bits {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction address bit count {instruction_address_bits} is not divisible by virtual RA count {instruction_ra_count}",
        )));
    }

    let instruction_rows = timed_stage5("stage5.rows.instruction_read_raf", || {
        stage5_instruction_read_raf_rows(config, witness)
    })?;
    let instruction_request = timed_stage5_value("stage5.request.instruction_read_raf", || {
        SumcheckInstructionReadRafStateRequest::new(
            "stage5.instruction_read_raf",
            instruction_rows,
            stage2
                .batch
                .instruction_claim_reduction
                .opening_point
                .clone(),
            prefix.instruction_gamma,
            prefix.input_claims.instruction_read_raf,
            config.log_t,
            instruction_address_bits,
            instruction_ra_chunk_bits,
            instruction_sumcheck_phases(config.log_t),
        )
        .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"])
    });

    let (_fixed_ram_raf_address, fixed_ram_raf_cycle_point) = stage2
        .batch
        .ram_raf_evaluation
        .opening_point
        .split_at(config.log_k);
    let (fixed_ram_address_point, fixed_ram_read_write_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);
    let (_fixed_ram_val_check_address, fixed_ram_val_check_cycle_point) = stage4
        .batch
        .ram_val_check
        .opening_point
        .split_at(config.log_k);
    let (fixed_register_address_point, fixed_register_cycle_point) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);

    let ram_rows = timed_stage5("stage5.rows.ram", || stage5_ram_rows(witness))?;
    let ram_request = timed_stage5_value("stage5.request.ram_ra_claim_reduction", || {
        SumcheckRamRaClaimReductionStateRequest::new(
            "stage5.ram_ra_claim_reduction",
            ram_rows,
            fixed_ram_address_point.to_vec(),
            fixed_ram_raf_cycle_point.to_vec(),
            fixed_ram_read_write_cycle_point.to_vec(),
            fixed_ram_val_check_cycle_point.to_vec(),
            prefix.ram_gamma,
            prefix.input_claims.ram_ra_claim_reduction,
            config.log_t,
            config.log_k,
        )
        .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"])
    });
    let register_rows = timed_stage5("stage5.rows.registers", || stage5_register_rows(witness))?;
    let registers_request = timed_stage5_value("stage5.request.registers_val_evaluation", || {
        SumcheckRegistersValEvaluationStateRequest::new(
            "stage5.registers_val_evaluation",
            register_rows,
            fixed_register_address_point.to_vec(),
            fixed_register_cycle_point.to_vec(),
            prefix.input_claims.registers_val_evaluation,
            config.log_t,
            REGISTER_ADDRESS_BITS,
        )
        .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"])
    });
    let mut instruction_state = timed_stage5("stage5.materialize.instruction_read_raf", || {
        backend.materialize_sumcheck_instruction_read_raf_state(&instruction_request)
    })?;
    let mut ram_state = timed_stage5("stage5.materialize.ram_ra_claim_reduction", || {
        backend.materialize_sumcheck_ram_ra_claim_reduction_state(&ram_request)
    })?;
    let mut registers_state = timed_stage5("stage5.materialize.registers_val_evaluation", || {
        backend.materialize_sumcheck_registers_val_evaluation_state(&registers_request)
    })?;

    let batching_coefficients = timed_stage5_value("stage5.batching_challenges", || {
        append_sumcheck_claim(transcript, &prefix.input_claims.instruction_read_raf);
        append_sumcheck_claim(transcript, &prefix.input_claims.ram_ra_claim_reduction);
        append_sumcheck_claim(transcript, &prefix.input_claims.registers_val_evaluation);
        (0..STAGE5_BATCH_COEFFICIENTS)
            .map(|_| transcript.challenge_scalar())
            .collect::<Vec<_>>()
    });
    let [instruction_coefficient, ram_coefficient, registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 5 batch expected exactly three batching coefficients",
        ));
    };
    let coefficients = Stage5BatchCoefficients {
        instruction_read_raf: *instruction_coefficient,
        ram_ra_claim_reduction: *ram_coefficient,
        registers_val_evaluation: *registers_coefficient,
    };

    let front_padding_rounds = instruction_address_bits;
    let mut individual_claims = [
        prefix.input_claims.instruction_read_raf,
        prefix
            .input_claims
            .ram_ra_claim_reduction
            .mul_pow_2(front_padding_rounds),
        prefix
            .input_claims
            .registers_val_evaluation
            .mul_pow_2(front_padding_rounds),
    ];
    let mut running_claim = coefficients.instruction_read_raf * individual_claims[0]
        + coefficients.ram_ra_claim_reduction * individual_claims[1]
        + coefficients.registers_val_evaluation * individual_claims[2];
    let max_rounds = front_padding_rounds + config.log_t;
    let mut round_polynomials = Vec::with_capacity(max_rounds);
    let mut sumcheck_point = Vec::with_capacity(max_rounds);
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut round_instruction_read_raf_ms = 0.0;
    let mut round_ram_ra_claim_reduction_ms = 0.0;
    let mut round_registers_val_evaluation_ms = 0.0;
    let mut round_combine_ms = 0.0;
    let mut round_transcript_ms = 0.0;
    let mut bind_instruction_read_raf_ms = 0.0;
    let mut bind_ram_ra_claim_reduction_ms = 0.0;
    let mut bind_registers_val_evaluation_ms = 0.0;

    for round in 0..max_rounds {
        let instruction_poly = timed_stage5_accumulate(&mut round_instruction_read_raf_ms, || {
            backend.evaluate_sumcheck_instruction_read_raf_round(
                &instruction_state,
                individual_claims[0],
            )
        })?;
        let ram_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            timed_stage5_accumulate(&mut round_ram_ra_claim_reduction_ms, || {
                backend.evaluate_sumcheck_ram_ra_claim_reduction_round(
                    &ram_state,
                    individual_claims[1],
                )
            })?
        };
        let registers_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[2] * two_inv])
        } else {
            timed_stage5_accumulate(&mut round_registers_val_evaluation_ms, || {
                backend.evaluate_sumcheck_registers_val_evaluation_round(
                    &registers_state,
                    individual_claims[2],
                )
            })?
        };
        let round_poly = timed_stage5_accumulate(&mut round_combine_ms, || {
            let mut round_poly = UnivariatePoly::zero();
            round_poly += &(&instruction_poly * coefficients.instruction_read_raf);
            round_poly += &(&ram_poly * coefficients.ram_ra_claim_reduction);
            round_poly += &(&registers_poly * coefficients.registers_val_evaluation);
            round_poly
        });
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 5 batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = timed_stage5_accumulate(&mut round_transcript_ms, || {
            CompressedLabeledRoundPoly::sumcheck(&round_poly).append_to_transcript(transcript);
            transcript.challenge()
        });
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        individual_claims[0] = instruction_poly.evaluate(challenge);
        individual_claims[1] = if round < front_padding_rounds {
            individual_claims[1] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        individual_claims[2] = if round < front_padding_rounds {
            individual_claims[2] * two_inv
        } else {
            registers_poly.evaluate(challenge)
        };
        timed_stage5_accumulate(&mut bind_instruction_read_raf_ms, || {
            backend.bind_sumcheck_instruction_read_raf_state(&mut instruction_state, challenge)
        })?;
        if round >= front_padding_rounds {
            timed_stage5_accumulate(&mut bind_ram_ra_claim_reduction_ms, || {
                backend.bind_sumcheck_ram_ra_claim_reduction_state(&mut ram_state, challenge)
            })?;
            timed_stage5_accumulate(&mut bind_registers_val_evaluation_ms, || {
                backend
                    .bind_sumcheck_registers_val_evaluation_state(&mut registers_state, challenge)
            })?;
        }
        round_polynomials.push(round_poly.compress());
    }
    record_stage5_accumulated(
        "stage5.rounds.instruction_read_raf",
        round_instruction_read_raf_ms,
    );
    record_stage5_accumulated(
        "stage5.rounds.ram_ra_claim_reduction",
        round_ram_ra_claim_reduction_ms,
    );
    record_stage5_accumulated(
        "stage5.rounds.registers_val_evaluation",
        round_registers_val_evaluation_ms,
    );
    record_stage5_accumulated("stage5.rounds.combine", round_combine_ms);
    record_stage5_accumulated("stage5.rounds.transcript", round_transcript_ms);
    record_stage5_accumulated(
        "stage5.bind.instruction_read_raf",
        bind_instruction_read_raf_ms,
    );
    record_stage5_accumulated(
        "stage5.bind.ram_ra_claim_reduction",
        bind_ram_ra_claim_reduction_ms,
    );
    record_stage5_accumulated(
        "stage5.bind.registers_val_evaluation",
        bind_registers_val_evaluation_ms,
    );

    let (
        instruction_sumcheck_point,
        instruction_opening_point,
        instruction_lookup_table_flag_opening_point,
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_sumcheck_point,
        ram_ra_claim_reduction_opening_point,
        registers_val_evaluation_sumcheck_point,
        registers_val_evaluation_opening_point,
    ) = timed_stage5("stage5.derived_points", || {
        let instruction_sumcheck_point = sumcheck_point.clone();
        let instruction_opening_point = config
            .instruction_read_raf_dimensions
            .opening_point(&instruction_sumcheck_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let instruction_lookup_table_flag_opening_point = instruction_opening_point.r_cycle.clone();
        let instruction_raf_flag_opening_point = instruction_opening_point.r_cycle.clone();
        let instruction_ra_opening_points = instruction_opening_point
            .r_address
            .chunks(instruction_ra_chunk_bits)
            .map(|r_address_chunk| {
                [
                    r_address_chunk,
                    instruction_opening_point.r_cycle.as_slice(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();
        let ram_ra_claim_reduction_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
        let ram_ra_claim_reduction_cycle_point =
            reversed_point(&ram_ra_claim_reduction_sumcheck_point);
        let ram_ra_claim_reduction_opening_point = [
            fixed_ram_address_point,
            ram_ra_claim_reduction_cycle_point.as_slice(),
        ]
        .concat();
        let registers_val_evaluation_sumcheck_point =
            sumcheck_point[front_padding_rounds..].to_vec();
        let registers_val_evaluation_cycle_point =
            reversed_point(&registers_val_evaluation_sumcheck_point);
        let registers_val_evaluation_opening_point = [
            fixed_register_address_point,
            registers_val_evaluation_cycle_point.as_slice(),
        ]
        .concat();
        Ok::<_, ProverError>((
            instruction_sumcheck_point,
            instruction_opening_point,
            instruction_lookup_table_flag_opening_point,
            instruction_ra_opening_points,
            instruction_raf_flag_opening_point,
            ram_ra_claim_reduction_sumcheck_point,
            ram_ra_claim_reduction_opening_point,
            registers_val_evaluation_sumcheck_point,
            registers_val_evaluation_opening_point,
        ))
    })?;

    let instruction_output = timed_stage5("stage5.output.instruction_read_raf", || {
        backend.output_sumcheck_instruction_read_raf_state(&instruction_state)
    })?;
    let instruction_internal_final = instruction_output.final_claim;
    let ram_output = timed_stage5("stage5.output.ram_ra_claim_reduction", || {
        backend.output_sumcheck_ram_ra_claim_reduction_state(&ram_state)
    })?;
    let registers_output = timed_stage5("stage5.output.registers_val_evaluation", || {
        backend.output_sumcheck_registers_val_evaluation_state(&registers_state)
    })?;
    let output_openings = Stage5RegularBatchOutputOpeningClaims {
        instruction_read_raf: InstructionReadRafOutputOpeningClaims {
            lookup_table_flags: instruction_output.lookup_table_flags,
            instruction_ra: instruction_output.instruction_ra,
            instruction_raf_flag: instruction_output.instruction_raf_flag,
        },
        ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
        },
        registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims {
            rd_inc: registers_output.rd_inc,
            rd_wa: registers_output.rd_wa,
        },
    };
    let (instruction_expected, ram_expected, registers_expected) =
        timed_stage5("stage5.expected_outputs", || {
            let instruction_expected = expected_instruction_read_raf_output(
                config,
                prefix.instruction_gamma,
                &stage2.batch.instruction_claim_reduction.opening_point,
                &instruction_opening_point.r_address,
                &instruction_opening_point.r_cycle,
                &output_openings.instruction_read_raf,
            )?;
            let ram_expected = expected_ram_ra_claim_reduction_output(
                config,
                prefix.ram_gamma,
                fixed_ram_raf_cycle_point,
                fixed_ram_read_write_cycle_point,
                fixed_ram_val_check_cycle_point,
                &ram_ra_claim_reduction_opening_point,
                output_openings.ram_ra_claim_reduction.ram_ra,
            )?;
            let registers_expected = expected_registers_val_evaluation_output(
                config,
                fixed_register_cycle_point,
                &registers_val_evaluation_opening_point,
                output_openings.registers_val_evaluation.rd_inc,
                output_openings.registers_val_evaluation.rd_wa,
            )?;
            Ok::<_, ProverError>((instruction_expected, ram_expected, registers_expected))
        })?;
    let expected_final_claim = coefficients.instruction_read_raf * instruction_expected
        + coefficients.ram_ra_claim_reduction * ram_expected
        + coefficients.registers_val_evaluation * registers_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 batch final claim did not match output openings: instruction={}, ram={}, registers={}, instruction_internal={}, instruction_expected_internal={}",
            individual_claims[0] == instruction_expected,
            individual_claims[1] == ram_expected,
            individual_claims[2] == registers_expected,
            instruction_internal_final == individual_claims[0],
            instruction_internal_final == instruction_expected,
        )));
    }

    Ok(Stage5RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        output_openings,
        expected_outputs: Stage5RegularBatchExpectedOutputs {
            instruction_read_raf: instruction_expected,
            ram_ra_claim_reduction: ram_expected,
            registers_val_evaluation: registers_expected,
        },
        batching_coefficients,
        sumcheck_point,
        sumcheck_final_claim: running_claim,
        expected_final_claim,
        instruction_read_raf_sumcheck_point: instruction_sumcheck_point,
        instruction_read_raf_r_address: instruction_opening_point.r_address,
        instruction_read_raf_r_cycle: instruction_opening_point.r_cycle,
        instruction_read_raf_full_opening_point: instruction_opening_point.opening_point,
        instruction_lookup_table_flag_opening_point,
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_sumcheck_point,
        ram_ra_claim_reduction_opening_point,
        registers_val_evaluation_sumcheck_point,
        registers_val_evaluation_opening_point,
    })
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
#[expect(clippy::too_many_arguments)]
fn prove_stage5_committed_specialized_regular_batch_sumcheck<F, W, B, T, VC>(
    config: Stage5ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage5CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage5_dependencies(config, stage2, stage4)?;
    let instruction_address_bits = config
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    let instruction_ra_count = config
        .instruction_read_raf_dimensions
        .num_virtual_ra_polys();
    let instruction_ra_chunk_bits = instruction_address_bits
        .checked_div(instruction_ra_count)
        .ok_or_else(|| {
            invalid_sumcheck_output(
                "Stage 5 instruction read-RAF config has no virtual RA polynomials",
            )
        })?;
    if instruction_ra_chunk_bits * instruction_ra_count != instruction_address_bits {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction address bit count {instruction_address_bits} is not divisible by virtual RA count {instruction_ra_count}",
        )));
    }

    let instruction_request = SumcheckInstructionReadRafStateRequest::new(
        "stage5.instruction_read_raf",
        stage5_instruction_read_raf_rows(config, witness)?,
        stage2
            .batch
            .instruction_claim_reduction
            .opening_point
            .clone(),
        prefix.instruction_gamma,
        prefix.input_claims.instruction_read_raf,
        config.log_t,
        instruction_address_bits,
        instruction_ra_chunk_bits,
        instruction_sumcheck_phases(config.log_t),
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let (_fixed_ram_raf_address, fixed_ram_raf_cycle_point) = stage2
        .batch
        .ram_raf_evaluation
        .opening_point
        .split_at(config.log_k);
    let (fixed_ram_address_point, fixed_ram_read_write_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);
    let (_fixed_ram_val_check_address, fixed_ram_val_check_cycle_point) = stage4
        .batch
        .ram_val_check
        .opening_point
        .split_at(config.log_k);
    let (fixed_register_address_point, fixed_register_cycle_point) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);

    let ram_request = SumcheckRamRaClaimReductionStateRequest::new(
        "stage5.ram_ra_claim_reduction",
        stage5_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_raf_cycle_point.to_vec(),
        fixed_ram_read_write_cycle_point.to_vec(),
        fixed_ram_val_check_cycle_point.to_vec(),
        prefix.ram_gamma,
        prefix.input_claims.ram_ra_claim_reduction,
        config.log_t,
        config.log_k,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let registers_request = SumcheckRegistersValEvaluationStateRequest::new(
        "stage5.registers_val_evaluation",
        stage5_register_rows(witness)?,
        fixed_register_address_point.to_vec(),
        fixed_register_cycle_point.to_vec(),
        prefix.input_claims.registers_val_evaluation,
        config.log_t,
        REGISTER_ADDRESS_BITS,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let mut instruction_state =
        backend.materialize_sumcheck_instruction_read_raf_state(&instruction_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_ra_claim_reduction_state(&ram_request)?;
    let mut registers_state =
        backend.materialize_sumcheck_registers_val_evaluation_state(&registers_request)?;

    let batching_coefficients = (0..STAGE5_BATCH_COEFFICIENTS)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [instruction_coefficient, ram_coefficient, registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 5 batch expected exactly three batching coefficients",
        ));
    };
    let coefficients = Stage5BatchCoefficients {
        instruction_read_raf: *instruction_coefficient,
        ram_ra_claim_reduction: *ram_coefficient,
        registers_val_evaluation: *registers_coefficient,
    };
    let front_padding_rounds = instruction_address_bits;
    let mut individual_claims = [
        prefix.input_claims.instruction_read_raf,
        prefix
            .input_claims
            .ram_ra_claim_reduction
            .mul_pow_2(front_padding_rounds),
        prefix
            .input_claims
            .registers_val_evaluation
            .mul_pow_2(front_padding_rounds),
    ];
    let mut running_claim = coefficients.instruction_read_raf * individual_claims[0]
        + coefficients.ram_ra_claim_reduction * individual_claims[1]
        + coefficients.registers_val_evaluation * individual_claims[2];
    let max_rounds = front_padding_rounds + config.log_t;
    let mut sumcheck_point = Vec::with_capacity(max_rounds);
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, max_rounds)?;
    let two_inv = F::from_u64(2).inv_or_zero();

    for round in 0..max_rounds {
        let instruction_poly = backend.evaluate_sumcheck_instruction_read_raf_round(
            &instruction_state,
            individual_claims[0],
        )?;
        let ram_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend
                .evaluate_sumcheck_ram_ra_claim_reduction_round(&ram_state, individual_claims[1])?
        };
        let registers_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[2] * two_inv])
        } else {
            backend.evaluate_sumcheck_registers_val_evaluation_round(
                &registers_state,
                individual_claims[2],
            )?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&instruction_poly * coefficients.instruction_read_raf);
        round_poly += &(&ram_poly * coefficients.ram_ra_claim_reduction);
        round_poly += &(&registers_poly * coefficients.registers_val_evaluation);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 5 committed batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = builder.commit_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        individual_claims[0] = instruction_poly.evaluate(challenge);
        individual_claims[1] = if round < front_padding_rounds {
            individual_claims[1] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        individual_claims[2] = if round < front_padding_rounds {
            individual_claims[2] * two_inv
        } else {
            registers_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_instruction_read_raf_state(&mut instruction_state, challenge)?;
        if round >= front_padding_rounds {
            backend.bind_sumcheck_ram_ra_claim_reduction_state(&mut ram_state, challenge)?;
            backend
                .bind_sumcheck_registers_val_evaluation_state(&mut registers_state, challenge)?;
        }
    }

    let instruction_sumcheck_point = sumcheck_point.clone();
    let instruction_opening_point = config
        .instruction_read_raf_dimensions
        .opening_point(&instruction_sumcheck_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let instruction_lookup_table_flag_opening_point = instruction_opening_point.r_cycle.clone();
    let instruction_raf_flag_opening_point = instruction_opening_point.r_cycle.clone();
    let instruction_ra_opening_points = instruction_opening_point
        .r_address
        .chunks(instruction_ra_chunk_bits)
        .map(|r_address_chunk| {
            [
                r_address_chunk,
                instruction_opening_point.r_cycle.as_slice(),
            ]
            .concat()
        })
        .collect::<Vec<_>>();
    let ram_ra_claim_reduction_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
    let ram_ra_claim_reduction_cycle_point = reversed_point(&ram_ra_claim_reduction_sumcheck_point);
    let ram_ra_claim_reduction_opening_point = [
        fixed_ram_address_point,
        ram_ra_claim_reduction_cycle_point.as_slice(),
    ]
    .concat();
    let registers_val_evaluation_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
    let registers_val_evaluation_cycle_point =
        reversed_point(&registers_val_evaluation_sumcheck_point);
    let registers_val_evaluation_opening_point = [
        fixed_register_address_point,
        registers_val_evaluation_cycle_point.as_slice(),
    ]
    .concat();

    let instruction_output =
        backend.output_sumcheck_instruction_read_raf_state(&instruction_state)?;
    let instruction_internal_final = instruction_output.final_claim;
    let ram_output = backend.output_sumcheck_ram_ra_claim_reduction_state(&ram_state)?;
    let registers_output =
        backend.output_sumcheck_registers_val_evaluation_state(&registers_state)?;
    let output_openings = Stage5RegularBatchOutputOpeningClaims {
        instruction_read_raf: InstructionReadRafOutputOpeningClaims {
            lookup_table_flags: instruction_output.lookup_table_flags,
            instruction_ra: instruction_output.instruction_ra,
            instruction_raf_flag: instruction_output.instruction_raf_flag,
        },
        ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
        },
        registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims {
            rd_inc: registers_output.rd_inc,
            rd_wa: registers_output.rd_wa,
        },
    };
    let instruction_expected = expected_instruction_read_raf_output(
        config,
        prefix.instruction_gamma,
        &stage2.batch.instruction_claim_reduction.opening_point,
        &instruction_opening_point.r_address,
        &instruction_opening_point.r_cycle,
        &output_openings.instruction_read_raf,
    )?;
    let ram_expected = expected_ram_ra_claim_reduction_output(
        config,
        prefix.ram_gamma,
        fixed_ram_raf_cycle_point,
        fixed_ram_read_write_cycle_point,
        fixed_ram_val_check_cycle_point,
        &ram_ra_claim_reduction_opening_point,
        output_openings.ram_ra_claim_reduction.ram_ra,
    )?;
    let registers_expected = expected_registers_val_evaluation_output(
        config,
        fixed_register_cycle_point,
        &registers_val_evaluation_opening_point,
        output_openings.registers_val_evaluation.rd_inc,
        output_openings.registers_val_evaluation.rd_wa,
    )?;
    let expected_final_claim = coefficients.instruction_read_raf * instruction_expected
        + coefficients.ram_ra_claim_reduction * ram_expected
        + coefficients.registers_val_evaluation * registers_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 batch final claim did not match output openings: instruction={}, ram={}, registers={}, instruction_internal={}, instruction_expected_internal={}",
            individual_claims[0] == instruction_expected,
            individual_claims[1] == ram_expected,
            individual_claims[2] == registers_expected,
            instruction_internal_final == individual_claims[0],
            instruction_internal_final == instruction_expected,
        )));
    }

    let output_claim_values = stage5_committed_output_claim_values(&output_openings);
    let built = builder.finish(&output_claim_values, transcript)?;
    let public = Stage5PublicOutput {
        challenges: sumcheck_point,
        batching_coefficients,
        instruction_gamma: prefix.instruction_gamma,
        ram_gamma: prefix.ram_gamma,
    };
    let verifier_output = Stage5ClearOutput {
        public: public.clone(),
        output_claims: output_openings,
        batch: VerifiedStage5Batch {
            batching_coefficients: public.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(public.challenges.clone()),
            sumcheck_final_claim: running_claim,
            expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: prefix.input_claims.instruction_read_raf,
                sumcheck_point: instruction_sumcheck_point,
                r_address: instruction_opening_point.r_address,
                r_cycle: instruction_opening_point.r_cycle,
                full_opening_point: instruction_opening_point.opening_point,
                lookup_table_flag_opening_point: instruction_lookup_table_flag_opening_point,
                instruction_ra_opening_points,
                instruction_raf_flag_opening_point,
                expected_output_claim: instruction_expected,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.ram_ra_claim_reduction,
                sumcheck_point: ram_ra_claim_reduction_sumcheck_point,
                opening_point: ram_ra_claim_reduction_opening_point,
                expected_output_claim: ram_expected,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.registers_val_evaluation,
                sumcheck_point: registers_val_evaluation_sumcheck_point,
                opening_point: registers_val_evaluation_opening_point,
                expected_output_claim: registers_expected,
            },
        },
    };
    Ok(Stage5CommittedBoundaryOutput {
        stage5_sumcheck_proof: built.proof,
        public,
        output_claim_values,
        verifier_output,
        committed_witness: built.witness,
    })
}

#[cfg(feature = "field-inline")]
#[expect(
    clippy::too_many_arguments,
    reason = "Stage 5 field-inline batch mirrors verifier instance order."
)]
fn prove_stage5_specialized_regular_batch_sumcheck<F, W, FI, B, T, C>(
    config: Stage5ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    validate_stage5_dependencies(config, stage2, stage4)?;

    let instruction_address_bits = config
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    let instruction_ra_count = config
        .instruction_read_raf_dimensions
        .num_virtual_ra_polys();
    let instruction_ra_chunk_bits = instruction_address_bits
        .checked_div(instruction_ra_count)
        .ok_or_else(|| {
            invalid_sumcheck_output(
                "Stage 5 instruction read-RAF config has no virtual RA polynomials",
            )
        })?;
    if instruction_ra_chunk_bits * instruction_ra_count != instruction_address_bits {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction address bit count {instruction_address_bits} is not divisible by virtual RA count {instruction_ra_count}",
        )));
    }

    let instruction_request = SumcheckInstructionReadRafStateRequest::new(
        "stage5.instruction_read_raf",
        stage5_instruction_read_raf_rows(config, witness)?,
        stage2
            .batch
            .instruction_claim_reduction
            .opening_point
            .clone(),
        prefix.instruction_gamma,
        prefix.input_claims.instruction_read_raf,
        config.log_t,
        instruction_address_bits,
        instruction_ra_chunk_bits,
        instruction_sumcheck_phases(config.log_t),
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);

    let (_fixed_ram_raf_address, fixed_ram_raf_cycle_point) = stage2
        .batch
        .ram_raf_evaluation
        .opening_point
        .split_at(config.log_k);
    let (fixed_ram_address_point, fixed_ram_read_write_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);
    let (_fixed_ram_val_check_address, fixed_ram_val_check_cycle_point) = stage4
        .batch
        .ram_val_check
        .opening_point
        .split_at(config.log_k);
    let (fixed_register_address_point, fixed_register_cycle_point) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let field_log_k = config.field_inline.field_register_log_k;
    let (fixed_field_register_address_point, fixed_field_register_cycle_point) = stage4
        .batch
        .field_registers_read_write
        .opening_point
        .split_at(field_log_k);

    let ram_request = SumcheckRamRaClaimReductionStateRequest::new(
        "stage5.ram_ra_claim_reduction",
        stage5_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_raf_cycle_point.to_vec(),
        fixed_ram_read_write_cycle_point.to_vec(),
        fixed_ram_val_check_cycle_point.to_vec(),
        prefix.ram_gamma,
        prefix.input_claims.ram_ra_claim_reduction,
        config.log_t,
        config.log_k,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let registers_request = SumcheckRegistersValEvaluationStateRequest::new(
        "stage5.registers_val_evaluation",
        stage5_register_rows(witness)?,
        fixed_register_address_point.to_vec(),
        fixed_register_cycle_point.to_vec(),
        prefix.input_claims.registers_val_evaluation,
        config.log_t,
        REGISTER_ADDRESS_BITS,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let field_registers_request = SumcheckFieldRegistersValEvaluationStateRequest::new(
        "stage5.field_registers_val_evaluation",
        stage5_field_register_rows(field_inline_witness)?,
        fixed_field_register_address_point.to_vec(),
        fixed_field_register_cycle_point.to_vec(),
        prefix.input_claims.field_registers_val_evaluation,
        config.log_t,
        field_log_k,
    )
    .with_optimization_ids(&["OPT-FLD-003", "OPT-REL-010"]);

    let mut instruction_state =
        backend.materialize_sumcheck_instruction_read_raf_state(&instruction_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_ra_claim_reduction_state(&ram_request)?;
    let mut registers_state =
        backend.materialize_sumcheck_registers_val_evaluation_state(&registers_request)?;
    let mut field_registers_state = backend
        .materialize_sumcheck_field_registers_val_evaluation_state(&field_registers_request)?;

    append_sumcheck_claim(transcript, &prefix.input_claims.instruction_read_raf);
    append_sumcheck_claim(transcript, &prefix.input_claims.ram_ra_claim_reduction);
    append_sumcheck_claim(transcript, &prefix.input_claims.registers_val_evaluation);
    append_sumcheck_claim(
        transcript,
        &prefix.input_claims.field_registers_val_evaluation,
    );
    let batching_coefficients = (0..STAGE5_BATCH_COEFFICIENTS)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [instruction_coefficient, ram_coefficient, registers_coefficient, field_registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 5 field-inline batch expected exactly four batching coefficients",
        ));
    };
    let coefficients = Stage5BatchCoefficients {
        instruction_read_raf: *instruction_coefficient,
        ram_ra_claim_reduction: *ram_coefficient,
        registers_val_evaluation: *registers_coefficient,
        field_registers_val_evaluation: *field_registers_coefficient,
    };

    let front_padding_rounds = instruction_address_bits;
    let mut individual_claims = [
        prefix.input_claims.instruction_read_raf,
        prefix
            .input_claims
            .ram_ra_claim_reduction
            .mul_pow_2(front_padding_rounds),
        prefix
            .input_claims
            .registers_val_evaluation
            .mul_pow_2(front_padding_rounds),
        prefix
            .input_claims
            .field_registers_val_evaluation
            .mul_pow_2(front_padding_rounds),
    ];
    let mut running_claim = coefficients.instruction_read_raf * individual_claims[0]
        + coefficients.ram_ra_claim_reduction * individual_claims[1]
        + coefficients.registers_val_evaluation * individual_claims[2]
        + coefficients.field_registers_val_evaluation * individual_claims[3];
    let max_rounds = front_padding_rounds + config.log_t;
    let mut round_polynomials = Vec::with_capacity(max_rounds);
    let mut sumcheck_point = Vec::with_capacity(max_rounds);
    let two_inv = F::from_u64(2).inv_or_zero();

    for round in 0..max_rounds {
        let instruction_poly = backend.evaluate_sumcheck_instruction_read_raf_round(
            &instruction_state,
            individual_claims[0],
        )?;
        let ram_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend
                .evaluate_sumcheck_ram_ra_claim_reduction_round(&ram_state, individual_claims[1])?
        };
        let registers_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[2] * two_inv])
        } else {
            backend.evaluate_sumcheck_registers_val_evaluation_round(
                &registers_state,
                individual_claims[2],
            )?
        };
        let field_registers_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[3] * two_inv])
        } else {
            backend.evaluate_sumcheck_field_registers_val_evaluation_round(
                &field_registers_state,
                individual_claims[3],
            )?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&instruction_poly * coefficients.instruction_read_raf);
        round_poly += &(&ram_poly * coefficients.ram_ra_claim_reduction);
        round_poly += &(&registers_poly * coefficients.registers_val_evaluation);
        round_poly += &(&field_registers_poly * coefficients.field_registers_val_evaluation);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 5 field-inline batch round {round} sumcheck invariant failed"
            )));
        }

        CompressedLabeledRoundPoly::sumcheck(&round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        individual_claims[0] = instruction_poly.evaluate(challenge);
        individual_claims[1] = if round < front_padding_rounds {
            individual_claims[1] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        individual_claims[2] = if round < front_padding_rounds {
            individual_claims[2] * two_inv
        } else {
            registers_poly.evaluate(challenge)
        };
        individual_claims[3] = if round < front_padding_rounds {
            individual_claims[3] * two_inv
        } else {
            field_registers_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_instruction_read_raf_state(&mut instruction_state, challenge)?;
        if round >= front_padding_rounds {
            backend.bind_sumcheck_ram_ra_claim_reduction_state(&mut ram_state, challenge)?;
            backend
                .bind_sumcheck_registers_val_evaluation_state(&mut registers_state, challenge)?;
            backend.bind_sumcheck_field_registers_val_evaluation_state(
                &mut field_registers_state,
                challenge,
            )?;
        }
        round_polynomials.push(round_poly.compress());
    }

    let instruction_sumcheck_point = sumcheck_point.clone();
    let instruction_opening_point = config
        .instruction_read_raf_dimensions
        .opening_point(&instruction_sumcheck_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let instruction_lookup_table_flag_opening_point = instruction_opening_point.r_cycle.clone();
    let instruction_raf_flag_opening_point = instruction_opening_point.r_cycle.clone();
    let instruction_ra_opening_points = instruction_opening_point
        .r_address
        .chunks(instruction_ra_chunk_bits)
        .map(|r_address_chunk| {
            [
                r_address_chunk,
                instruction_opening_point.r_cycle.as_slice(),
            ]
            .concat()
        })
        .collect::<Vec<_>>();

    let ram_ra_claim_reduction_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
    let ram_ra_claim_reduction_cycle_point = reversed_point(&ram_ra_claim_reduction_sumcheck_point);
    let ram_ra_claim_reduction_opening_point = [
        fixed_ram_address_point,
        ram_ra_claim_reduction_cycle_point.as_slice(),
    ]
    .concat();
    let registers_val_evaluation_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
    let registers_val_evaluation_cycle_point =
        reversed_point(&registers_val_evaluation_sumcheck_point);
    let registers_val_evaluation_opening_point = [
        fixed_register_address_point,
        registers_val_evaluation_cycle_point.as_slice(),
    ]
    .concat();
    let field_registers_val_evaluation_sumcheck_point =
        sumcheck_point[front_padding_rounds..].to_vec();
    let field_registers_val_evaluation_cycle_point =
        reversed_point(&field_registers_val_evaluation_sumcheck_point);
    let field_registers_val_evaluation_opening_point = [
        fixed_field_register_address_point,
        field_registers_val_evaluation_cycle_point.as_slice(),
    ]
    .concat();

    let instruction_output =
        backend.output_sumcheck_instruction_read_raf_state(&instruction_state)?;
    let instruction_internal_final = instruction_output.final_claim;
    let ram_output = backend.output_sumcheck_ram_ra_claim_reduction_state(&ram_state)?;
    let registers_output =
        backend.output_sumcheck_registers_val_evaluation_state(&registers_state)?;
    let field_registers_output =
        backend.output_sumcheck_field_registers_val_evaluation_state(&field_registers_state)?;
    let output_openings = Stage5RegularBatchOutputOpeningClaims {
        instruction_read_raf: InstructionReadRafOutputOpeningClaims {
            lookup_table_flags: instruction_output.lookup_table_flags,
            instruction_ra: instruction_output.instruction_ra,
            instruction_raf_flag: instruction_output.instruction_raf_flag,
        },
        ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
        },
        registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims {
            rd_inc: registers_output.rd_inc,
            rd_wa: registers_output.rd_wa,
        },
        field_inline: FieldInlineStage5Claims {
            field_registers_val_evaluation: FieldRegistersValEvaluationOutputOpeningClaims {
                field_rd_inc: field_registers_output.field_rd_inc,
                field_rd_wa: field_registers_output.field_rd_wa,
            },
        },
    };
    let instruction_expected = expected_instruction_read_raf_output(
        config,
        prefix.instruction_gamma,
        &stage2.batch.instruction_claim_reduction.opening_point,
        &instruction_opening_point.r_address,
        &instruction_opening_point.r_cycle,
        &output_openings.instruction_read_raf,
    )?;
    let ram_expected = expected_ram_ra_claim_reduction_output(
        config,
        prefix.ram_gamma,
        fixed_ram_raf_cycle_point,
        fixed_ram_read_write_cycle_point,
        fixed_ram_val_check_cycle_point,
        &ram_ra_claim_reduction_opening_point,
        output_openings.ram_ra_claim_reduction.ram_ra,
    )?;
    let registers_expected = expected_registers_val_evaluation_output(
        config,
        fixed_register_cycle_point,
        &registers_val_evaluation_opening_point,
        output_openings.registers_val_evaluation.rd_inc,
        output_openings.registers_val_evaluation.rd_wa,
    )?;
    let field_registers_expected = expected_field_registers_val_evaluation_output(
        config,
        fixed_field_register_cycle_point,
        &field_registers_val_evaluation_opening_point,
        output_openings
            .field_inline
            .field_registers_val_evaluation
            .field_rd_inc,
        output_openings
            .field_inline
            .field_registers_val_evaluation
            .field_rd_wa,
    )?;
    let expected_final_claim = coefficients.instruction_read_raf * instruction_expected
        + coefficients.ram_ra_claim_reduction * ram_expected
        + coefficients.registers_val_evaluation * registers_expected
        + coefficients.field_registers_val_evaluation * field_registers_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 field-inline batch final claim did not match output openings: instruction={}, ram={}, registers={}, field_registers={}, instruction_internal={}, instruction_expected_internal={}",
            individual_claims[0] == instruction_expected,
            individual_claims[1] == ram_expected,
            individual_claims[2] == registers_expected,
            individual_claims[3] == field_registers_expected,
            instruction_internal_final == individual_claims[0],
            instruction_internal_final == instruction_expected,
        )));
    }

    Ok(Stage5RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        output_openings,
        expected_outputs: Stage5RegularBatchExpectedOutputs {
            instruction_read_raf: instruction_expected,
            ram_ra_claim_reduction: ram_expected,
            registers_val_evaluation: registers_expected,
            field_registers_val_evaluation: field_registers_expected,
        },
        batching_coefficients,
        sumcheck_point,
        sumcheck_final_claim: running_claim,
        expected_final_claim,
        instruction_read_raf_sumcheck_point: instruction_sumcheck_point,
        instruction_read_raf_r_address: instruction_opening_point.r_address,
        instruction_read_raf_r_cycle: instruction_opening_point.r_cycle,
        instruction_read_raf_full_opening_point: instruction_opening_point.opening_point,
        instruction_lookup_table_flag_opening_point,
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_sumcheck_point,
        ram_ra_claim_reduction_opening_point,
        registers_val_evaluation_sumcheck_point,
        registers_val_evaluation_opening_point,
        field_registers_val_evaluation_sumcheck_point,
        field_registers_val_evaluation_opening_point,
    })
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
#[expect(clippy::too_many_arguments)]
fn prove_stage5_committed_specialized_regular_batch_sumcheck<F, W, FI, B, T, VC>(
    config: Stage5ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage5CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage5_dependencies(config, stage2, stage4)?;
    let instruction_address_bits = config
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    let instruction_ra_count = config
        .instruction_read_raf_dimensions
        .num_virtual_ra_polys();
    let instruction_ra_chunk_bits = instruction_address_bits
        .checked_div(instruction_ra_count)
        .ok_or_else(|| {
            invalid_sumcheck_output(
                "Stage 5 instruction read-RAF config has no virtual RA polynomials",
            )
        })?;
    if instruction_ra_chunk_bits * instruction_ra_count != instruction_address_bits {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction address bit count {instruction_address_bits} is not divisible by virtual RA count {instruction_ra_count}",
        )));
    }

    let instruction_request = SumcheckInstructionReadRafStateRequest::new(
        "stage5.instruction_read_raf",
        stage5_instruction_read_raf_rows(config, witness)?,
        stage2
            .batch
            .instruction_claim_reduction
            .opening_point
            .clone(),
        prefix.instruction_gamma,
        prefix.input_claims.instruction_read_raf,
        config.log_t,
        instruction_address_bits,
        instruction_ra_chunk_bits,
        instruction_sumcheck_phases(config.log_t),
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let (_fixed_ram_raf_address, fixed_ram_raf_cycle_point) = stage2
        .batch
        .ram_raf_evaluation
        .opening_point
        .split_at(config.log_k);
    let (fixed_ram_address_point, fixed_ram_read_write_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);
    let (_fixed_ram_val_check_address, fixed_ram_val_check_cycle_point) = stage4
        .batch
        .ram_val_check
        .opening_point
        .split_at(config.log_k);
    let (fixed_register_address_point, fixed_register_cycle_point) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let field_log_k = config.field_inline.field_register_log_k;
    let (fixed_field_register_address_point, fixed_field_register_cycle_point) = stage4
        .batch
        .field_registers_read_write
        .opening_point
        .split_at(field_log_k);

    let ram_request = SumcheckRamRaClaimReductionStateRequest::new(
        "stage5.ram_ra_claim_reduction",
        stage5_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_raf_cycle_point.to_vec(),
        fixed_ram_read_write_cycle_point.to_vec(),
        fixed_ram_val_check_cycle_point.to_vec(),
        prefix.ram_gamma,
        prefix.input_claims.ram_ra_claim_reduction,
        config.log_t,
        config.log_k,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let registers_request = SumcheckRegistersValEvaluationStateRequest::new(
        "stage5.registers_val_evaluation",
        stage5_register_rows(witness)?,
        fixed_register_address_point.to_vec(),
        fixed_register_cycle_point.to_vec(),
        prefix.input_claims.registers_val_evaluation,
        config.log_t,
        REGISTER_ADDRESS_BITS,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let field_registers_request = SumcheckFieldRegistersValEvaluationStateRequest::new(
        "stage5.field_registers_val_evaluation",
        stage5_field_register_rows(field_inline_witness)?,
        fixed_field_register_address_point.to_vec(),
        fixed_field_register_cycle_point.to_vec(),
        prefix.input_claims.field_registers_val_evaluation,
        config.log_t,
        field_log_k,
    )
    .with_optimization_ids(&["OPT-FLD-003", "OPT-REL-010"]);

    let mut instruction_state =
        backend.materialize_sumcheck_instruction_read_raf_state(&instruction_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_ra_claim_reduction_state(&ram_request)?;
    let mut registers_state =
        backend.materialize_sumcheck_registers_val_evaluation_state(&registers_request)?;
    let mut field_registers_state = backend
        .materialize_sumcheck_field_registers_val_evaluation_state(&field_registers_request)?;

    let batching_coefficients = (0..STAGE5_BATCH_COEFFICIENTS)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [instruction_coefficient, ram_coefficient, registers_coefficient, field_registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 5 field-inline batch expected exactly four batching coefficients",
        ));
    };
    let coefficients = Stage5BatchCoefficients {
        instruction_read_raf: *instruction_coefficient,
        ram_ra_claim_reduction: *ram_coefficient,
        registers_val_evaluation: *registers_coefficient,
        field_registers_val_evaluation: *field_registers_coefficient,
    };
    let front_padding_rounds = instruction_address_bits;
    let mut individual_claims = [
        prefix.input_claims.instruction_read_raf,
        prefix
            .input_claims
            .ram_ra_claim_reduction
            .mul_pow_2(front_padding_rounds),
        prefix
            .input_claims
            .registers_val_evaluation
            .mul_pow_2(front_padding_rounds),
        prefix
            .input_claims
            .field_registers_val_evaluation
            .mul_pow_2(front_padding_rounds),
    ];
    let mut running_claim = coefficients.instruction_read_raf * individual_claims[0]
        + coefficients.ram_ra_claim_reduction * individual_claims[1]
        + coefficients.registers_val_evaluation * individual_claims[2]
        + coefficients.field_registers_val_evaluation * individual_claims[3];
    let max_rounds = front_padding_rounds + config.log_t;
    let mut sumcheck_point = Vec::with_capacity(max_rounds);
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, max_rounds)?;
    let two_inv = F::from_u64(2).inv_or_zero();

    for round in 0..max_rounds {
        let instruction_poly = backend.evaluate_sumcheck_instruction_read_raf_round(
            &instruction_state,
            individual_claims[0],
        )?;
        let ram_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend
                .evaluate_sumcheck_ram_ra_claim_reduction_round(&ram_state, individual_claims[1])?
        };
        let registers_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[2] * two_inv])
        } else {
            backend.evaluate_sumcheck_registers_val_evaluation_round(
                &registers_state,
                individual_claims[2],
            )?
        };
        let field_registers_poly = if round < front_padding_rounds {
            UnivariatePoly::new(vec![individual_claims[3] * two_inv])
        } else {
            backend.evaluate_sumcheck_field_registers_val_evaluation_round(
                &field_registers_state,
                individual_claims[3],
            )?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&instruction_poly * coefficients.instruction_read_raf);
        round_poly += &(&ram_poly * coefficients.ram_ra_claim_reduction);
        round_poly += &(&registers_poly * coefficients.registers_val_evaluation);
        round_poly += &(&field_registers_poly * coefficients.field_registers_val_evaluation);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 5 committed field-inline batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = builder.commit_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        individual_claims[0] = instruction_poly.evaluate(challenge);
        individual_claims[1] = if round < front_padding_rounds {
            individual_claims[1] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        individual_claims[2] = if round < front_padding_rounds {
            individual_claims[2] * two_inv
        } else {
            registers_poly.evaluate(challenge)
        };
        individual_claims[3] = if round < front_padding_rounds {
            individual_claims[3] * two_inv
        } else {
            field_registers_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_instruction_read_raf_state(&mut instruction_state, challenge)?;
        if round >= front_padding_rounds {
            backend.bind_sumcheck_ram_ra_claim_reduction_state(&mut ram_state, challenge)?;
            backend
                .bind_sumcheck_registers_val_evaluation_state(&mut registers_state, challenge)?;
            backend.bind_sumcheck_field_registers_val_evaluation_state(
                &mut field_registers_state,
                challenge,
            )?;
        }
    }

    let instruction_sumcheck_point = sumcheck_point.clone();
    let instruction_opening_point = config
        .instruction_read_raf_dimensions
        .opening_point(&instruction_sumcheck_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let instruction_lookup_table_flag_opening_point = instruction_opening_point.r_cycle.clone();
    let instruction_raf_flag_opening_point = instruction_opening_point.r_cycle.clone();
    let instruction_ra_opening_points = instruction_opening_point
        .r_address
        .chunks(instruction_ra_chunk_bits)
        .map(|r_address_chunk| {
            [
                r_address_chunk,
                instruction_opening_point.r_cycle.as_slice(),
            ]
            .concat()
        })
        .collect::<Vec<_>>();
    let ram_ra_claim_reduction_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
    let ram_ra_claim_reduction_cycle_point = reversed_point(&ram_ra_claim_reduction_sumcheck_point);
    let ram_ra_claim_reduction_opening_point = [
        fixed_ram_address_point,
        ram_ra_claim_reduction_cycle_point.as_slice(),
    ]
    .concat();
    let registers_val_evaluation_sumcheck_point = sumcheck_point[front_padding_rounds..].to_vec();
    let registers_val_evaluation_cycle_point =
        reversed_point(&registers_val_evaluation_sumcheck_point);
    let registers_val_evaluation_opening_point = [
        fixed_register_address_point,
        registers_val_evaluation_cycle_point.as_slice(),
    ]
    .concat();
    let field_registers_val_evaluation_sumcheck_point =
        sumcheck_point[front_padding_rounds..].to_vec();
    let field_registers_val_evaluation_cycle_point =
        reversed_point(&field_registers_val_evaluation_sumcheck_point);
    let field_registers_val_evaluation_opening_point = [
        fixed_field_register_address_point,
        field_registers_val_evaluation_cycle_point.as_slice(),
    ]
    .concat();

    let instruction_output =
        backend.output_sumcheck_instruction_read_raf_state(&instruction_state)?;
    let instruction_internal_final = instruction_output.final_claim;
    let ram_output = backend.output_sumcheck_ram_ra_claim_reduction_state(&ram_state)?;
    let registers_output =
        backend.output_sumcheck_registers_val_evaluation_state(&registers_state)?;
    let field_registers_output =
        backend.output_sumcheck_field_registers_val_evaluation_state(&field_registers_state)?;
    let output_openings = Stage5RegularBatchOutputOpeningClaims {
        instruction_read_raf: InstructionReadRafOutputOpeningClaims {
            lookup_table_flags: instruction_output.lookup_table_flags,
            instruction_ra: instruction_output.instruction_ra,
            instruction_raf_flag: instruction_output.instruction_raf_flag,
        },
        ram_ra_claim_reduction: RamRaClaimReductionOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
        },
        registers_val_evaluation: RegistersValEvaluationOutputOpeningClaims {
            rd_inc: registers_output.rd_inc,
            rd_wa: registers_output.rd_wa,
        },
        field_inline: FieldInlineStage5Claims {
            field_registers_val_evaluation: FieldRegistersValEvaluationOutputOpeningClaims {
                field_rd_inc: field_registers_output.field_rd_inc,
                field_rd_wa: field_registers_output.field_rd_wa,
            },
        },
    };
    let instruction_expected = expected_instruction_read_raf_output(
        config,
        prefix.instruction_gamma,
        &stage2.batch.instruction_claim_reduction.opening_point,
        &instruction_opening_point.r_address,
        &instruction_opening_point.r_cycle,
        &output_openings.instruction_read_raf,
    )?;
    let ram_expected = expected_ram_ra_claim_reduction_output(
        config,
        prefix.ram_gamma,
        fixed_ram_raf_cycle_point,
        fixed_ram_read_write_cycle_point,
        fixed_ram_val_check_cycle_point,
        &ram_ra_claim_reduction_opening_point,
        output_openings.ram_ra_claim_reduction.ram_ra,
    )?;
    let registers_expected = expected_registers_val_evaluation_output(
        config,
        fixed_register_cycle_point,
        &registers_val_evaluation_opening_point,
        output_openings.registers_val_evaluation.rd_inc,
        output_openings.registers_val_evaluation.rd_wa,
    )?;
    let field_registers_expected = expected_field_registers_val_evaluation_output(
        config,
        fixed_field_register_cycle_point,
        &field_registers_val_evaluation_opening_point,
        output_openings
            .field_inline
            .field_registers_val_evaluation
            .field_rd_inc,
        output_openings
            .field_inline
            .field_registers_val_evaluation
            .field_rd_wa,
    )?;
    let expected_final_claim = coefficients.instruction_read_raf * instruction_expected
        + coefficients.ram_ra_claim_reduction * ram_expected
        + coefficients.registers_val_evaluation * registers_expected
        + coefficients.field_registers_val_evaluation * field_registers_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 field-inline batch final claim did not match output openings: instruction={}, ram={}, registers={}, field_registers={}, instruction_internal={}, instruction_expected_internal={}",
            individual_claims[0] == instruction_expected,
            individual_claims[1] == ram_expected,
            individual_claims[2] == registers_expected,
            individual_claims[3] == field_registers_expected,
            instruction_internal_final == individual_claims[0],
            instruction_internal_final == instruction_expected,
        )));
    }

    let output_claim_values = stage5_committed_output_claim_values(&output_openings);
    let built = builder.finish(&output_claim_values, transcript)?;
    let public = Stage5PublicOutput {
        challenges: sumcheck_point,
        batching_coefficients,
        instruction_gamma: prefix.instruction_gamma,
        ram_gamma: prefix.ram_gamma,
    };
    let verifier_output = Stage5ClearOutput {
        public: public.clone(),
        output_claims: output_openings,
        batch: VerifiedStage5Batch {
            batching_coefficients: public.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(public.challenges.clone()),
            sumcheck_final_claim: running_claim,
            expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: prefix.input_claims.instruction_read_raf,
                sumcheck_point: instruction_sumcheck_point,
                r_address: instruction_opening_point.r_address,
                r_cycle: instruction_opening_point.r_cycle,
                full_opening_point: instruction_opening_point.opening_point,
                lookup_table_flag_opening_point: instruction_lookup_table_flag_opening_point,
                instruction_ra_opening_points,
                instruction_raf_flag_opening_point,
                expected_output_claim: instruction_expected,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.ram_ra_claim_reduction,
                sumcheck_point: ram_ra_claim_reduction_sumcheck_point,
                opening_point: ram_ra_claim_reduction_opening_point,
                expected_output_claim: ram_expected,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.registers_val_evaluation,
                sumcheck_point: registers_val_evaluation_sumcheck_point,
                opening_point: registers_val_evaluation_opening_point,
                expected_output_claim: registers_expected,
            },
            field_registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.field_registers_val_evaluation,
                sumcheck_point: field_registers_val_evaluation_sumcheck_point,
                opening_point: field_registers_val_evaluation_opening_point,
                expected_output_claim: field_registers_expected,
            },
        },
    };
    Ok(Stage5CommittedBoundaryOutput {
        stage5_sumcheck_proof: built.proof,
        public,
        output_claim_values,
        verifier_output,
        committed_witness: built.witness,
    })
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchFrontierProof<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub output_claim: F,
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
pub fn prove_stage5_regular_batch_sumcheck_for_frontier<F, W, B, T, C>(
    input: &Stage5ProverInput<'_, F, W>,
    backend: &mut B,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchFrontierProof<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let proof_output = prove_stage5_specialized_regular_batch_sumcheck::<F, W, B, T, C>(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage4,
        prefix,
        transcript,
    )?;
    Ok(Stage5RegularBatchFrontierProof {
        proof: proof_output.proof,
        challenges: proof_output.sumcheck_point,
        batching_coefficients: proof_output.batching_coefficients,
        output_claim: proof_output.sumcheck_final_claim,
    })
}

fn expected_instruction_read_raf_output<F: Field>(
    config: Stage5ProverConfig,
    gamma: F,
    fixed_cycle_point: &[F],
    r_address: &[F],
    r_cycle: &[F],
    openings: &InstructionReadRafOutputOpeningClaims<F>,
) -> Result<F, ProverError> {
    let instruction_address_bits = config
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    if r_address.len() != instruction_address_bits {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction output address point has {} challenges, expected {instruction_address_bits}",
            r_address.len()
        )));
    }
    if r_cycle.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction output cycle point has {} challenges, expected {}",
            r_cycle.len(),
            config.log_t
        )));
    }
    if openings.lookup_table_flags.len() != LookupTableKind::<64>::COUNT {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction table flag claim count is {}, expected {}",
            openings.lookup_table_flags.len(),
            LookupTableKind::<64>::COUNT
        )));
    }
    if openings.instruction_ra.len()
        != config
            .instruction_read_raf_dimensions
            .num_virtual_ra_polys()
    {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction virtual RA claim count is {}, expected {}",
            openings.instruction_ra.len(),
            config
                .instruction_read_raf_dimensions
                .num_virtual_ra_polys()
        )));
    }

    let eq_cycle = try_eq_mle(fixed_cycle_point, r_cycle)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let table_value = LookupTableKind::<64>::iter()
        .zip(openings.lookup_table_flags.iter())
        .map(|(table, &claim)| table.evaluate_mle::<F, F>(r_address) * claim)
        .sum::<F>();
    let ra_product = openings.instruction_ra.iter().copied().product::<F>();
    let left_operand =
        OperandPolynomial::new(instruction_address_bits, OperandSide::Left).evaluate(r_address);
    let right_operand =
        OperandPolynomial::new(instruction_address_bits, OperandSide::Right).evaluate(r_address);
    let identity = IdentityPolynomial::new(instruction_address_bits).evaluate(r_address);
    let gamma2 = gamma * gamma;
    let constant = gamma * left_operand + gamma2 * right_operand;
    let raf_coeff = gamma2 * identity - constant;

    Ok(
        eq_cycle
            * ra_product
            * (table_value + constant + raf_coeff * openings.instruction_raf_flag),
    )
}

fn expected_ram_ra_claim_reduction_output<F: Field>(
    config: Stage5ProverConfig,
    ram_gamma: F,
    fixed_raf_cycle_point: &[F],
    fixed_read_write_cycle_point: &[F],
    fixed_val_check_cycle_point: &[F],
    opening_point: &[F],
    ram_ra: F,
) -> Result<F, ProverError> {
    if opening_point.len() != config.log_k + config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 RAM RA claim-reduction opening point has {} variables, expected {}",
            opening_point.len(),
            config.log_k + config.log_t
        )));
    }
    let (_, r_cycle) = opening_point.split_at(config.log_k);
    let ram_gamma2 = ram_gamma * ram_gamma;
    let eq_cycle_raf = try_eq_mle(fixed_raf_cycle_point, r_cycle)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let eq_cycle_read_write = try_eq_mle(fixed_read_write_cycle_point, r_cycle)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let eq_cycle_val_check = try_eq_mle(fixed_val_check_cycle_point, r_cycle)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    Ok((eq_cycle_raf + ram_gamma * eq_cycle_read_write + ram_gamma2 * eq_cycle_val_check) * ram_ra)
}

fn expected_registers_val_evaluation_output<F: Field>(
    config: Stage5ProverConfig,
    fixed_cycle_point: &[F],
    opening_point: &[F],
    rd_inc: F,
    rd_wa: F,
) -> Result<F, ProverError> {
    if opening_point.len() != REGISTER_ADDRESS_BITS + config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 register value-evaluation opening point has {} variables, expected {}",
            opening_point.len(),
            REGISTER_ADDRESS_BITS + config.log_t
        )));
    }
    let (_, r_cycle) = opening_point.split_at(REGISTER_ADDRESS_BITS);
    Ok(LtPolynomial::evaluate(r_cycle, fixed_cycle_point) * rd_inc * rd_wa)
}

#[cfg(feature = "field-inline")]
fn expected_field_registers_val_evaluation_output<F: Field>(
    config: Stage5ProverConfig,
    fixed_cycle_point: &[F],
    opening_point: &[F],
    field_rd_inc: F,
    field_rd_wa: F,
) -> Result<F, ProverError> {
    let field_log_k = config.field_inline.field_register_log_k;
    if opening_point.len() != field_log_k + config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 field-register value-evaluation opening point has {} variables, expected {}",
            opening_point.len(),
            field_log_k + config.log_t
        )));
    }
    let (_, r_cycle) = opening_point.split_at(field_log_k);
    Ok(LtPolynomial::evaluate(r_cycle, fixed_cycle_point) * field_rd_inc * field_rd_wa)
}

fn stage5_instruction_read_raf_rows<W>(
    config: Stage5ProverConfig,
    witness: &W,
) -> Result<Vec<SumcheckInstructionReadRafRow>, ProverError>
where
    W: JoltVmStage5InstructionReadRafRows,
{
    let address_bits = config
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    if address_bits > u128::BITS as usize {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 instruction address width {address_bits} exceeds u128 width"
        )));
    }
    Ok(witness
        .stage5_instruction_read_raf_rows(config.log_t)?
        .into_iter()
        .map(|row| {
            SumcheckInstructionReadRafRow::new(
                row.lookup_index,
                row.table_index,
                row.interleaved_operands,
            )
        })
        .collect())
}

const fn instruction_sumcheck_phases(log_t: usize) -> usize {
    if log_t < INSTRUCTION_PHASES_THRESHOLD_LOG_T {
        16
    } else {
        8
    }
}

fn stage5_register_rows<W>(witness: &W) -> Result<Vec<SumcheckRegistersReadWriteRow>, ProverError>
where
    W: JoltVmRegisterReadWriteRows,
{
    Ok(witness
        .register_read_write_rows()?
        .into_iter()
        .map(stage5_register_row)
        .collect())
}

fn stage5_register_row(row: JoltVmRegisterReadWriteRow) -> SumcheckRegistersReadWriteRow {
    SumcheckRegistersReadWriteRow {
        rs1: row.rs1.map(|read| SumcheckRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rs2: row.rs2.map(|read| SumcheckRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rd: row.rd.map(|write| SumcheckRegisterWrite {
            register: write.register,
            pre_value: write.pre_value,
            post_value: write.post_value,
        }),
        rd_increment: row.rd_increment,
    }
}

#[cfg(feature = "field-inline")]
fn stage5_field_register_rows<F, FI>(
    witness: &FI,
) -> Result<Vec<SumcheckFieldRegistersReadWriteRow<F>>, ProverError>
where
    F: Field,
    FI: FieldInlineRegisterReadWriteRows<F>,
{
    Ok(witness
        .field_inline_register_read_write_rows()?
        .into_iter()
        .map(stage5_field_register_row)
        .collect())
}

#[cfg(feature = "field-inline")]
fn stage5_field_register_row<F: Field>(
    row: FieldInlineRegisterReadWriteRow<F>,
) -> SumcheckFieldRegistersReadWriteRow<F> {
    SumcheckFieldRegistersReadWriteRow {
        rs1: row.rs1.map(|read| SumcheckFieldRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rs2: row.rs2.map(|read| SumcheckFieldRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rd: row.rd.map(|write| SumcheckFieldRegisterWrite {
            register: write.register,
            pre_value: write.pre_value,
            post_value: write.post_value,
        }),
        rd_increment: row.rd_increment,
    }
}

fn stage5_ram_rows<W>(witness: &W) -> Result<Vec<SumcheckRamReadWriteRow>, ProverError>
where
    W: JoltVmStage2Rows,
{
    Ok(witness
        .stage2_rows()?
        .into_iter()
        .map(stage5_ram_row)
        .collect())
}

fn stage5_ram_row(row: JoltVmStage2TraceRow) -> SumcheckRamReadWriteRow {
    SumcheckRamReadWriteRow {
        remapped_ram_address: row.remapped_ram_address,
        ram_read_value: row.ram_read_value,
        ram_write_value: row.ram_write_value,
        ram_increment: row.ram_increment,
    }
}

pub fn append_stage5_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage5RegularBatchOutputOpeningClaims<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.instruction_read_raf.lookup_table_flags {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.instruction_read_raf.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_read_raf.instruction_raf_flag,
    );
    transcript.append_labeled(b"opening_claim", &claims.ram_ra_claim_reduction.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_val_evaluation.rd_inc);
    transcript.append_labeled(b"opening_claim", &claims.registers_val_evaluation.rd_wa);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_val_evaluation;
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_inc);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_wa);
    }
}

#[cfg(feature = "zk")]
fn stage5_committed_output_claim_values<F: Field>(
    claims: &Stage5RegularBatchOutputOpeningClaims<F>,
) -> Vec<F> {
    let mut values = Vec::with_capacity(stage5_committed_output_claim_count(claims));
    values.extend(
        claims
            .instruction_read_raf
            .lookup_table_flags
            .iter()
            .copied(),
    );
    values.extend(claims.instruction_read_raf.instruction_ra.iter().copied());
    values.push(claims.instruction_read_raf.instruction_raf_flag);
    values.push(claims.ram_ra_claim_reduction.ram_ra);
    values.push(claims.registers_val_evaluation.rd_inc);
    values.push(claims.registers_val_evaluation.rd_wa);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_val_evaluation;
        values.push(field_claims.field_rd_inc);
        values.push(field_claims.field_rd_wa);
    }
    values
}

#[cfg(feature = "zk")]
fn stage5_committed_output_claim_count<F: Field>(
    claims: &Stage5RegularBatchOutputOpeningClaims<F>,
) -> usize {
    claims.instruction_read_raf.lookup_table_flags.len()
        + claims.instruction_read_raf.instruction_ra.len()
        + 1
        + 1
        + 2
        + {
            #[cfg(feature = "field-inline")]
            {
                2
            }
            #[cfg(not(feature = "field-inline"))]
            {
                0
            }
        }
}

fn validate_stage5_dependencies<F: Field>(
    config: Stage5ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
) -> Result<(), ProverError> {
    if stage2.batch.product_remainder.opening_point
        != stage2.batch.instruction_claim_reduction.opening_point
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 instruction read-RAF dependencies use different cycle opening points"
                .to_owned(),
        });
    }
    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output;
    let reduced_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);
    if reduced_lookup_output != product_lookup_output {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 instruction read-RAF lookup-output dependencies disagree".to_owned(),
        });
    }

    let expected_trace_vars = config.log_t;
    for (label, point) in [
        (
            "product remainder",
            &stage2.batch.product_remainder.opening_point,
        ),
        (
            "instruction claim reduction",
            &stage2.batch.instruction_claim_reduction.opening_point,
        ),
    ] {
        if point.len() != expected_trace_vars {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 5 {label} opening point has {} variables, expected {expected_trace_vars}",
                    point.len()
                ),
            });
        }
    }

    let expected_ram_vars = config.log_k + config.log_t;
    for (label, point) in [
        (
            "RAM RAF evaluation",
            &stage2.batch.ram_raf_evaluation.opening_point,
        ),
        ("RAM read-write", &stage2.batch.ram_read_write.opening_point),
        ("RAM value-check", &stage4.batch.ram_val_check.opening_point),
    ] {
        if point.len() != expected_ram_vars {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 5 {label} opening point has {} variables, expected {expected_ram_vars}",
                    point.len()
                ),
            });
        }
    }
    let (ram_raf_address, _) = stage2
        .batch
        .ram_raf_evaluation
        .opening_point
        .split_at(config.log_k);
    let (ram_read_write_address, _) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);
    let (ram_val_check_address, _) = stage4
        .batch
        .ram_val_check
        .opening_point
        .split_at(config.log_k);
    if ram_raf_address != ram_read_write_address || ram_val_check_address != ram_read_write_address
    {
        return Err(ProverError::InvalidStageRequest {
            reason:
                "Stage 5 RAM RA claim-reduction dependencies use different address opening points"
                    .to_owned(),
        });
    }

    let expected_register_vars = REGISTER_ADDRESS_BITS + config.log_t;
    if stage4.batch.registers_read_write.opening_point.len() != expected_register_vars {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 register read-write opening point has {} variables, expected {expected_register_vars}",
                stage4.batch.registers_read_write.opening_point.len()
            ),
        });
    }
    #[cfg(feature = "field-inline")]
    {
        let expected_field_register_vars = config.field_inline.field_register_log_k + config.log_t;
        if stage4.batch.field_registers_read_write.opening_point.len()
            != expected_field_register_vars
        {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 5 field-register read-write opening point has {} variables, expected {expected_field_register_vars}",
                    stage4.batch.field_registers_read_write.opening_point.len()
                ),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Stage5BatchCoefficients<F: Field> {
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    field_registers_val_evaluation: F,
}

fn reversed_point<F: Field>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

#[cfg(feature = "zk")]
fn validate_stage5_committed_checked(
    config: Stage5ProverConfig,
    checked: &jolt_verifier::CheckedInputs,
) -> Result<(), ProverError> {
    if !checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 committed-boundary prover received transparent checked inputs"
                .to_owned(),
        });
    }
    if checked.trace_length != (1usize << config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 checked trace length {} does not match log_t {}",
                checked.trace_length, config.log_t
            ),
        });
    }
    if checked.ram_K != (1usize << config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 5 checked RAM K {} does not match log_k {}",
                checked.ram_K, config.log_k
            ),
        });
    }
    Ok(())
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
