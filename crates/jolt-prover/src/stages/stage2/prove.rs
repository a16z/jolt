use jolt_backends::{
    BackendRelationId, BackendValueSlot, RamReadWriteSumcheckBackend, SumcheckBackend,
    SumcheckRamOutputCheckStateRequest, SumcheckRamRafStateRequest, SumcheckRamReadWriteRow,
    SumcheckRamReadWriteStateRequest, SumcheckRegularBatchInstance,
    SumcheckRegularBatchLinearFactor, SumcheckRegularBatchLinearTerm, SumcheckRegularBatchState,
};
#[cfg(feature = "field-inline")]
use jolt_backends::{SumcheckEvaluationRequest, SumcheckViewEvaluationRequest};
#[cfg(not(feature = "field-inline"))]
use jolt_backends::{
    SumcheckLinearProductOutput, SumcheckProductUniskipRequest, SumcheckProductUniskipRow,
    SumcheckRowProductQuery,
};
#[cfg(feature = "field-inline")]
use jolt_backends::{
    SumcheckMaterializationOutput, SumcheckMaterializationRequest,
    SumcheckViewMaterializationRequest,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineVirtualPolynomial;
use jolt_claims::protocols::jolt::formulas::{
    dimensions::ReadWriteOpeningPoint, ram::RamRafEvaluationDimensions,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
#[cfg(feature = "field-inline")]
use jolt_poly::TensorEqTable;
use jolt_poly::{
    eq_index_msb,
    lagrange::{
        centered_domain_start, centered_lagrange_evals, centered_lagrange_kernel,
        interpolate_to_coeffs, poly_mul,
    },
    range_mask_mle_msb, sparse_segments_mle_msb,
    thread::unsafe_allocate_zero_vec,
    try_eq_mle, EqPolynomial, IdentityPolynomial, MultilinearEvaluation, Point, Polynomial,
    UnivariatePoly,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, ClearSumcheckProof, CompressedLabeledRoundPoly,
    CompressedSumcheckProof, LabeledRoundPoly, RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::inputs::{
    InstructionClaimReductionOutputOpeningClaims, ProductRemainderOutputOpeningClaims,
    RamReadWriteOutputOpeningClaims, Stage2BatchOutputOpeningClaims, Stage2Claims,
};
use jolt_verifier::stages::stage2::outputs::{
    Stage2ClearOutput, Stage2PublicOutput, VerifiedProductUniSkip, VerifiedStage2Batch,
    VerifiedStage2Sumcheck,
};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::protocols::jolt_vm::JoltVmProductUniskipRows;
use jolt_witness::{
    protocols::jolt_vm::{
        JoltVmNamespace, JoltVmStage2Rows, JoltVmStage2TraceRow, JOLT_VM_NAMESPACE,
    },
    WitnessProvider,
};
#[cfg(feature = "field-inline")]
use jolt_witness::{OracleRef, WitnessNamespace};
use rayon::prelude::*;

#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::ProverError;

#[cfg(feature = "zk")]
use super::output::Stage2CommittedBoundaryOutput;
#[cfg(feature = "field-inline")]
use super::output::Stage2FieldInlineProductOutputOpeningClaims;
#[cfg(feature = "field-inline")]
use super::request::primary_view_requirement;
use super::request::{
    build_stage2_instruction_claim_opening_evaluation_request,
    build_stage2_product_remainder_opening_evaluation_request,
    build_stage2_ram_read_write_opening_evaluation_request,
    build_stage2_ram_terminal_opening_evaluation_request,
};
#[cfg(not(feature = "field-inline"))]
use super::request::{SPARTAN_PRODUCT_UNISKIP_RELATION, STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS};
use super::{
    input::{
        Stage2BatchProverConfig, Stage2ProductUniSkipInput, Stage2ProverConfig, Stage2ProverInput,
    },
    output::{
        instruction_claim_openings_from_evaluations, product_remainder_openings_from_evaluations,
        ram_read_write_openings_from_evaluations, ram_terminal_openings_from_evaluations,
        Stage2ProductUniSkipOutput, Stage2ProverOutput, Stage2RamTerminalOutputOpeningClaims,
        Stage2RegularBatchInputClaims, Stage2RegularBatchPrefixOutput,
    },
};

const PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT: usize = SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE - 1;
const STAGE2_RAM_READ_WRITE_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "ram.read_write_checking");
const STAGE2_RAM_RAF_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "ram.raf_evaluation");
const STAGE2_RAM_OUTPUT_CHECK_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "ram.output_check");
const STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];

#[cfg(feature = "frontier-harness")]
fn timed_stage2<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage2<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(feature = "frontier-harness")]
fn timed_stage2_accumulate<T>(accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    *accumulator += start.elapsed().as_secs_f64() * 1000.0;
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage2_accumulate<T>(_accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "frontier-harness")]
fn record_stage2_accumulated(label: &'static str, time_ms: f64) {
    crate::timing::record_stage_timing(label, time_ms);
}

#[cfg(not(feature = "frontier-harness"))]
fn record_stage2_accumulated(_label: &'static str, _time_ms: f64) {}

#[cfg(not(feature = "field-inline"))]
pub fn prove<F, W, B, T, C>(
    input: Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }
    if input.checked.ram_K != (1usize << input.config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked RAM K {} does not match log_k {}",
                input.checked.ram_K, input.config.log_k
            ),
        });
    }

    let (stage2_rows, initial_ram_state, final_ram_state) = timed_stage2("stage2.rows", || {
        Ok::<_, ProverError>((
            input.witness.stage2_rows()?,
            input.witness.initial_ram_state_words()?,
            input.witness.final_ram_state_words()?,
        ))
    })?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = timed_stage2("stage2.product_uniskip", || {
        prove_stage2_product_uniskip_from_stage2_rows::<F, B, T, C>(
            Stage2ProverConfig::new(input.config.log_t),
            &product_input,
            &stage2_rows,
            backend,
            transcript,
        )
    })?;

    let batch_prefix = timed_stage2("stage2.prefix", || {
        derive_stage2_regular_batch_prefix(input.config, input.stage1, &product_uniskip, transcript)
    })?;
    let (ram_read_write, ram_raf, ram_output_check, instances) =
        timed_stage2("stage2.build_requests", || {
            let backend_rows = stage2_backend_rows(&stage2_rows);
            let ram_read_write = build_ram_read_write_state_request(
                input.config,
                backend_rows.clone(),
                &initial_ram_state,
                &product_uniskip,
                &batch_prefix,
            )?;
            let ram_raf = build_ram_raf_state_request(
                input.config,
                input.checked,
                input.stage1,
                backend_rows,
                &product_uniskip,
            )?;
            let ram_output_check = build_ram_output_check_state_request(
                input.config,
                input.checked,
                &final_ram_state,
                &batch_prefix,
            )?;
            let instances = build_regular_batch_instances(
                input.config,
                &stage2_rows,
                &product_uniskip,
                &batch_prefix,
            )?;
            Ok::<_, ProverError>((ram_read_write, ram_raf, ram_output_check, instances))
        })?;
    let batch = timed_stage2("stage2.regular_batch", || {
        prove_regular_batch_sumcheck::<F, T, C, B>(
            ram_read_write,
            ram_raf,
            ram_output_check,
            instances,
            backend,
            transcript,
        )
    })?;
    let opening_points =
        Stage2OpeningPoints::from_batch(input.config, &batch.challenges, &product_uniskip.tau_low)?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = timed_stage2("stage2.openings.tail", || {
        evaluate_stage2_tail_openings_from_rows(
            input.config,
            &stage2_rows,
            &opening_points.product_opening,
            &opening_points.instruction_opening,
        )
    })?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims = Stage2BatchOutputOpeningClaims {
        ram_read_write,
        product_remainder: tail_openings.product_remainder,
        instruction_claim_reduction: tail_openings.instruction_claim_reduction,
        ram_raf_evaluation: terminal.ram_raf_evaluation,
        ram_output_check: terminal.ram_output_check,
    };
    let expected = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip,
        &batch_prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected.final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            &expected,
        ));
    }

    append_stage2_opening_claims(transcript, &output_claims);

    let claims = Stage2Claims {
        product_uniskip_output_claim: product_uniskip.output_claim,
        batch_outputs: output_claims.clone(),
    };
    let public = Stage2PublicOutput {
        challenges: batch.challenges.clone(),
        batching_coefficients: batch.batching_coefficients.clone(),
        product_uniskip_challenge: product_uniskip.challenge,
        product_tau_low: product_uniskip.tau_low.clone(),
        product_tau_high: product_uniskip.tau_high,
        ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
        instruction_gamma: batch_prefix.instruction_gamma,
        output_address_challenges: batch_prefix.output_address_challenges.clone(),
    };
    let verifier_output = Stage2ClearOutput {
        public,
        output_claims,
        product_uniskip: VerifiedProductUniSkip {
            tau_low: product_uniskip.tau_low.clone(),
            tau_high: product_uniskip.tau_high,
            input_claim: product_uniskip.input_claim,
            sumcheck_point: Point::high_to_low(vec![product_uniskip.challenge]),
            sumcheck_final_claim: product_uniskip.output_claim,
            expected_output_claim: product_uniskip.output_claim,
        },
        batch: VerifiedStage2Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(batch.challenges.clone()),
            sumcheck_final_claim: batch.output_claim,
            expected_final_claim: expected.final_claim,
            ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
            instruction_gamma: batch_prefix.instruction_gamma,
            output_address_challenges: batch_prefix.output_address_challenges,
            ram_read_write: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_read_write,
                sumcheck_point: opening_points.ram_read_write_sumcheck,
                opening_point: opening_points.ram_read_write_opening.opening_point,
                expected_output_claim: expected.ram_read_write,
            },
            product_remainder: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.product_remainder,
                sumcheck_point: opening_points.product_sumcheck,
                opening_point: opening_points.product_opening,
                expected_output_claim: expected.product_remainder,
            },
            instruction_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.instruction_claim_reduction,
                sumcheck_point: opening_points.instruction_sumcheck,
                opening_point: opening_points.instruction_opening,
                expected_output_claim: expected.instruction_claim_reduction,
            },
            ram_raf_evaluation: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_raf_evaluation,
                sumcheck_point: opening_points.ram_raf_sumcheck,
                opening_point: opening_points.ram_raf_opening,
                expected_output_claim: expected.ram_raf_evaluation,
            },
            ram_output_check: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_output_check,
                sumcheck_point: opening_points.ram_output_check_sumcheck,
                opening_point: opening_points.ram_output_check_opening,
                expected_output_claim: expected.ram_output_check,
            },
        },
    };

    Ok(Stage2ProverOutput {
        product_uniskip_proof: product_uniskip.proof,
        regular_batch_proof: batch.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "field-inline")]
pub fn prove<F, W, FI, B, T, C>(
    input: Stage2ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    FI: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>
        + SumcheckBackend<F, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }
    if input.checked.ram_K != (1usize << input.config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked RAM K {} does not match log_k {}",
                input.checked.ram_K, input.config.log_k
            ),
        });
    }

    let (stage2_rows, initial_ram_state, final_ram_state) = timed_stage2("stage2.rows", || {
        Ok::<_, ProverError>((
            input.witness.stage2_rows()?,
            input.witness.initial_ram_state_words()?,
            input.witness.final_ram_state_words()?,
        ))
    })?;
    let field_factors = timed_stage2("stage2.field_inline_materialize_factors", || {
        Stage2FieldInlineMaterializedFactors::new(input.config, input.field_inline_witness, backend)
    })?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = timed_stage2("stage2.product_uniskip", || {
        prove_stage2_product_uniskip_from_stage2_rows_field_inline::<F, T, C>(
            Stage2ProverConfig::new(input.config.log_t),
            &product_input,
            &stage2_rows,
            &field_factors,
            transcript,
        )
    })?;

    let batch_prefix = timed_stage2("stage2.prefix", || {
        derive_stage2_regular_batch_prefix(input.config, input.stage1, &product_uniskip, transcript)
    })?;
    let (ram_read_write, ram_raf, ram_output_check, instances) =
        timed_stage2("stage2.build_requests", || {
            let backend_rows = stage2_backend_rows(&stage2_rows);
            let ram_read_write = build_ram_read_write_state_request(
                input.config,
                backend_rows.clone(),
                &initial_ram_state,
                &product_uniskip,
                &batch_prefix,
            )?;
            let ram_raf = build_ram_raf_state_request(
                input.config,
                input.checked,
                input.stage1,
                backend_rows,
                &product_uniskip,
            )?;
            let ram_output_check = build_ram_output_check_state_request(
                input.config,
                input.checked,
                &final_ram_state,
                &batch_prefix,
            )?;
            let instances = build_regular_batch_instances(
                input.config,
                &stage2_rows,
                &product_uniskip,
                &batch_prefix,
                &field_factors,
            )?;
            Ok::<_, ProverError>((ram_read_write, ram_raf, ram_output_check, instances))
        })?;
    let batch = timed_stage2("stage2.regular_batch", || {
        prove_regular_batch_sumcheck::<F, T, C, B>(
            ram_read_write,
            ram_raf,
            ram_output_check,
            instances,
            backend,
            transcript,
        )
    })?;
    let opening_points =
        Stage2OpeningPoints::from_batch(input.config, &batch.challenges, &product_uniskip.tau_low)?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = timed_stage2("stage2.openings.tail", || {
        evaluate_stage2_tail_openings_from_rows(
            input.config,
            &stage2_rows,
            &opening_points.product_opening,
            &opening_points.instruction_opening,
        )
    })?;
    let field_inline = timed_stage2("stage2.openings.field_inline", || {
        evaluate_stage2_field_inline_openings_from_factors(
            input.config,
            &field_factors,
            &opening_points.field_registers_claim_reduction_opening,
        )
    })?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims = Stage2BatchOutputOpeningClaims {
        ram_read_write,
        product_remainder: tail_openings.product_remainder,
        field_inline: field_inline.clone().into(),
        instruction_claim_reduction: tail_openings.instruction_claim_reduction,
        ram_raf_evaluation: terminal.ram_raf_evaluation,
        ram_output_check: terminal.ram_output_check,
    };
    let expected = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip,
        &batch_prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected.final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            &expected,
        ));
    }

    append_stage2_opening_claims(transcript, &output_claims);

    let claims = Stage2Claims {
        product_uniskip_output_claim: product_uniskip.output_claim,
        batch_outputs: output_claims.clone(),
    };
    let public = Stage2PublicOutput {
        challenges: batch.challenges.clone(),
        batching_coefficients: batch.batching_coefficients.clone(),
        product_uniskip_challenge: product_uniskip.challenge,
        product_tau_low: product_uniskip.tau_low.clone(),
        product_tau_high: product_uniskip.tau_high,
        ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
        instruction_gamma: batch_prefix.instruction_gamma,
        field_registers_claim_reduction_gamma: batch_prefix.field_registers_claim_reduction_gamma,
        output_address_challenges: batch_prefix.output_address_challenges.clone(),
    };
    let verifier_output = Stage2ClearOutput {
        public,
        output_claims,
        product_uniskip: VerifiedProductUniSkip {
            tau_low: product_uniskip.tau_low.clone(),
            tau_high: product_uniskip.tau_high,
            input_claim: product_uniskip.input_claim,
            sumcheck_point: Point::high_to_low(vec![product_uniskip.challenge]),
            sumcheck_final_claim: product_uniskip.output_claim,
            expected_output_claim: product_uniskip.output_claim,
        },
        batch: VerifiedStage2Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(batch.challenges.clone()),
            sumcheck_final_claim: batch.output_claim,
            expected_final_claim: expected.final_claim,
            ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
            instruction_gamma: batch_prefix.instruction_gamma,
            field_registers_claim_reduction_gamma: batch_prefix
                .field_registers_claim_reduction_gamma,
            output_address_challenges: batch_prefix.output_address_challenges,
            ram_read_write: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_read_write,
                sumcheck_point: opening_points.ram_read_write_sumcheck,
                opening_point: opening_points.ram_read_write_opening.opening_point,
                expected_output_claim: expected.ram_read_write,
            },
            product_remainder: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.product_remainder,
                sumcheck_point: opening_points.product_sumcheck.clone(),
                opening_point: opening_points.product_opening.clone(),
                expected_output_claim: expected.product_remainder,
            },
            instruction_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.instruction_claim_reduction,
                sumcheck_point: opening_points.instruction_sumcheck,
                opening_point: opening_points.instruction_opening,
                expected_output_claim: expected.instruction_claim_reduction,
            },
            field_registers_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.field_registers_claim_reduction,
                sumcheck_point: opening_points.field_registers_claim_reduction_sumcheck,
                opening_point: opening_points.field_registers_claim_reduction_opening,
                expected_output_claim: expected.field_registers_claim_reduction,
            },
            ram_raf_evaluation: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_raf_evaluation,
                sumcheck_point: opening_points.ram_raf_sumcheck,
                opening_point: opening_points.ram_raf_opening,
                expected_output_claim: expected.ram_raf_evaluation,
            },
            ram_output_check: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_output_check,
                sumcheck_point: opening_points.ram_output_check_sumcheck,
                opening_point: opening_points.ram_output_check_opening,
                expected_output_claim: expected.ram_output_check,
            },
        },
    };

    Ok(Stage2ProverOutput {
        product_uniskip_proof: product_uniskip.proof,
        regular_batch_proof: batch.proof,
        claims,
        verifier_output,
    })
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
pub fn prove_committed_boundary<F, W, B, T, VC>(
    input: Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<Stage2CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if !input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 committed prover received non-ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }
    if input.checked.ram_K != (1usize << input.config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked RAM K {} does not match log_k {}",
                input.checked.ram_K, input.config.log_k
            ),
        });
    }

    let (stage2_rows, initial_ram_state, final_ram_state) = timed_stage2("stage2.rows", || {
        Ok::<_, ProverError>((
            input.witness.stage2_rows()?,
            input.witness.initial_ram_state_words()?,
            input.witness.final_ram_state_words()?,
        ))
    })?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = timed_stage2("stage2.product_uniskip", || {
        prove_stage2_product_uniskip_committed_from_stage2_rows::<F, B, T, VC>(
            Stage2ProverConfig::new(input.config.log_t),
            &product_input,
            &stage2_rows,
            backend,
            vc_setup,
            transcript,
        )
    })?;

    let batch_prefix = timed_stage2("stage2.prefix", || {
        derive_stage2_regular_batch_prefix(
            input.config,
            input.stage1,
            &product_uniskip.output,
            transcript,
        )
    })?;
    let backend_rows = stage2_backend_rows(&stage2_rows);
    let ram_read_write = build_ram_read_write_state_request(
        input.config,
        backend_rows.clone(),
        &initial_ram_state,
        &product_uniskip.output,
        &batch_prefix,
    )?;
    let ram_raf = build_ram_raf_state_request(
        input.config,
        input.checked,
        input.stage1,
        backend_rows,
        &product_uniskip.output,
    )?;
    let ram_output_check = build_ram_output_check_state_request(
        input.config,
        input.checked,
        &final_ram_state,
        &batch_prefix,
    )?;
    let instances = build_regular_batch_instances(
        input.config,
        &stage2_rows,
        &product_uniskip.output,
        &batch_prefix,
    )?;
    let batch = prove_regular_batch_sumcheck_committed::<F, T, B, VC>(
        ram_read_write,
        ram_raf,
        ram_output_check,
        instances,
        backend,
        vc_setup,
        transcript,
    )?;
    let opening_points = Stage2OpeningPoints::from_batch(
        input.config,
        &batch.challenges,
        &product_uniskip.output.tau_low,
    )?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = evaluate_stage2_tail_openings_from_rows(
        input.config,
        &stage2_rows,
        &opening_points.product_opening,
        &opening_points.instruction_opening,
    )?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims = Stage2BatchOutputOpeningClaims {
        ram_read_write,
        product_remainder: tail_openings.product_remainder,
        instruction_claim_reduction: tail_openings.instruction_claim_reduction,
        ram_raf_evaluation: terminal.ram_raf_evaluation,
        ram_output_check: terminal.ram_output_check,
    };
    let expected = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip.output,
        &batch_prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected.final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            &expected,
        ));
    }

    let batch_output_claim_values = stage2_committed_output_claim_values(&output_claims);
    let public = Stage2PublicOutput {
        challenges: batch.challenges.clone(),
        batching_coefficients: batch.batching_coefficients.clone(),
        product_uniskip_challenge: product_uniskip.output.challenge,
        product_tau_low: product_uniskip.output.tau_low.clone(),
        product_tau_high: product_uniskip.output.tau_high,
        ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
        instruction_gamma: batch_prefix.instruction_gamma,
        output_address_challenges: batch_prefix.output_address_challenges.clone(),
    };
    let verifier_output = Stage2ClearOutput {
        public: public.clone(),
        output_claims,
        product_uniskip: VerifiedProductUniSkip {
            tau_low: product_uniskip.output.tau_low.clone(),
            tau_high: product_uniskip.output.tau_high,
            input_claim: product_uniskip.output.input_claim,
            sumcheck_point: Point::high_to_low(vec![product_uniskip.output.challenge]),
            sumcheck_final_claim: product_uniskip.output.output_claim,
            expected_output_claim: product_uniskip.output.output_claim,
        },
        batch: VerifiedStage2Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(batch.challenges.clone()),
            sumcheck_final_claim: batch.output_claim,
            expected_final_claim: expected.final_claim,
            ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
            instruction_gamma: batch_prefix.instruction_gamma,
            output_address_challenges: batch_prefix.output_address_challenges,
            ram_read_write: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_read_write,
                sumcheck_point: opening_points.ram_read_write_sumcheck,
                opening_point: opening_points.ram_read_write_opening.opening_point,
                expected_output_claim: expected.ram_read_write,
            },
            product_remainder: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.product_remainder,
                sumcheck_point: opening_points.product_sumcheck,
                opening_point: opening_points.product_opening,
                expected_output_claim: expected.product_remainder,
            },
            instruction_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.instruction_claim_reduction,
                sumcheck_point: opening_points.instruction_sumcheck,
                opening_point: opening_points.instruction_opening,
                expected_output_claim: expected.instruction_claim_reduction,
            },
            ram_raf_evaluation: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_raf_evaluation,
                sumcheck_point: opening_points.ram_raf_sumcheck,
                opening_point: opening_points.ram_raf_opening,
                expected_output_claim: expected.ram_raf_evaluation,
            },
            ram_output_check: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_output_check,
                sumcheck_point: opening_points.ram_output_check_sumcheck,
                opening_point: opening_points.ram_output_check_opening,
                expected_output_claim: expected.ram_output_check,
            },
        },
    };
    let built_batch = batch.finish(&batch_output_claim_values, transcript)?;

    Ok(Stage2CommittedBoundaryOutput {
        product_uniskip_proof: product_uniskip.output.proof,
        regular_batch_proof: built_batch.proof,
        public,
        verifier_output,
        product_uniskip_output_claim_values: product_uniskip.output_claim_values,
        batch_output_claim_values,
        product_uniskip_committed_witness: product_uniskip.committed_witness,
        batch_committed_witness: built_batch.witness,
    })
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
pub fn prove_committed_boundary<F, W, FI, B, T, VC>(
    input: Stage2ProverInput<'_, F, W, FI>,
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<Stage2CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    FI: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>
        + SumcheckBackend<F, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if !input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 committed prover received non-ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }
    if input.checked.ram_K != (1usize << input.config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked RAM K {} does not match log_k {}",
                input.checked.ram_K, input.config.log_k
            ),
        });
    }

    let (stage2_rows, initial_ram_state, final_ram_state) = timed_stage2("stage2.rows", || {
        Ok::<_, ProverError>((
            input.witness.stage2_rows()?,
            input.witness.initial_ram_state_words()?,
            input.witness.final_ram_state_words()?,
        ))
    })?;
    let field_factors = timed_stage2("stage2.field_inline_materialize_factors", || {
        Stage2FieldInlineMaterializedFactors::new(input.config, input.field_inline_witness, backend)
    })?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = timed_stage2("stage2.product_uniskip", || {
        prove_stage2_product_uniskip_committed_from_stage2_rows_field_inline::<F, T, VC>(
            Stage2ProverConfig::new(input.config.log_t),
            &product_input,
            &stage2_rows,
            &field_factors,
            vc_setup,
            transcript,
        )
    })?;

    let batch_prefix = timed_stage2("stage2.prefix", || {
        derive_stage2_regular_batch_prefix(
            input.config,
            input.stage1,
            &product_uniskip.output,
            transcript,
        )
    })?;
    let (ram_read_write, ram_raf, ram_output_check, instances) =
        timed_stage2("stage2.build_requests", || {
            let backend_rows = stage2_backend_rows(&stage2_rows);
            let ram_read_write = build_ram_read_write_state_request(
                input.config,
                backend_rows.clone(),
                &initial_ram_state,
                &product_uniskip.output,
                &batch_prefix,
            )?;
            let ram_raf = build_ram_raf_state_request(
                input.config,
                input.checked,
                input.stage1,
                backend_rows,
                &product_uniskip.output,
            )?;
            let ram_output_check = build_ram_output_check_state_request(
                input.config,
                input.checked,
                &final_ram_state,
                &batch_prefix,
            )?;
            let instances = build_regular_batch_instances(
                input.config,
                &stage2_rows,
                &product_uniskip.output,
                &batch_prefix,
                &field_factors,
            )?;
            Ok::<_, ProverError>((ram_read_write, ram_raf, ram_output_check, instances))
        })?;
    let batch = timed_stage2("stage2.regular_batch", || {
        prove_regular_batch_sumcheck_committed::<F, T, B, VC>(
            ram_read_write,
            ram_raf,
            ram_output_check,
            instances,
            backend,
            vc_setup,
            transcript,
        )
    })?;
    let opening_points = Stage2OpeningPoints::from_batch(
        input.config,
        &batch.challenges,
        &product_uniskip.output.tau_low,
    )?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = timed_stage2("stage2.openings.tail", || {
        evaluate_stage2_tail_openings_from_rows(
            input.config,
            &stage2_rows,
            &opening_points.product_opening,
            &opening_points.instruction_opening,
        )
    })?;
    let field_inline = timed_stage2("stage2.openings.field_inline", || {
        evaluate_stage2_field_inline_openings_from_factors(
            input.config,
            &field_factors,
            &opening_points.field_registers_claim_reduction_opening,
        )
    })?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims = Stage2BatchOutputOpeningClaims {
        ram_read_write,
        product_remainder: tail_openings.product_remainder,
        field_inline: field_inline.clone().into(),
        instruction_claim_reduction: tail_openings.instruction_claim_reduction,
        ram_raf_evaluation: terminal.ram_raf_evaluation,
        ram_output_check: terminal.ram_output_check,
    };
    let expected = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip.output,
        &batch_prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected.final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            &expected,
        ));
    }

    let batch_output_claim_values = stage2_committed_output_claim_values(&output_claims);
    let public = Stage2PublicOutput {
        challenges: batch.challenges.clone(),
        batching_coefficients: batch.batching_coefficients.clone(),
        product_uniskip_challenge: product_uniskip.output.challenge,
        product_tau_low: product_uniskip.output.tau_low.clone(),
        product_tau_high: product_uniskip.output.tau_high,
        ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
        instruction_gamma: batch_prefix.instruction_gamma,
        field_registers_claim_reduction_gamma: batch_prefix.field_registers_claim_reduction_gamma,
        output_address_challenges: batch_prefix.output_address_challenges.clone(),
    };
    let verifier_output = Stage2ClearOutput {
        public: public.clone(),
        output_claims,
        product_uniskip: VerifiedProductUniSkip {
            tau_low: product_uniskip.output.tau_low.clone(),
            tau_high: product_uniskip.output.tau_high,
            input_claim: product_uniskip.output.input_claim,
            sumcheck_point: Point::high_to_low(vec![product_uniskip.output.challenge]),
            sumcheck_final_claim: product_uniskip.output.output_claim,
            expected_output_claim: product_uniskip.output.output_claim,
        },
        batch: VerifiedStage2Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(batch.challenges.clone()),
            sumcheck_final_claim: batch.output_claim,
            expected_final_claim: expected.final_claim,
            ram_read_write_gamma: batch_prefix.ram_read_write_gamma,
            instruction_gamma: batch_prefix.instruction_gamma,
            field_registers_claim_reduction_gamma: batch_prefix
                .field_registers_claim_reduction_gamma,
            output_address_challenges: batch_prefix.output_address_challenges,
            ram_read_write: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_read_write,
                sumcheck_point: opening_points.ram_read_write_sumcheck,
                opening_point: opening_points.ram_read_write_opening.opening_point,
                expected_output_claim: expected.ram_read_write,
            },
            product_remainder: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.product_remainder,
                sumcheck_point: opening_points.product_sumcheck.clone(),
                opening_point: opening_points.product_opening.clone(),
                expected_output_claim: expected.product_remainder,
            },
            instruction_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.instruction_claim_reduction,
                sumcheck_point: opening_points.instruction_sumcheck,
                opening_point: opening_points.instruction_opening,
                expected_output_claim: expected.instruction_claim_reduction,
            },
            field_registers_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.field_registers_claim_reduction,
                sumcheck_point: opening_points.field_registers_claim_reduction_sumcheck,
                opening_point: opening_points.field_registers_claim_reduction_opening,
                expected_output_claim: expected.field_registers_claim_reduction,
            },
            ram_raf_evaluation: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_raf_evaluation,
                sumcheck_point: opening_points.ram_raf_sumcheck,
                opening_point: opening_points.ram_raf_opening,
                expected_output_claim: expected.ram_raf_evaluation,
            },
            ram_output_check: VerifiedStage2Sumcheck {
                input_claim: batch_prefix.input_claims.ram_output_check,
                sumcheck_point: opening_points.ram_output_check_sumcheck,
                opening_point: opening_points.ram_output_check_opening,
                expected_output_claim: expected.ram_output_check,
            },
        },
    };
    let built_batch = batch.finish(&batch_output_claim_values, transcript)?;

    Ok(Stage2CommittedBoundaryOutput {
        product_uniskip_proof: product_uniskip.output.proof,
        regular_batch_proof: built_batch.proof,
        public,
        verifier_output,
        product_uniskip_output_claim_values: product_uniskip.output_claim_values,
        batch_output_claim_values,
        product_uniskip_committed_witness: product_uniskip.committed_witness,
        batch_committed_witness: built_batch.witness,
    })
}

#[cfg(not(feature = "field-inline"))]
pub fn prove_stage2_product_uniskip<F, W, B, T, C>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    witness: &W,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [input.product, input.should_branch, input.should_jump];
    let extended_evals = product_uniskip_extended_evals(config, witness, backend, &input.tau_low)?;
    prove_stage2_product_uniskip_with_extended_evals(
        input,
        &base_evals,
        &extended_evals,
        transcript,
    )
}

#[cfg(not(feature = "field-inline"))]
fn prove_stage2_product_uniskip_from_stage2_rows<F, B, T, C>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [input.product, input.should_branch, input.should_jump];
    let product_rows = product_uniskip_rows_from_stage2_rows(&config, stage2_rows)?;
    let extended_evals =
        product_uniskip_extended_evals_from_rows(config, &product_rows, backend, &input.tau_low)?;
    prove_stage2_product_uniskip_with_extended_evals(
        input,
        &base_evals,
        &extended_evals,
        transcript,
    )
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
#[expect(
    dead_code,
    reason = "stage2-row path is the active frontier entry point"
)]
fn prove_stage2_product_uniskip_committed<F, W, B, T, VC>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    witness: &W,
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [input.product, input.should_branch, input.should_jump];
    let extended_evals = product_uniskip_extended_evals(config, witness, backend, &input.tau_low)?;
    prove_stage2_product_uniskip_committed_with_extended_evals::<F, T, VC>(
        input,
        &base_evals,
        &extended_evals,
        vc_setup,
        transcript,
    )
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn prove_stage2_product_uniskip_committed_from_stage2_rows<F, B, T, VC>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [input.product, input.should_branch, input.should_jump];
    let product_rows = product_uniskip_rows_from_stage2_rows(&config, stage2_rows)?;
    let extended_evals =
        product_uniskip_extended_evals_from_rows(config, &product_rows, backend, &input.tau_low)?;
    prove_stage2_product_uniskip_committed_with_extended_evals::<F, T, VC>(
        input,
        &base_evals,
        &extended_evals,
        vc_setup,
        transcript,
    )
}

#[cfg(feature = "field-inline")]
pub fn prove_stage2_product_uniskip<F, W, FI, B, T, C>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    FI: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + SumcheckBackend<F, FieldInlineNamespace>,
    T: Transcript<Challenge = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [
        input.product,
        input.should_branch,
        input.should_jump,
        input.field_product,
        input.field_inv_product,
    ];
    let extended_evals = product_uniskip_extended_evals_field_inline(
        config,
        witness,
        field_inline_witness,
        backend,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_with_extended_evals(
        input,
        &base_evals,
        &extended_evals,
        transcript,
    )
}

#[cfg(feature = "field-inline")]
fn prove_stage2_product_uniskip_from_stage2_rows_field_inline<F, T, C>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [
        input.product,
        input.should_branch,
        input.should_jump,
        input.field_product,
        input.field_inv_product,
    ];
    let extended_evals = product_uniskip_extended_evals_field_inline_from_stage2_rows(
        config,
        stage2_rows,
        field_factors,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_with_extended_evals(
        input,
        &base_evals,
        &extended_evals,
        transcript,
    )
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
#[expect(
    dead_code,
    reason = "stage2-row path is the active frontier entry point"
)]
fn prove_stage2_product_uniskip_committed<F, W, FI, B, T, VC>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    FI: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + SumcheckBackend<F, FieldInlineNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [
        input.product,
        input.should_branch,
        input.should_jump,
        input.field_product,
        input.field_inv_product,
    ];
    let extended_evals = product_uniskip_extended_evals_field_inline(
        config,
        witness,
        field_inline_witness,
        backend,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_committed_with_extended_evals::<F, T, VC>(
        input,
        &base_evals,
        &extended_evals,
        vc_setup,
        transcript,
    )
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
fn prove_stage2_product_uniskip_committed_from_stage2_rows_field_inline<F, T, VC>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if input.tau_low.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {}",
            input.tau_low.len(),
            config.log_t
        )));
    }

    let base_evals = [
        input.product,
        input.should_branch,
        input.should_jump,
        input.field_product,
        input.field_inv_product,
    ];
    let extended_evals = product_uniskip_extended_evals_field_inline_from_stage2_rows(
        config,
        stage2_rows,
        field_factors,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_committed_with_extended_evals::<F, T, VC>(
        input,
        &base_evals,
        &extended_evals,
        vc_setup,
        transcript,
    )
}

fn prove_stage2_product_uniskip_with_extended_evals<F, T, C>(
    input: &Stage2ProductUniSkipInput<F>,
    base_evals: &[F; SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let tau_high = transcript.challenge();
    let uniskip_poly =
        build_product_uniskip_first_round_poly(base_evals, extended_evals, tau_high)?;

    let weights = centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)
        .map_err(invalid_sumcheck_output)?;
    let input_claim = selected_product_uniskip_input_claim(input, &weights)?;
    let round_sum = centered_domain_sum(&uniskip_poly, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE)?;
    if round_sum != input_claim {
        return Err(invalid_sumcheck_output(
            "Stage 2 product uni-skip first-round polynomial does not sum to the input claim",
        ));
    }

    LabeledRoundPoly::uniskip(&uniskip_poly).append_to_transcript(transcript);
    let challenge = transcript.challenge();
    let output_claim = uniskip_poly.evaluate(challenge);
    transcript.append_labeled(b"opening_claim", &output_claim);

    Ok(Stage2ProductUniSkipOutput {
        proof: SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
            round_polynomials: vec![uniskip_poly],
        })),
        input_claim,
        output_claim,
        challenge,
        tau_high,
        tau_low: input.tau_low.clone(),
    })
}

#[cfg(feature = "zk")]
fn prove_stage2_product_uniskip_committed_with_extended_evals<F, T, VC>(
    input: &Stage2ProductUniSkipInput<F>,
    base_evals: &[F; SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let tau_high = transcript.challenge();
    let uniskip_poly =
        build_product_uniskip_first_round_poly(base_evals, extended_evals, tau_high)?;

    let weights = centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)
        .map_err(invalid_sumcheck_output)?;
    let input_claim = selected_product_uniskip_input_claim(input, &weights)?;
    let round_sum = centered_domain_sum(&uniskip_poly, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE)?;
    if round_sum != input_claim {
        return Err(invalid_sumcheck_output(
            "Stage 2 product uni-skip first-round polynomial does not sum to the input claim",
        ));
    }

    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, 1)?;
    let challenge = builder.commit_round(&uniskip_poly, transcript)?;
    let output_claim = uniskip_poly.evaluate(challenge);
    let output_claim_values = vec![output_claim];
    let built = builder.finish(&output_claim_values, transcript)?;

    Ok(CommittedProductUniSkip {
        output: Stage2ProductUniSkipOutput {
            proof: built.proof,
            input_claim,
            output_claim,
            challenge,
            tau_high,
            tau_low: input.tau_low.clone(),
        },
        output_claim_values,
        committed_witness: built.witness,
    })
}

pub fn derive_stage2_regular_batch_prefix<F, T, C>(
    config: Stage2BatchProverConfig,
    stage1: &Stage1ClearOutput<F>,
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    transcript: &mut T,
) -> Result<Stage2RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let read_write_dimensions = config.rw_config.ram_dimensions(config.log_t, config.log_k);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: format!("invalid Stage 2 RAM read-write dimensions: {error}"),
            }
        })?;

    let ram_read_write_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_registers_claim_reduction_gamma = transcript.challenge_scalar();
    let output_address_challenges = (0..config.log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    let gamma2 = instruction_gamma * instruction_gamma;
    let gamma3 = gamma2 * instruction_gamma;
    let gamma4 = gamma3 * instruction_gamma;
    #[cfg(feature = "field-inline")]
    let field_gamma2 =
        field_registers_claim_reduction_gamma * field_registers_claim_reduction_gamma;
    let input_claims = Stage2RegularBatchInputClaims {
        ram_read_write: stage1.outer.ram_read_value
            + ram_read_write_gamma * stage1.outer.ram_write_value,
        product_remainder: product_uniskip.output_claim,
        instruction_claim_reduction: stage1.outer.lookup_output
            + instruction_gamma * stage1.outer.left_lookup_operand
            + gamma2 * stage1.outer.right_lookup_operand
            + gamma3 * stage1.outer.left_instruction_input
            + gamma4 * stage1.outer.right_instruction_input,
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction: stage1.field_inline.field_rd_value
            + field_registers_claim_reduction_gamma * stage1.field_inline.field_rs1_value
            + field_gamma2 * stage1.field_inline.field_rs2_value,
        ram_raf_evaluation: F::pow2(raf_dimensions.phase3_cycle_rounds())
            * stage1.outer.ram_address,
        ram_output_check: F::zero(),
    };

    Ok(Stage2RegularBatchPrefixOutput {
        input_claims,
        ram_read_write_gamma,
        instruction_gamma,
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction_gamma,
        output_address_challenges,
    })
}

pub fn evaluate_stage2_product_remainder_openings<F, W, B>(
    config: Stage2ProverConfig,
    witness: &W,
    backend: &mut B,
    opening_point: Vec<F>,
) -> Result<ProductRemainderOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    if opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 product-remainder opening point has {} variables, expected {}",
                opening_point.len(),
                config.log_t
            ),
        });
    }
    let final_cycle = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        ProverError::InvalidStageRequest {
            reason: format!("Stage 2 trace length overflows for log_t={}", config.log_t),
        }
    })? - 1;
    let final_cycle_eq = eq_index_msb(&opening_point, final_cycle);
    let request =
        build_stage2_product_remainder_opening_evaluation_request(config, witness, opening_point)?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    let mut claims = product_remainder_openings_from_evaluations(&request, evaluations)?;
    // Product virtualization exposes `1 - not_next_noop`; core sets
    // `not_next_noop = false` on the final cycle so the product-stage
    // NextIsNoop opening includes a final-cycle one.
    claims.next_is_noop += final_cycle_eq;
    Ok(claims)
}

pub fn evaluate_stage2_instruction_claim_openings<F, W, B>(
    config: Stage2ProverConfig,
    witness: &W,
    backend: &mut B,
    opening_point: Vec<F>,
) -> Result<InstructionClaimReductionOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    if opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 instruction-claim opening point has {} variables, expected {}",
                opening_point.len(),
                config.log_t
            ),
        });
    }
    let request =
        build_stage2_instruction_claim_opening_evaluation_request(config, witness, opening_point)?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    instruction_claim_openings_from_evaluations(&request, evaluations)
}

fn evaluate_stage2_tail_openings_from_rows<F>(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
    product_opening_point: &[F],
    instruction_opening_point: &[F],
) -> Result<Stage2TailOutputOpenings<F>, ProverError>
where
    F: Field,
{
    if product_opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 product-remainder opening point has {} variables, expected {}",
                product_opening_point.len(),
                config.log_t
            ),
        });
    }
    if instruction_opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 instruction-claim opening point has {} variables, expected {}",
                instruction_opening_point.len(),
                config.log_t
            ),
        });
    }
    let expected_rows = 1usize << config.log_t;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 tail opening inputs have inconsistent row counts".to_owned(),
        });
    }

    let product_eq = EqPolynomial::<F>::evals(product_opening_point, None);
    let product = (0..expected_rows)
        .into_par_iter()
        .map(|cycle| {
            let eq = product_eq[cycle];
            let row = &rows[cycle];
            (
                eq.mul_u64(row.left_instruction_input),
                eq.mul_i128(row.right_instruction_input),
                bool_term(eq, row.jump_flag),
                bool_term(eq, row.write_lookup_output_to_rd_flag),
                eq.mul_u64(row.lookup_output),
                bool_term(eq, row.branch_flag),
                bool_term(eq, row.next_is_noop),
                bool_term(eq, row.virtual_instruction_flag),
            )
        })
        .reduce(
            || {
                (
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                )
            },
            |left, right| {
                (
                    left.0 + right.0,
                    left.1 + right.1,
                    left.2 + right.2,
                    left.3 + right.3,
                    left.4 + right.4,
                    left.5 + right.5,
                    left.6 + right.6,
                    left.7 + right.7,
                )
            },
        );

    let instruction_eq = EqPolynomial::<F>::evals(instruction_opening_point, None);
    let (left_lookup_operand, right_lookup_operand) = (0..expected_rows)
        .into_par_iter()
        .map(|cycle| {
            let eq = instruction_eq[cycle];
            let row = &rows[cycle];
            (
                eq.mul_u64(row.left_lookup_operand),
                eq.mul_u128(row.right_lookup_operand),
            )
        })
        .reduce(
            || (F::zero(), F::zero()),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );

    Ok(Stage2TailOutputOpenings {
        product_remainder: ProductRemainderOutputOpeningClaims {
            left_instruction_input: product.0,
            right_instruction_input: product.1,
            jump_flag: product.2,
            write_lookup_output_to_rd: product.3,
            lookup_output: product.4,
            branch_flag: product.5,
            next_is_noop: product.6,
            virtual_instruction: product.7,
        },
        instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims {
            lookup_output: None,
            left_lookup_operand,
            right_lookup_operand,
            left_instruction_input: None,
            right_instruction_input: None,
        },
    })
}

#[cfg(feature = "field-inline")]
pub fn evaluate_stage2_field_inline_product_openings<F, W, B>(
    config: Stage2ProverConfig,
    witness: &W,
    backend: &mut B,
    opening_point: Vec<F>,
) -> Result<Stage2FieldInlineProductOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, FieldInlineNamespace>,
{
    if opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 field-inline product opening point has {} variables, expected {}",
                opening_point.len(),
                config.log_t
            ),
        });
    }
    let variables = [
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineVirtualPolynomial::FieldRdValue,
    ];
    let views = variables
        .iter()
        .copied()
        .enumerate()
        .map(|(index, variable)| {
            let oracle = OracleRef::virtual_polynomial(variable);
            let view = primary_view_requirement::<F, W, FieldInlineNamespace>(witness, oracle)?;
            Ok(SumcheckViewEvaluationRequest::new(
                BackendValueSlot(index as u32),
                view,
                opening_point.clone(),
            ))
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let request =
        SumcheckEvaluationRequest::new("stage2.field_inline_product.output_openings", views);
    let evaluations = backend.evaluate_sumcheck_views(&request, witness)?;
    let mut values = evaluations
        .into_iter()
        .map(|output| (output.slot, output.value))
        .collect::<std::collections::BTreeMap<_, _>>();
    Ok(Stage2FieldInlineProductOutputOpeningClaims {
        field_rs1_value: values
            .remove(&BackendValueSlot(0))
            .ok_or_else(|| invalid_sumcheck_output("missing Stage 2 field-inline rs1 opening"))?,
        field_rs2_value: values
            .remove(&BackendValueSlot(1))
            .ok_or_else(|| invalid_sumcheck_output("missing Stage 2 field-inline rs2 opening"))?,
        field_rd_value: values
            .remove(&BackendValueSlot(2))
            .ok_or_else(|| invalid_sumcheck_output("missing Stage 2 field-inline rd opening"))?,
    })
}

#[cfg(feature = "field-inline")]
fn evaluate_stage2_field_inline_openings_from_factors<F>(
    config: Stage2BatchProverConfig,
    factors: &Stage2FieldInlineMaterializedFactors<F>,
    opening_point: &[F],
) -> Result<Stage2FieldInlineProductOutputOpeningClaims<F>, ProverError>
where
    F: Field,
{
    if opening_point.len() != config.log_t {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 field-inline opening point has {} variables, expected {}",
                opening_point.len(),
                config.log_t
            ),
        });
    }
    let rows = 1usize << config.log_t;
    if factors.rs1.len() != rows || factors.rs2.len() != rows || factors.rd.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 field-inline factors have inconsistent row counts".to_owned(),
        });
    }

    let eq = EqPolynomial::<F>::evals(opening_point, None);
    let (rs1, rs2, rd) = (0..rows)
        .into_par_iter()
        .map(|cycle| {
            let eq = eq[cycle];
            (
                eq * factors.rs1[cycle],
                eq * factors.rs2[cycle],
                eq * factors.rd[cycle],
            )
        })
        .reduce(
            || (F::zero(), F::zero(), F::zero()),
            |left, right| (left.0 + right.0, left.1 + right.1, left.2 + right.2),
        );
    Ok(Stage2FieldInlineProductOutputOpeningClaims {
        field_rs1_value: rs1,
        field_rs2_value: rs2,
        field_rd_value: rd,
    })
}

pub fn evaluate_stage2_ram_read_write_openings<F, W, B>(
    config: Stage2BatchProverConfig,
    witness: &W,
    backend: &mut B,
    opening_point: Vec<F>,
) -> Result<RamReadWriteOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let expected = config.log_k + config.log_t;
    if opening_point.len() != expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM read-write opening point has {} variables, expected {expected}",
                opening_point.len()
            ),
        });
    }
    let request =
        build_stage2_ram_read_write_opening_evaluation_request(config, witness, opening_point)?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    ram_read_write_openings_from_evaluations(&request, evaluations)
}

pub fn evaluate_stage2_ram_terminal_openings<F, W, B>(
    config: Stage2BatchProverConfig,
    witness: &W,
    backend: &mut B,
    ram_raf_opening_point: Vec<F>,
    ram_output_check_opening_point: Vec<F>,
) -> Result<Stage2RamTerminalOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let ram_raf_expected = config.log_k + config.log_t;
    if ram_raf_opening_point.len() != ram_raf_expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM RAF evaluation opening point has {} variables, expected {ram_raf_expected}",
                ram_raf_opening_point.len()
            ),
        });
    }
    if ram_output_check_opening_point.len() != config.log_k {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM output-check opening point has {} variables, expected {}",
                ram_output_check_opening_point.len(),
                config.log_k
            ),
        });
    }
    let request = build_stage2_ram_terminal_opening_evaluation_request(
        config,
        witness,
        ram_raf_opening_point,
        ram_output_check_opening_point,
    )?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    ram_terminal_openings_from_evaluations(&request, evaluations)
}

struct RegularBatchProof<F: Field, C> {
    proof: SumcheckProof<F, C>,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    output_claim: F,
    ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

#[derive(Clone)]
struct Stage2TailOutputOpenings<F: Field> {
    product_remainder: ProductRemainderOutputOpeningClaims<F>,
    instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims<F>,
}

#[cfg(feature = "zk")]
struct CommittedProductUniSkip<F: Field, C> {
    output: Stage2ProductUniSkipOutput<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
struct PendingCommittedRegularBatch<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    output_claim: F,
    ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

#[cfg(feature = "zk")]
impl<F, VC> PendingCommittedRegularBatch<'_, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<crate::committed::BuiltCommittedSumcheck<F, VC::Output>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        self.builder.finish(output_claim_values, transcript)
    }
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
#[derive(Clone)]
pub struct Stage2RegularBatchFrontierProof<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub output_claim: F,
}

struct Stage2OpeningPoints<F: Field> {
    ram_read_write_sumcheck: Vec<F>,
    ram_read_write_opening: ReadWriteOpeningPoint<F>,
    product_sumcheck: Vec<F>,
    product_opening: Vec<F>,
    instruction_sumcheck: Vec<F>,
    instruction_opening: Vec<F>,
    #[cfg(feature = "field-inline")]
    field_registers_claim_reduction_sumcheck: Vec<F>,
    #[cfg(feature = "field-inline")]
    field_registers_claim_reduction_opening: Vec<F>,
    ram_raf_sumcheck: Vec<F>,
    ram_raf_opening: Vec<F>,
    ram_output_check_sumcheck: Vec<F>,
    ram_output_check_opening: Vec<F>,
}

impl<F: Field> Stage2OpeningPoints<F> {
    fn from_batch(
        config: Stage2BatchProverConfig,
        challenges: &[F],
        product_tau_low: &[F],
    ) -> Result<Self, ProverError> {
        let read_write_dimensions = config.rw_config.ram_dimensions(config.log_t, config.log_k);
        let read_write_rounds = config.log_t + config.log_k;
        if challenges.len() != read_write_rounds {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch returned {} challenges, expected {read_write_rounds}",
                challenges.len()
            )));
        }

        let ram_read_write_sumcheck = challenges.to_vec();
        let ram_read_write_opening = read_write_dimensions
            .read_write_opening_point(&ram_read_write_sumcheck)
            .map_err(invalid_sumcheck_output)?;

        let product_offset = read_write_rounds - config.log_t;
        let product_sumcheck = challenges[product_offset..].to_vec();
        let mut product_opening = product_sumcheck.clone();
        product_opening.reverse();
        let instruction_sumcheck = product_sumcheck.clone();
        let instruction_opening = product_opening.clone();

        let ram_terminal_rounds =
            config.log_t + config.log_k - read_write_dimensions.phase1_num_rounds();
        let terminal_offset = read_write_dimensions.phase1_num_rounds();
        let terminal_end = terminal_offset + ram_terminal_rounds;
        let terminal_point = challenges
            .get(terminal_offset..terminal_end)
            .ok_or_else(|| {
                invalid_sumcheck_output(format!(
                    "Stage 2 terminal point range {terminal_offset}..{terminal_end} exceeds {} challenges",
                    challenges.len()
                ))
            })?
            .to_vec();
        let ram_raf_address_point = read_write_dimensions
            .address_opening_point(&terminal_point)
            .map_err(invalid_sumcheck_output)?;
        let ram_raf_opening = [ram_raf_address_point.as_slice(), product_tau_low].concat();
        let ram_output_check_opening = ram_raf_address_point;

        Ok(Self {
            ram_read_write_sumcheck,
            ram_read_write_opening,
            product_sumcheck: product_sumcheck.clone(),
            product_opening: product_opening.clone(),
            instruction_sumcheck,
            instruction_opening,
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction_sumcheck: product_sumcheck.clone(),
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction_opening: product_opening.clone(),
            ram_raf_sumcheck: terminal_point.clone(),
            ram_raf_opening,
            ram_output_check_sumcheck: terminal_point,
            ram_output_check_opening,
        })
    }
}

struct ExpectedRegularBatchOutputs<F: Field> {
    ram_read_write: F,
    product_remainder: F,
    instruction_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_claim_reduction: F,
    ram_raf_evaluation: F,
    ram_output_check: F,
    final_claim: F,
}

fn stage2_regular_batch_output_mismatch<F: Field>(
    batch_output_claim: F,
    expected: &ExpectedRegularBatchOutputs<F>,
) -> ProverError {
    #[cfg(not(feature = "field-inline"))]
    let reason = format!(
        "Stage 2 regular batch final claim did not match output openings: got {}, expected {}; components ram_read_write={}, product_remainder={}, instruction_claim_reduction={}, ram_raf_evaluation={}, ram_output_check={}",
        batch_output_claim,
        expected.final_claim,
        expected.ram_read_write,
        expected.product_remainder,
        expected.instruction_claim_reduction,
        expected.ram_raf_evaluation,
        expected.ram_output_check,
    );
    #[cfg(feature = "field-inline")]
    let reason = format!(
        "Stage 2 regular batch final claim did not match output openings: got {}, expected {}; components ram_read_write={}, product_remainder={}, instruction_claim_reduction={}, field_registers_claim_reduction={}, ram_raf_evaluation={}, ram_output_check={}",
        batch_output_claim,
        expected.final_claim,
        expected.ram_read_write,
        expected.product_remainder,
        expected.instruction_claim_reduction,
        expected.field_registers_claim_reduction,
        expected.ram_raf_evaluation,
        expected.ram_output_check,
    );
    invalid_sumcheck_output(reason)
}

fn prove_regular_batch_sumcheck<F, T, C, B>(
    ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    ram_raf: SumcheckRamRafStateRequest<F>,
    ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<RegularBatchProof<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
{
    let mut ram_state = timed_stage2("stage2.regular_batch.materialize_ram_read_write", || {
        backend.materialize_sumcheck_ram_read_write_state(&ram_read_write)
    })?;
    let mut ram_raf_state = timed_stage2("stage2.regular_batch.materialize_ram_raf", || {
        backend.materialize_sumcheck_ram_raf_state(&ram_raf)
    })?;
    let mut ram_output_check_state =
        timed_stage2("stage2.regular_batch.materialize_ram_output_check", || {
            backend.materialize_sumcheck_ram_output_check_state(&ram_output_check)
        })?;
    let mut state = timed_stage2("stage2.regular_batch.materialize_tail", || {
        Ok::<_, ProverError>(SumcheckRegularBatchState::new(
            "stage2.regular_batch.tail",
            instances,
        ))
    })?;
    let ram_read_write_rounds = ram_read_write.log_t + ram_read_write.log_k;
    let ram_raf_rounds = ram_raf.log_t + ram_raf.log_k - ram_raf.phase1_num_rounds;
    let ram_output_check_rounds =
        ram_output_check.log_t + ram_output_check.log_k - ram_output_check.phase1_num_rounds;
    let max_num_rounds = std::iter::once(ram_read_write_rounds)
        .chain(
            state
                .instances
                .iter()
                .map(SumcheckRegularBatchInstance::num_rounds),
        )
        .chain([ram_raf_rounds, ram_output_check_rounds])
        .max()
        .ok_or_else(|| invalid_sumcheck_output("Stage 2 regular batch has no instances"))?;
    let ram_raf_offset = max_num_rounds - ram_raf_rounds;
    let ram_output_check_offset = max_num_rounds - ram_output_check_rounds;

    append_sumcheck_claim(transcript, &ram_read_write.input_claim);
    for instance in &state.instances {
        append_sumcheck_claim(transcript, &instance.input_claim);
    }
    append_sumcheck_claim(transcript, &ram_raf.input_claim);
    append_sumcheck_claim(transcript, &F::zero());

    let tail_start = 1;
    let terminal_start = tail_start + state.instances.len();
    let ram_raf_index = terminal_start;
    let ram_output_check_index = terminal_start + 1;
    let instance_count = state.instances.len() + 3;
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let mut individual_claims = std::iter::once(ram_read_write.input_claim)
        .chain(state.instances.iter().map(|instance| {
            instance
                .input_claim
                .mul_pow_2(max_num_rounds - instance.num_rounds())
        }))
        .chain([
            ram_raf.input_claim.mul_pow_2(ram_raf_offset),
            F::zero().mul_pow_2(ram_output_check_offset),
        ])
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();
    let two_inv = F::from_u64(2).inv_or_zero();

    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);
    let mut ram_read_write_round_ms = 0.0;
    let mut tail_round_ms = 0.0;
    let mut ram_raf_round_ms = 0.0;
    let mut ram_output_check_round_ms = 0.0;
    let mut combine_round_ms = 0.0;
    let mut transcript_round_ms = 0.0;
    let mut bind_ram_read_write_ms = 0.0;
    let mut bind_tail_ms = 0.0;
    let mut bind_ram_raf_ms = 0.0;
    let mut bind_ram_output_check_ms = 0.0;
    for round in 0..max_num_rounds {
        let ram_poly = timed_stage2_accumulate(&mut ram_read_write_round_ms, || {
            backend.evaluate_sumcheck_ram_read_write_round(&ram_state, individual_claims[0])
        })?;
        let tail_messages = timed_stage2_accumulate(&mut tail_round_ms, || {
            backend.evaluate_sumcheck_regular_batch_round(
                &mut state,
                round,
                max_num_rounds,
                &individual_claims[tail_start..terminal_start],
            )
        })?
        .into_iter()
        .collect::<Vec<_>>();
        if tail_messages.len() != state.instances.len() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch round {round} returned {} instance messages, expected {}",
                tail_messages.len() + 1,
                state.instances.len() + 1
            )));
        }
        let mut univariate_polys = Vec::with_capacity(instance_count);
        univariate_polys.push(ram_poly);
        for (expected_index, round_message) in tail_messages.into_iter().enumerate() {
            if round_message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 2 regular batch round {round} returned instance {}, expected {expected_index}",
                    round_message.instance_index
                )));
            }
            univariate_polys.push(round_message.polynomial);
        }
        if round < ram_raf_offset {
            univariate_polys.push(UnivariatePoly::new(vec![
                individual_claims[ram_raf_index] * two_inv,
            ]));
        } else {
            univariate_polys.push(timed_stage2_accumulate(&mut ram_raf_round_ms, || {
                backend.evaluate_sumcheck_ram_raf_round(
                    &ram_raf_state,
                    individual_claims[ram_raf_index],
                )
            })?);
        }
        if round < ram_output_check_offset {
            univariate_polys.push(UnivariatePoly::new(vec![
                individual_claims[ram_output_check_index] * two_inv,
            ]));
        } else {
            univariate_polys.push(timed_stage2_accumulate(
                &mut ram_output_check_round_ms,
                || {
                    backend.evaluate_sumcheck_ram_output_check_round(
                        &ram_output_check_state,
                        individual_claims[ram_output_check_index],
                    )
                },
            )?);
        }

        let (batched_poly, challenge) = timed_stage2_accumulate(&mut combine_round_ms, || {
            let mut batched_poly = UnivariatePoly::zero();
            for (poly, coefficient) in univariate_polys.iter().zip(&batching_coefficients) {
                batched_poly += &(poly * *coefficient);
            }
            let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
            if round_sum != running_claim {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 2 regular batch round {round} sumcheck invariant failed"
                )));
            }
            let batched_poly = trim_round_polynomial(batched_poly);
            timed_stage2_accumulate(&mut transcript_round_ms, || {
                CompressedLabeledRoundPoly::sumcheck(&batched_poly)
                    .append_to_transcript(transcript);
            });
            let challenge =
                timed_stage2_accumulate(&mut transcript_round_ms, || transcript.challenge());
            Ok::<_, ProverError>((batched_poly, challenge))
        })?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);
        round_polynomials.push(batched_poly.compress());

        for (claim, poly) in individual_claims.iter_mut().zip(univariate_polys) {
            *claim = poly.evaluate(challenge);
        }
        timed_stage2_accumulate(&mut bind_ram_read_write_ms, || {
            backend.bind_sumcheck_ram_read_write_state(&mut ram_state, challenge)
        })?;
        timed_stage2_accumulate(&mut bind_tail_ms, || {
            backend.bind_sumcheck_regular_batch_state(&mut state, round, max_num_rounds, challenge)
        })?;
        if round >= ram_raf_offset {
            timed_stage2_accumulate(&mut bind_ram_raf_ms, || {
                backend.bind_sumcheck_ram_raf_state(&mut ram_raf_state, challenge)
            })?;
        }
        if round >= ram_output_check_offset {
            timed_stage2_accumulate(&mut bind_ram_output_check_ms, || {
                backend.bind_sumcheck_ram_output_check_state(&mut ram_output_check_state, challenge)
            })?;
        }
    }
    record_stage2_accumulated(
        "stage2.regular_batch.rounds.ram_read_write",
        ram_read_write_round_ms,
    );
    record_stage2_accumulated("stage2.regular_batch.rounds.tail", tail_round_ms);
    record_stage2_accumulated("stage2.regular_batch.rounds.ram_raf", ram_raf_round_ms);
    record_stage2_accumulated(
        "stage2.regular_batch.rounds.ram_output_check",
        ram_output_check_round_ms,
    );
    record_stage2_accumulated("stage2.regular_batch.rounds.combine", combine_round_ms);
    record_stage2_accumulated(
        "stage2.regular_batch.rounds.transcript",
        transcript_round_ms,
    );
    record_stage2_accumulated(
        "stage2.regular_batch.bind.ram_read_write",
        bind_ram_read_write_ms,
    );
    record_stage2_accumulated("stage2.regular_batch.bind.tail", bind_tail_ms);
    record_stage2_accumulated("stage2.regular_batch.bind.ram_raf", bind_ram_raf_ms);
    record_stage2_accumulated(
        "stage2.regular_batch.bind.ram_output_check",
        bind_ram_output_check_ms,
    );
    let [val, ra, inc] = timed_stage2("stage2.bound_outputs.ram_read_write", || {
        backend.output_sumcheck_ram_read_write_state(&ram_state)
    })?;
    let ram_raf_evaluation = timed_stage2("stage2.bound_outputs.ram_raf", || {
        backend.output_sumcheck_ram_raf_state(&ram_raf_state)
    })?;
    let ram_output_check = timed_stage2("stage2.bound_outputs.ram_output_check", || {
        backend.output_sumcheck_ram_output_check_state(&ram_output_check_state)
    })?;

    Ok(RegularBatchProof {
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        challenges,
        batching_coefficients,
        output_claim: running_claim,
        ram_read_write: RamReadWriteOutputOpeningClaims { val, ra, inc },
        ram_raf_evaluation,
        ram_output_check,
    })
}

#[cfg(feature = "zk")]
fn prove_regular_batch_sumcheck_committed<'a, F, T, B, VC>(
    ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    ram_raf: SumcheckRamRafStateRequest<F>,
    ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    backend: &mut B,
    vc_setup: &'a VC::Setup,
    transcript: &mut T,
) -> Result<PendingCommittedRegularBatch<'a, F, VC>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    VC: VectorCommitment<Field = F>,
{
    let mut ram_state = timed_stage2("stage2.regular_batch.materialize_ram_read_write", || {
        backend.materialize_sumcheck_ram_read_write_state(&ram_read_write)
    })?;
    let mut ram_raf_state = timed_stage2("stage2.regular_batch.materialize_ram_raf", || {
        backend.materialize_sumcheck_ram_raf_state(&ram_raf)
    })?;
    let mut ram_output_check_state =
        timed_stage2("stage2.regular_batch.materialize_ram_output_check", || {
            backend.materialize_sumcheck_ram_output_check_state(&ram_output_check)
        })?;
    let mut state = timed_stage2("stage2.regular_batch.materialize_tail", || {
        Ok::<_, ProverError>(SumcheckRegularBatchState::new(
            "stage2.regular_batch.tail",
            instances,
        ))
    })?;
    let ram_read_write_rounds = ram_read_write.log_t + ram_read_write.log_k;
    let ram_raf_rounds = ram_raf.log_t + ram_raf.log_k - ram_raf.phase1_num_rounds;
    let ram_output_check_rounds =
        ram_output_check.log_t + ram_output_check.log_k - ram_output_check.phase1_num_rounds;
    let max_num_rounds = std::iter::once(ram_read_write_rounds)
        .chain(
            state
                .instances
                .iter()
                .map(SumcheckRegularBatchInstance::num_rounds),
        )
        .chain([ram_raf_rounds, ram_output_check_rounds])
        .max()
        .ok_or_else(|| invalid_sumcheck_output("Stage 2 regular batch has no instances"))?;
    let ram_raf_offset = max_num_rounds - ram_raf_rounds;
    let ram_output_check_offset = max_num_rounds - ram_output_check_rounds;

    let tail_start = 1;
    let terminal_start = tail_start + state.instances.len();
    let ram_raf_index = terminal_start;
    let ram_output_check_index = terminal_start + 1;
    let instance_count = state.instances.len() + 3;
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let mut individual_claims = std::iter::once(ram_read_write.input_claim)
        .chain(state.instances.iter().map(|instance| {
            instance
                .input_claim
                .mul_pow_2(max_num_rounds - instance.num_rounds())
        }))
        .chain([
            ram_raf.input_claim.mul_pow_2(ram_raf_offset),
            F::zero().mul_pow_2(ram_output_check_offset),
        ])
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();
    let two_inv = F::from_u64(2).inv_or_zero();

    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, max_num_rounds)?;
    let mut ram_read_write_round_ms = 0.0;
    let mut tail_round_ms = 0.0;
    let mut ram_raf_round_ms = 0.0;
    let mut ram_output_check_round_ms = 0.0;
    let mut combine_round_ms = 0.0;
    let mut transcript_round_ms = 0.0;
    let mut bind_ram_read_write_ms = 0.0;
    let mut bind_tail_ms = 0.0;
    let mut bind_ram_raf_ms = 0.0;
    let mut bind_ram_output_check_ms = 0.0;
    for round in 0..max_num_rounds {
        let ram_poly = timed_stage2_accumulate(&mut ram_read_write_round_ms, || {
            backend.evaluate_sumcheck_ram_read_write_round(&ram_state, individual_claims[0])
        })?;
        let tail_messages = timed_stage2_accumulate(&mut tail_round_ms, || {
            backend.evaluate_sumcheck_regular_batch_round(
                &mut state,
                round,
                max_num_rounds,
                &individual_claims[tail_start..terminal_start],
            )
        })?
        .into_iter()
        .collect::<Vec<_>>();
        if tail_messages.len() != state.instances.len() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch round {round} returned {} instance messages, expected {}",
                tail_messages.len() + 1,
                state.instances.len() + 1
            )));
        }
        let mut univariate_polys = Vec::with_capacity(instance_count);
        univariate_polys.push(ram_poly);
        for (expected_index, round_message) in tail_messages.into_iter().enumerate() {
            if round_message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 2 regular batch round {round} returned instance {}, expected {expected_index}",
                    round_message.instance_index
                )));
            }
            univariate_polys.push(round_message.polynomial);
        }
        if round < ram_raf_offset {
            univariate_polys.push(UnivariatePoly::new(vec![
                individual_claims[ram_raf_index] * two_inv,
            ]));
        } else {
            univariate_polys.push(timed_stage2_accumulate(&mut ram_raf_round_ms, || {
                backend.evaluate_sumcheck_ram_raf_round(
                    &ram_raf_state,
                    individual_claims[ram_raf_index],
                )
            })?);
        }
        if round < ram_output_check_offset {
            univariate_polys.push(UnivariatePoly::new(vec![
                individual_claims[ram_output_check_index] * two_inv,
            ]));
        } else {
            univariate_polys.push(timed_stage2_accumulate(
                &mut ram_output_check_round_ms,
                || {
                    backend.evaluate_sumcheck_ram_output_check_round(
                        &ram_output_check_state,
                        individual_claims[ram_output_check_index],
                    )
                },
            )?);
        }

        let (batched_poly, challenge) = timed_stage2_accumulate(&mut combine_round_ms, || {
            let mut batched_poly = UnivariatePoly::zero();
            for (poly, coefficient) in univariate_polys.iter().zip(&batching_coefficients) {
                batched_poly += &(poly * *coefficient);
            }
            let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
            if round_sum != running_claim {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 2 committed regular batch round {round} sumcheck invariant failed"
                )));
            }

            let batched_poly = trim_round_polynomial(batched_poly);
            let challenge = timed_stage2_accumulate(&mut transcript_round_ms, || {
                builder.commit_round(&batched_poly, transcript)
            })?;
            Ok::<_, ProverError>((batched_poly, challenge))
        })?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);

        for (claim, poly) in individual_claims.iter_mut().zip(univariate_polys) {
            *claim = poly.evaluate(challenge);
        }
        timed_stage2_accumulate(&mut bind_ram_read_write_ms, || {
            backend.bind_sumcheck_ram_read_write_state(&mut ram_state, challenge)
        })?;
        timed_stage2_accumulate(&mut bind_tail_ms, || {
            backend.bind_sumcheck_regular_batch_state(&mut state, round, max_num_rounds, challenge)
        })?;
        if round >= ram_raf_offset {
            timed_stage2_accumulate(&mut bind_ram_raf_ms, || {
                backend.bind_sumcheck_ram_raf_state(&mut ram_raf_state, challenge)
            })?;
        }
        if round >= ram_output_check_offset {
            timed_stage2_accumulate(&mut bind_ram_output_check_ms, || {
                backend.bind_sumcheck_ram_output_check_state(&mut ram_output_check_state, challenge)
            })?;
        }
    }
    record_stage2_accumulated(
        "stage2.regular_batch.rounds.ram_read_write",
        ram_read_write_round_ms,
    );
    record_stage2_accumulated("stage2.regular_batch.rounds.tail", tail_round_ms);
    record_stage2_accumulated("stage2.regular_batch.rounds.ram_raf", ram_raf_round_ms);
    record_stage2_accumulated(
        "stage2.regular_batch.rounds.ram_output_check",
        ram_output_check_round_ms,
    );
    record_stage2_accumulated("stage2.regular_batch.rounds.combine", combine_round_ms);
    record_stage2_accumulated(
        "stage2.regular_batch.rounds.transcript",
        transcript_round_ms,
    );
    record_stage2_accumulated(
        "stage2.regular_batch.bind.ram_read_write",
        bind_ram_read_write_ms,
    );
    record_stage2_accumulated("stage2.regular_batch.bind.tail", bind_tail_ms);
    record_stage2_accumulated("stage2.regular_batch.bind.ram_raf", bind_ram_raf_ms);
    record_stage2_accumulated(
        "stage2.regular_batch.bind.ram_output_check",
        bind_ram_output_check_ms,
    );
    let [val, ra, inc] = timed_stage2("stage2.bound_outputs.ram_read_write", || {
        backend.output_sumcheck_ram_read_write_state(&ram_state)
    })?;
    let ram_raf_evaluation = timed_stage2("stage2.bound_outputs.ram_raf", || {
        backend.output_sumcheck_ram_raf_state(&ram_raf_state)
    })?;
    let ram_output_check = timed_stage2("stage2.bound_outputs.ram_output_check", || {
        backend.output_sumcheck_ram_output_check_state(&ram_output_check_state)
    })?;

    Ok(PendingCommittedRegularBatch {
        builder,
        challenges,
        batching_coefficients,
        output_claim: running_claim,
        ram_read_write: RamReadWriteOutputOpeningClaims { val, ra, inc },
        ram_raf_evaluation,
        ram_output_check,
    })
}

fn trim_round_polynomial<F: Field>(poly: UnivariatePoly<F>) -> UnivariatePoly<F> {
    let mut coefficients = poly.into_coefficients();
    while coefficients.len() > 2 && coefficients.last().is_some_and(|value| *value == F::zero()) {
        let _ = coefficients.pop();
    }
    UnivariatePoly::new(coefficients)
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
pub fn prove_stage2_regular_batch_sumcheck_for_frontier<F, W, B, T, C>(
    input: &Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage2RegularBatchFrontierProof<F, C>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let stage2_rows = input.witness.stage2_rows()?;
    let initial_ram_state = input.witness.initial_ram_state_words()?;
    let final_ram_state = input.witness.final_ram_state_words()?;
    let instances =
        build_regular_batch_instances(input.config, &stage2_rows, product_uniskip, prefix)?;
    let backend_rows = stage2_backend_rows(&stage2_rows);
    let batch = prove_regular_batch_sumcheck::<F, T, C, B>(
        build_ram_read_write_state_request(
            input.config,
            backend_rows.clone(),
            &initial_ram_state,
            product_uniskip,
            prefix,
        )?,
        build_ram_raf_state_request(
            input.config,
            input.checked,
            input.stage1,
            backend_rows,
            product_uniskip,
        )?,
        build_ram_output_check_state_request(
            input.config,
            input.checked,
            &final_ram_state,
            prefix,
        )?,
        instances,
        backend,
        transcript,
    )?;
    Ok(Stage2RegularBatchFrontierProof {
        proof: batch.proof,
        challenges: batch.challenges,
        batching_coefficients: batch.batching_coefficients,
        output_claim: batch.output_claim,
    })
}

#[cfg(not(feature = "field-inline"))]
fn build_regular_batch_instances<F, C>(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
{
    validate_stage2_rows_for_regular_batch(config, rows)?;
    let row_count = 1usize << config.log_t;
    let final_cycle = row_count - 1;

    let product_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        product_uniskip.challenge,
    )
    .map_err(invalid_sumcheck_output)?;
    if product_weights.len() != 3 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product remainder expected 3 weights, got {}",
            product_weights.len()
        )));
    }
    let tau_scale = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        product_uniskip.tau_high,
        product_uniskip.challenge,
    )
    .map_err(invalid_sumcheck_output)?;
    let tau_eq_by_cycle = EqPolynomial::<F>::evals(&product_uniskip.tau_low, None);

    let instruction_gamma = prefix.instruction_gamma;
    let instruction_gamma2 = instruction_gamma * instruction_gamma;
    let instruction_gamma3 = instruction_gamma2 * instruction_gamma;
    let instruction_gamma4 = instruction_gamma3 * instruction_gamma;

    let mut product_tau_eq = unsafe_allocate_zero_vec(row_count);
    let mut product_left = unsafe_allocate_zero_vec(row_count);
    let mut product_right = unsafe_allocate_zero_vec(row_count);
    let mut instruction_eq = unsafe_allocate_zero_vec(row_count);
    let mut instruction_reduced = unsafe_allocate_zero_vec(row_count);

    product_tau_eq
        .par_iter_mut()
        .zip(product_left.par_iter_mut())
        .zip(product_right.par_iter_mut())
        .zip(instruction_eq.par_iter_mut())
        .zip(instruction_reduced.par_iter_mut())
        .enumerate()
        .for_each(
            |(
                index,
                (
                    (((product_tau_eq, product_left), product_right), instruction_eq),
                    instruction_reduced,
                ),
            )| {
                let cycle = bit_reverse(index, config.log_t);
                let row = &rows[cycle];
                let tau_eq = tau_eq_by_cycle[cycle];
                *product_tau_eq = tau_eq;
                *instruction_eq = tau_eq;
                *product_left = product_weights[0].mul_u64(row.left_instruction_input)
                    + product_weights[1].mul_u64(row.lookup_output)
                    + bool_term(product_weights[2], row.jump_flag);
                *product_right = product_weights[0].mul_i128(row.right_instruction_input)
                    + bool_term(product_weights[1], row.branch_flag)
                    + bool_term(
                        product_weights[2],
                        cycle != final_cycle && !row.next_is_noop,
                    );
                *instruction_reduced = F::one().mul_u64(row.lookup_output)
                    + instruction_gamma.mul_u64(row.left_lookup_operand)
                    + instruction_gamma2.mul_u128(row.right_lookup_operand)
                    + instruction_gamma3.mul_u64(row.left_instruction_input)
                    + instruction_gamma4.mul_i128(row.right_instruction_input);
            },
        );

    Ok(vec![
        regular_batch_instance(
            "product remainder",
            product_uniskip.output_claim,
            tau_scale,
            vec![
                Polynomial::new(product_tau_eq),
                Polynomial::new(product_left),
                Polynomial::new(product_right),
            ],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
                regular_batch_factor(vec![regular_batch_term(2, F::one())]),
            ],
        ),
        regular_batch_instance(
            "instruction claim-reduction",
            prefix.input_claims.instruction_claim_reduction,
            F::one(),
            vec![
                Polynomial::new(instruction_eq),
                Polynomial::new(instruction_reduced),
            ],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
            ],
        ),
    ])
}

#[cfg(feature = "field-inline")]
fn build_regular_batch_instances<F, C>(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
{
    validate_stage2_rows_for_regular_batch(config, rows)?;
    let row_count = 1usize << config.log_t;
    let final_cycle = row_count - 1;

    let product_weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        product_uniskip.challenge,
    )
    .map_err(invalid_sumcheck_output)?;
    if product_weights.len() != 5 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product remainder expected 5 weights, got {}",
            product_weights.len()
        )));
    }
    let tau_scale = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        product_uniskip.tau_high,
        product_uniskip.challenge,
    )
    .map_err(invalid_sumcheck_output)?;
    let tau_eq_by_cycle = EqPolynomial::<F>::evals(&product_uniskip.tau_low, None);

    let instruction_gamma = prefix.instruction_gamma;
    let instruction_gamma2 = instruction_gamma * instruction_gamma;
    let instruction_gamma3 = instruction_gamma2 * instruction_gamma;
    let instruction_gamma4 = instruction_gamma3 * instruction_gamma;

    let field_gamma = prefix.field_registers_claim_reduction_gamma;
    let field_gamma2 = field_gamma * field_gamma;

    let mut product_tau_eq = unsafe_allocate_zero_vec(row_count);
    let mut product_left = unsafe_allocate_zero_vec(row_count);
    let mut product_right = unsafe_allocate_zero_vec(row_count);
    let mut instruction_eq = unsafe_allocate_zero_vec(row_count);
    let mut instruction_reduced = unsafe_allocate_zero_vec(row_count);
    let mut field_eq = unsafe_allocate_zero_vec(row_count);
    let mut field_reduced = unsafe_allocate_zero_vec(row_count);

    product_tau_eq
        .par_iter_mut()
        .zip(product_left.par_iter_mut())
        .zip(product_right.par_iter_mut())
        .zip(instruction_eq.par_iter_mut())
        .zip(instruction_reduced.par_iter_mut())
        .zip(field_eq.par_iter_mut())
        .zip(field_reduced.par_iter_mut())
        .enumerate()
        .for_each(
            |(
                index,
                (
                    (
                        (
                            (((product_tau_eq, product_left), product_right), instruction_eq),
                            instruction_reduced,
                        ),
                        field_eq,
                    ),
                    field_reduced,
                ),
            )| {
                let cycle = bit_reverse(index, config.log_t);
                let row = &rows[cycle];
                let tau_eq = tau_eq_by_cycle[cycle];
                let field_rs1 = field_factors.rs1[cycle];
                let field_rs2 = field_factors.rs2[cycle];
                let field_rd = field_factors.rd[cycle];
                *product_tau_eq = tau_eq;
                *instruction_eq = tau_eq;
                *field_eq = tau_eq;
                *product_left = product_weights[0].mul_u64(row.left_instruction_input)
                    + product_weights[1].mul_u64(row.lookup_output)
                    + bool_term(product_weights[2], row.jump_flag)
                    + (product_weights[3] + product_weights[4]) * field_rs1;
                *product_right = product_weights[0].mul_i128(row.right_instruction_input)
                    + bool_term(product_weights[1], row.branch_flag)
                    + bool_term(
                        product_weights[2],
                        cycle != final_cycle && !row.next_is_noop,
                    )
                    + product_weights[3] * field_rs2
                    + product_weights[4] * field_rd;
                *instruction_reduced = F::one().mul_u64(row.lookup_output)
                    + instruction_gamma.mul_u64(row.left_lookup_operand)
                    + instruction_gamma2.mul_u128(row.right_lookup_operand)
                    + instruction_gamma3.mul_u64(row.left_instruction_input)
                    + instruction_gamma4.mul_i128(row.right_instruction_input);
                *field_reduced = field_rd + field_gamma * field_rs1 + field_gamma2 * field_rs2;
            },
        );

    Ok(vec![
        regular_batch_instance(
            "product remainder",
            product_uniskip.output_claim,
            tau_scale,
            vec![
                Polynomial::new(product_tau_eq),
                Polynomial::new(product_left),
                Polynomial::new(product_right),
            ],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
                regular_batch_factor(vec![regular_batch_term(2, F::one())]),
            ],
        ),
        regular_batch_instance(
            "instruction claim-reduction",
            prefix.input_claims.instruction_claim_reduction,
            F::one(),
            vec![
                Polynomial::new(instruction_eq),
                Polynomial::new(instruction_reduced),
            ],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
            ],
        ),
        regular_batch_instance(
            "field-registers claim-reduction",
            prefix.input_claims.field_registers_claim_reduction,
            F::one(),
            vec![Polynomial::new(field_eq), Polynomial::new(field_reduced)],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
            ],
        ),
    ])
}

#[cfg(feature = "field-inline")]
struct Stage2FieldInlineMaterializedFactors<F: Field> {
    rs1: Vec<F>,
    rs2: Vec<F>,
    rd: Vec<F>,
}

#[cfg(feature = "field-inline")]
impl<F: Field> Stage2FieldInlineMaterializedFactors<F> {
    fn new<W, B>(
        config: Stage2BatchProverConfig,
        witness: &W,
        backend: &mut B,
    ) -> Result<Self, ProverError>
    where
        W: WitnessProvider<F, FieldInlineNamespace>,
        B: SumcheckBackend<F, FieldInlineNamespace>,
    {
        let trace_len = 1usize << config.log_t;
        Ok(Self {
            rs1: materialize_field_inline_oracle(
                witness,
                backend,
                FieldInlineVirtualPolynomial::FieldRs1Value,
                trace_len,
            )?,
            rs2: materialize_field_inline_oracle(
                witness,
                backend,
                FieldInlineVirtualPolynomial::FieldRs2Value,
                trace_len,
            )?,
            rd: materialize_field_inline_oracle(
                witness,
                backend,
                FieldInlineVirtualPolynomial::FieldRdValue,
                trace_len,
            )?,
        })
    }
}

#[cfg(feature = "field-inline")]
fn materialize_field_inline_oracle<F, W, B>(
    witness: &W,
    backend: &mut B,
    variable: FieldInlineVirtualPolynomial,
    expected_len: usize,
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, FieldInlineNamespace>,
{
    materialize_oracle(
        witness,
        backend,
        OracleRef::virtual_polynomial(variable),
        expected_len,
        "field-inline virtual",
    )
}

#[cfg(feature = "field-inline")]
fn materialize_oracle<F, W, B, N>(
    witness: &W,
    backend: &mut B,
    oracle: OracleRef<N>,
    expected_len: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
    B: SumcheckBackend<F, N>,
{
    let requirement = primary_view_requirement(witness, oracle)?;
    let request = SumcheckMaterializationRequest::new(
        "stage2.regular_batch.factor_materialization",
        vec![SumcheckViewMaterializationRequest::new(
            BackendValueSlot(0),
            requirement,
        )],
    );
    let mut outputs = backend.materialize_sumcheck_views(&request, witness)?;
    let values = take_single_materialization(&mut outputs, label)?;
    if values.len() != expected_len {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 {label} materialized {} rows, expected {expected_len}",
            values.len()
        )));
    }
    Ok(values)
}

#[cfg(feature = "field-inline")]
fn take_single_materialization<F: Field>(
    outputs: &mut Vec<SumcheckMaterializationOutput<F>>,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    if outputs.len() != 1 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 {label} materialization returned {} outputs, expected 1",
            outputs.len()
        )));
    }
    let output = outputs.pop().ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 2 {label} materialization returned no output"
        ))
    })?;
    if output.slot != BackendValueSlot(0) {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 {label} materialization used unexpected slot {:?}",
            output.slot
        )));
    }
    Ok(output.values)
}

fn regular_batch_term<F: Field>(
    polynomial: usize,
    coefficient: F,
) -> SumcheckRegularBatchLinearTerm<F> {
    SumcheckRegularBatchLinearTerm::new(polynomial, coefficient)
}

fn regular_batch_factor<F: Field>(
    terms: Vec<SumcheckRegularBatchLinearTerm<F>>,
) -> SumcheckRegularBatchLinearFactor<F> {
    SumcheckRegularBatchLinearFactor::from_terms(terms)
}

fn regular_batch_instance<F: Field>(
    label: &'static str,
    input_claim: F,
    scale: F,
    polynomials: Vec<Polynomial<F>>,
    factors: Vec<SumcheckRegularBatchLinearFactor<F>>,
) -> SumcheckRegularBatchInstance<F> {
    SumcheckRegularBatchInstance::new(label, input_claim, scale, polynomials, factors)
}

fn stage2_backend_rows(rows: &[JoltVmStage2TraceRow]) -> Vec<SumcheckRamReadWriteRow> {
    rows.iter()
        .map(|row| SumcheckRamReadWriteRow {
            remapped_ram_address: row.remapped_ram_address,
            ram_read_value: row.ram_read_value,
            ram_write_value: row.ram_write_value,
            ram_increment: row.ram_increment,
        })
        .collect()
}

fn validate_stage2_rows_for_regular_batch(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
) -> Result<(), ProverError> {
    let expected_rows = 1usize << config.log_t;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 regular-batch row witness returned {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    Ok(())
}

#[inline]
fn bool_term<F: Field>(coefficient: F, value: bool) -> F {
    if value {
        coefficient
    } else {
        F::zero()
    }
}

fn build_ram_read_write_state_request<F: Field>(
    config: Stage2BatchProverConfig,
    rows: Vec<SumcheckRamReadWriteRow>,
    initial_ram_state: &[u64],
    product_uniskip: &Stage2ProductUniSkipOutput<F, impl Sized>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
) -> Result<SumcheckRamReadWriteStateRequest<F>, ProverError> {
    let expected_rows = 1usize << config.log_t;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 RAM read-write row witness returned {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    Ok(SumcheckRamReadWriteStateRequest::new(
        "stage2.ram_read_write.state",
        rows,
        initial_ram_state.to_vec(),
        product_uniskip.tau_low.clone(),
        prefix.ram_read_write_gamma,
        prefix.input_claims.ram_read_write,
        config.log_t,
        config.log_k,
        config.rw_config.ram_rw_phase1_num_rounds as usize,
        config.rw_config.ram_rw_phase2_num_rounds as usize,
    )
    .with_relation(STAGE2_RAM_READ_WRITE_RELATION)
    .with_optimization_ids(STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS))
}

fn build_ram_raf_state_request<F: Field>(
    config: Stage2BatchProverConfig,
    checked: &jolt_verifier::CheckedInputs,
    stage1: &Stage1ClearOutput<F>,
    rows: Vec<SumcheckRamReadWriteRow>,
    product_uniskip: &Stage2ProductUniSkipOutput<F, impl Sized>,
) -> Result<SumcheckRamRafStateRequest<F>, ProverError> {
    let read_write_dimensions = config.rw_config.ram_dimensions(config.log_t, config.log_k);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: format!("invalid Stage 2 RAM RAF dimensions: {error}"),
            }
        })?;
    let input_claim = F::pow2(raf_dimensions.phase3_cycle_rounds()) * stage1.outer.ram_address;

    Ok(SumcheckRamRafStateRequest::new(
        "stage2.ram_raf.state",
        rows,
        product_uniskip.tau_low.clone(),
        input_claim,
        checked.public_io.memory_layout.get_lowest_address(),
        config.log_t,
        config.log_k,
        config.rw_config.ram_rw_phase1_num_rounds as usize,
        config.rw_config.ram_rw_phase2_num_rounds as usize,
    )
    .with_relation(STAGE2_RAM_RAF_RELATION)
    .with_optimization_ids(STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS))
}

fn build_ram_output_check_state_request<F: Field>(
    config: Stage2BatchProverConfig,
    checked: &jolt_verifier::CheckedInputs,
    final_ram_state: &[u64],
    prefix: &Stage2RegularBatchPrefixOutput<F>,
) -> Result<SumcheckRamOutputCheckStateRequest<F>, ProverError> {
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        ProverError::InvalidStageRequest {
            reason: format!("invalid public IO memory for Stage 2 output check: {error}"),
        }
    })?;
    let io_start = usize::try_from(public_memory.io_mask_start).map_err(|_| {
        ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 output-check IO start {} does not fit usize",
                public_memory.io_mask_start
            ),
        }
    })?;
    let io_end = usize::try_from(public_memory.io_mask_end).map_err(|_| {
        ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 output-check IO end {} does not fit usize",
                public_memory.io_mask_end
            ),
        }
    })?;
    let ram_len = 1usize << config.log_k;
    let mut public_io_state = vec![0_u64; ram_len];
    for segment in &public_memory.segments {
        let end = segment.start_index + segment.words.len();
        if end > ram_len {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 2 output-check public IO segment {}..{end} exceeds {ram_len} RAM words",
                    segment.start_index
                ),
            });
        }
        public_io_state[segment.start_index..end].copy_from_slice(&segment.words);
    }

    Ok(SumcheckRamOutputCheckStateRequest::new(
        "stage2.ram_output_check.state",
        final_ram_state.to_vec(),
        public_io_state,
        io_start,
        io_end,
        prefix.output_address_challenges.clone(),
        config.log_t,
        config.log_k,
        config.rw_config.ram_rw_phase1_num_rounds as usize,
        config.rw_config.ram_rw_phase2_num_rounds as usize,
    )
    .with_relation(STAGE2_RAM_OUTPUT_CHECK_RELATION)
    .with_optimization_ids(STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS))
}

fn expected_regular_batch_outputs<F: Field>(
    config: Stage2BatchProverConfig,
    checked: &jolt_verifier::CheckedInputs,
    product_uniskip: &Stage2ProductUniSkipOutput<F, impl Sized>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
    batching_coefficients: &[F],
    opening_points: &Stage2OpeningPoints<F>,
    claims: &Stage2BatchOutputOpeningClaims<F>,
) -> Result<ExpectedRegularBatchOutputs<F>, ProverError> {
    let eq_cycle = try_eq_mle(
        &product_uniskip.tau_low,
        &opening_points.ram_read_write_opening.r_cycle,
    )
    .map_err(invalid_sumcheck_output)?;
    let ram_read_write = eq_cycle
        * claims.ram_read_write.ra
        * (claims.ram_read_write.val
            + prefix.ram_read_write_gamma
                * (claims.ram_read_write.val + claims.ram_read_write.inc));

    let weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        product_uniskip.challenge,
    )
    .map_err(invalid_sumcheck_output)?;
    let product_tau_high_bound = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        product_uniskip.tau_high,
        product_uniskip.challenge,
    )
    .map_err(invalid_sumcheck_output)?;
    let product_tau_low_eq = try_eq_mle(&product_uniskip.tau_low, &opening_points.product_opening)
        .map_err(invalid_sumcheck_output)?;
    let product_left = weights[0] * claims.product_remainder.left_instruction_input
        + weights[1] * claims.product_remainder.lookup_output
        + weights[2] * claims.product_remainder.jump_flag;
    let product_right = weights[0] * claims.product_remainder.right_instruction_input
        + weights[1] * claims.product_remainder.branch_flag
        + weights[2] * (F::one() - claims.product_remainder.next_is_noop);
    #[cfg(feature = "field-inline")]
    let (product_left, product_right) = {
        let mut product_left = product_left;
        let mut product_right = product_right;
        product_left += (weights[3] + weights[4]) * claims.field_inline.product.field_rs1_value;
        product_right += weights[3] * claims.field_inline.product.field_rs2_value
            + weights[4] * claims.field_inline.product.field_rd_value;
        (product_left, product_right)
    };
    let product_remainder =
        product_tau_high_bound * product_tau_low_eq * product_left * product_right;

    let eq_spartan = try_eq_mle(
        &opening_points.instruction_opening,
        &product_uniskip.tau_low,
    )
    .map_err(invalid_sumcheck_output)?;
    let gamma = prefix.instruction_gamma;
    let gamma2 = gamma * gamma;
    let gamma3 = gamma2 * gamma;
    let gamma4 = gamma3 * gamma;
    let instruction_claim_reduction = eq_spartan
        * (claims
            .instruction_claim_reduction
            .lookup_output
            .unwrap_or(claims.product_remainder.lookup_output)
            + gamma * claims.instruction_claim_reduction.left_lookup_operand
            + gamma2 * claims.instruction_claim_reduction.right_lookup_operand
            + gamma3
                * claims
                    .instruction_claim_reduction
                    .left_instruction_input
                    .unwrap_or(claims.product_remainder.left_instruction_input)
            + gamma4
                * claims
                    .instruction_claim_reduction
                    .right_instruction_input
                    .unwrap_or(claims.product_remainder.right_instruction_input));

    #[cfg(feature = "field-inline")]
    let field_registers_claim_reduction = {
        let eq_spartan = try_eq_mle(
            &opening_points.field_registers_claim_reduction_opening,
            &product_uniskip.tau_low,
        )
        .map_err(invalid_sumcheck_output)?;
        let gamma = prefix.field_registers_claim_reduction_gamma;
        eq_spartan
            * (claims.field_inline.product.field_rd_value
                + gamma * claims.field_inline.product.field_rs1_value
                + gamma * gamma * claims.field_inline.product.field_rs2_value)
    };

    let ram_raf_unmap_address = IdentityPolynomial::new(config.log_k)
        .evaluate(&opening_points.ram_output_check_opening)
        * F::from_u64(8)
        + F::from_u64(checked.public_io.memory_layout.get_lowest_address());
    let ram_raf_evaluation = ram_raf_unmap_address * claims.ram_raf_evaluation;

    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        ProverError::InvalidStageRequest {
            reason: format!("invalid public IO memory for Stage 2 output check: {error}"),
        }
    })?;
    let output_eq = try_eq_mle(
        &prefix.output_address_challenges,
        &opening_points.ram_output_check_opening,
    )
    .map_err(invalid_sumcheck_output)?;
    let output_mask = range_mask_mle_msb(
        public_memory.io_mask_start,
        public_memory.io_mask_end,
        &opening_points.ram_output_check_opening,
    )
    .map_err(invalid_sumcheck_output)?;
    let io_num_vars = public_memory.io_num_vars();
    if opening_points.ram_output_check_opening.len() < io_num_vars {
        return Err(invalid_sumcheck_output(format!(
            "RAM output address point has {} variables, IO needs {io_num_vars}",
            opening_points.ram_output_check_opening.len()
        )));
    }
    let (r_hi, r_lo) = opening_points
        .ram_output_check_opening
        .split_at(opening_points.ram_output_check_opening.len() - io_num_vars);
    let hi_scale = r_hi
        .iter()
        .fold(F::one(), |acc, challenge| acc * (F::one() - *challenge));
    let val_io = hi_scale
        * sparse_segments_mle_msb(
            public_memory
                .segments
                .iter()
                .map(|segment| (segment.start_index, segment.words.as_slice())),
            r_lo,
        );
    let eq_io_mask = output_eq * output_mask;
    let ram_output_check = eq_io_mask * claims.ram_output_check - eq_io_mask * val_io;

    #[cfg(not(feature = "field-inline"))]
    let final_claim = {
        let [ram_read_write_coeff, product_coeff, instruction_coeff, ram_raf_coeff, ram_output_coeff] =
            batching_coefficients
        else {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch has {} coefficients, expected 5",
                batching_coefficients.len()
            )));
        };
        *ram_read_write_coeff * ram_read_write
            + *product_coeff * product_remainder
            + *instruction_coeff * instruction_claim_reduction
            + *ram_raf_coeff * ram_raf_evaluation
            + *ram_output_coeff * ram_output_check
    };
    #[cfg(feature = "field-inline")]
    let final_claim = {
        let [ram_read_write_coeff, product_coeff, instruction_coeff, field_registers_coeff, ram_raf_coeff, ram_output_coeff] =
            batching_coefficients
        else {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch has {} coefficients, expected 6",
                batching_coefficients.len()
            )));
        };
        *ram_read_write_coeff * ram_read_write
            + *product_coeff * product_remainder
            + *instruction_coeff * instruction_claim_reduction
            + *field_registers_coeff * field_registers_claim_reduction
            + *ram_raf_coeff * ram_raf_evaluation
            + *ram_output_coeff * ram_output_check
    };

    Ok(ExpectedRegularBatchOutputs {
        ram_read_write,
        product_remainder,
        instruction_claim_reduction,
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction,
        ram_raf_evaluation,
        ram_output_check,
        final_claim,
    })
}

fn append_stage2_opening_claims<F: Field>(
    transcript: &mut impl Transcript<Challenge = F>,
    claims: &Stage2BatchOutputOpeningClaims<F>,
) {
    transcript.append_labeled(b"opening_claim", &claims.ram_read_write.val);
    transcript.append_labeled(b"opening_claim", &claims.ram_read_write.ra);
    transcript.append_labeled(b"opening_claim", &claims.ram_read_write.inc);
    transcript.append_labeled(
        b"opening_claim",
        &claims.product_remainder.left_instruction_input,
    );
    transcript.append_labeled(
        b"opening_claim",
        &claims.product_remainder.right_instruction_input,
    );
    transcript.append_labeled(b"opening_claim", &claims.product_remainder.jump_flag);
    transcript.append_labeled(
        b"opening_claim",
        &claims.product_remainder.write_lookup_output_to_rd,
    );
    transcript.append_labeled(b"opening_claim", &claims.product_remainder.lookup_output);
    transcript.append_labeled(b"opening_claim", &claims.product_remainder.branch_flag);
    transcript.append_labeled(b"opening_claim", &claims.product_remainder.next_is_noop);
    transcript.append_labeled(
        b"opening_claim",
        &claims.product_remainder.virtual_instruction,
    );
    #[cfg(feature = "field-inline")]
    {
        transcript.append_labeled(
            b"opening_claim",
            &claims.field_inline.product.field_rs1_value,
        );
        transcript.append_labeled(
            b"opening_claim",
            &claims.field_inline.product.field_rs2_value,
        );
        transcript.append_labeled(
            b"opening_claim",
            &claims.field_inline.product.field_rd_value,
        );
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_claim_reduction.left_lookup_operand,
    );
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_claim_reduction.right_lookup_operand,
    );
    transcript.append_labeled(b"opening_claim", &claims.ram_raf_evaluation);
    transcript.append_labeled(b"opening_claim", &claims.ram_output_check);
}

#[cfg(feature = "zk")]
fn stage2_committed_output_claim_values<F: Field>(
    claims: &Stage2BatchOutputOpeningClaims<F>,
) -> Vec<F> {
    let mut values = vec![
        claims.ram_read_write.val,
        claims.ram_read_write.ra,
        claims.ram_read_write.inc,
        claims.product_remainder.left_instruction_input,
        claims.product_remainder.right_instruction_input,
        claims.product_remainder.jump_flag,
        claims.product_remainder.write_lookup_output_to_rd,
        claims.product_remainder.lookup_output,
        claims.product_remainder.branch_flag,
        claims.product_remainder.next_is_noop,
        claims.product_remainder.virtual_instruction,
    ];
    #[cfg(feature = "field-inline")]
    values.extend([
        claims.field_inline.product.field_rs1_value,
        claims.field_inline.product.field_rs2_value,
        claims.field_inline.product.field_rd_value,
    ]);
    values.extend([
        claims.instruction_claim_reduction.left_lookup_operand,
        claims.instruction_claim_reduction.right_lookup_operand,
        claims.ram_raf_evaluation,
        claims.ram_output_check,
    ]);
    values
}

fn bit_reverse(index: usize, bits: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - bits)
}

#[cfg(not(feature = "field-inline"))]
fn product_uniskip_extended_evals<F, W, B>(
    config: Stage2ProverConfig,
    witness: &W,
    backend: &mut B,
    tau_low: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let rows = witness
        .product_uniskip_rows()?
        .into_iter()
        .map(|row| {
            SumcheckProductUniskipRow::new(
                row.left_instruction,
                row.lookup_output,
                row.jump_flag,
                row.right_instruction,
                row.branch_flag,
                row.next_is_noop,
            )
        })
        .collect::<Vec<_>>();
    product_uniskip_extended_evals_from_rows(config, &rows, backend, tau_low)
}

#[cfg(not(feature = "field-inline"))]
fn product_uniskip_extended_evals_from_rows<F, B>(
    config: Stage2ProverConfig,
    rows: &[SumcheckProductUniskipRow],
    backend: &mut B,
    tau_low: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let expected_rows = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        ProverError::InvalidStageRequest {
            reason: format!("Stage 2 trace length overflows for log_t={}", config.log_t),
        }
    })?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 product uni-skip witness returned {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    let queries = uniskip_targets(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
    )?
    .into_iter()
    .enumerate()
    .map(|(index, target)| {
        let row_weights =
            centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, F::from_i64(target))
                .map_err(invalid_sumcheck_output)?;
        Ok(SumcheckRowProductQuery::new(
            value_slot(index)?,
            tau_low.to_vec(),
            row_weights,
            F::one(),
        ))
    })
    .collect::<Result<Vec<_>, ProverError>>()?;
    let request =
        SumcheckProductUniskipRequest::new("stage2.product_uniskip.extended_evals", &rows, queries)
            .with_relation(SPARTAN_PRODUCT_UNISKIP_RELATION)
            .with_optimization_ids(STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS);
    let outputs = backend.evaluate_sumcheck_product_uniskip_rows(&request)?;
    ordered_row_product_outputs(outputs, PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT)
}

#[cfg(not(feature = "field-inline"))]
fn product_uniskip_rows_from_stage2_rows(
    config: &Stage2ProverConfig,
    rows: &[JoltVmStage2TraceRow],
) -> Result<Vec<SumcheckProductUniskipRow>, ProverError> {
    let expected_rows = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        ProverError::InvalidStageRequest {
            reason: format!("Stage 2 trace length overflows for log_t={}", config.log_t),
        }
    })?;
    if rows.len() != expected_rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 product uni-skip stage rows have {} rows, expected {expected_rows}",
                rows.len()
            ),
        });
    }
    Ok(rows
        .iter()
        .map(|row| {
            SumcheckProductUniskipRow::new(
                row.left_instruction_input,
                row.lookup_output,
                row.jump_flag,
                row.right_instruction_input,
                row.branch_flag,
                row.next_is_noop,
            )
        })
        .collect())
}

#[cfg(feature = "field-inline")]
fn product_uniskip_extended_evals_field_inline<F, W, FI, B>(
    config: Stage2ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    tau_low: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: JoltVmProductUniskipRows + WitnessProvider<F, JoltVmNamespace>,
    FI: WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + SumcheckBackend<F, FieldInlineNamespace>,
{
    let rows = 1usize << config.log_t;
    let base_rows = witness.product_uniskip_rows()?;
    if base_rows.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 field-inline product uni-skip witness returned {} rows, expected {rows}",
                base_rows.len()
            ),
        });
    }
    let field_rs1_value = materialize_field_inline_oracle(
        field_inline_witness,
        backend,
        FieldInlineVirtualPolynomial::FieldRs1Value,
        rows,
    )?;
    let field_rs2_value = materialize_field_inline_oracle(
        field_inline_witness,
        backend,
        FieldInlineVirtualPolynomial::FieldRs2Value,
        rows,
    )?;
    let field_rd_value = materialize_field_inline_oracle(
        field_inline_witness,
        backend,
        FieldInlineVirtualPolynomial::FieldRdValue,
        rows,
    )?;
    let weights = uniskip_targets(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
    )?
    .into_iter()
    .map(|target| {
        centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, F::from_i64(target))
            .map_err(invalid_sumcheck_output)
    })
    .collect::<Result<Vec<_>, ProverError>>()?;

    let eq_table = TensorEqTable::new(tau_low);
    if eq_table.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 field-inline product uni-skip eq table has {} rows, expected {rows}",
                eq_table.len()
            ),
        });
    }

    Ok(eq_table.par_fold_out_in(
        || vec![F::zero(); weights.len()],
        |inner, cycle, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            let row = base_rows[cycle];
            let left_instruction_input = F::from_u64(row.left_instruction);
            let right_instruction_input = F::from_i128(row.right_instruction);
            let lookup_output = F::from_u64(row.lookup_output);
            let jump_flag = F::from_bool(row.jump_flag);
            let branch_flag = F::from_bool(row.branch_flag);
            let not_next_is_noop = F::from_bool(!row.next_is_noop);
            let field_rs1 = field_rs1_value[cycle];
            let field_rs2 = field_rs2_value[cycle];
            let field_rd = field_rd_value[cycle];

            for (total, weights) in inner.iter_mut().zip(&weights) {
                let left = weights[0] * left_instruction_input
                    + weights[1] * lookup_output
                    + weights[2] * jump_flag
                    + (weights[3] + weights[4]) * field_rs1;
                let right = weights[0] * right_instruction_input
                    + weights[1] * branch_flag
                    + weights[2] * not_next_is_noop
                    + weights[3] * field_rs2
                    + weights[4] * field_rd;
                *total += e_in * left * right;
            }
        },
        |_x_out, e_out, mut inner| {
            if e_out.is_zero() {
                inner.fill(F::zero());
            } else {
                for value in &mut inner {
                    *value *= e_out;
                }
            }
            inner
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        },
    ))
}

#[cfg(feature = "field-inline")]
fn product_uniskip_extended_evals_field_inline_from_stage2_rows<F>(
    config: Stage2ProverConfig,
    stage2_rows: &[JoltVmStage2TraceRow],
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    tau_low: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
{
    let rows = 1usize << config.log_t;
    if stage2_rows.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 field-inline product uni-skip stage rows have {} rows, expected {rows}",
                stage2_rows.len()
            ),
        });
    }
    if field_factors.rs1.len() != rows
        || field_factors.rs2.len() != rows
        || field_factors.rd.len() != rows
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 2 field-inline product uni-skip factors have inconsistent row counts"
                .to_owned(),
        });
    }

    let weights = uniskip_targets(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
    )?
    .into_iter()
    .map(|target| {
        centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, F::from_i64(target))
            .map_err(invalid_sumcheck_output)
    })
    .collect::<Result<Vec<_>, ProverError>>()?;

    let eq_table = TensorEqTable::new(tau_low);
    if eq_table.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 field-inline product uni-skip eq table has {} rows, expected {rows}",
                eq_table.len()
            ),
        });
    }

    Ok(eq_table.par_fold_out_in(
        || vec![F::zero(); weights.len()],
        |inner, cycle, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            let row = stage2_rows[cycle];
            let left_instruction_input = F::from_u64(row.left_instruction_input);
            let right_instruction_input = F::from_i128(row.right_instruction_input);
            let lookup_output = F::from_u64(row.lookup_output);
            let jump_flag = F::from_bool(row.jump_flag);
            let branch_flag = F::from_bool(row.branch_flag);
            let not_next_is_noop = F::from_bool(!row.next_is_noop);
            let field_rs1 = field_factors.rs1[cycle];
            let field_rs2 = field_factors.rs2[cycle];
            let field_rd = field_factors.rd[cycle];

            for (total, weights) in inner.iter_mut().zip(&weights) {
                let left = weights[0] * left_instruction_input
                    + weights[1] * lookup_output
                    + weights[2] * jump_flag
                    + (weights[3] + weights[4]) * field_rs1;
                let right = weights[0] * right_instruction_input
                    + weights[1] * branch_flag
                    + weights[2] * not_next_is_noop
                    + weights[3] * field_rs2
                    + weights[4] * field_rd;
                *total += e_in * left * right;
            }
        },
        |_x_out, e_out, mut inner| {
            if e_out.is_zero() {
                inner.fill(F::zero());
            } else {
                for value in &mut inner {
                    *value *= e_out;
                }
            }
            inner
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        },
    ))
}

fn selected_product_uniskip_input_claim<F: Field>(
    input: &Stage2ProductUniSkipInput<F>,
    weights: &[F],
) -> Result<F, ProverError> {
    let [product, should_branch, should_jump, rest @ ..] = weights else {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip expected at least 3 weights, got {}",
            weights.len()
        )));
    };
    let claim = *product * input.product
        + *should_branch * input.should_branch
        + *should_jump * input.should_jump;
    #[cfg(feature = "field-inline")]
    {
        let [field_product, field_inv_product] = rest else {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 field-inline product uni-skip expected 5 weights, got {}",
                weights.len()
            )));
        };
        Ok(claim
            + *field_product * input.field_product
            + *field_inv_product * input.field_inv_product)
    }
    #[cfg(not(feature = "field-inline"))]
    {
        if !rest.is_empty() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 product uni-skip expected 3 weights, got {}",
                weights.len()
            )));
        }
        Ok(claim)
    }
}

fn build_product_uniskip_first_round_poly<F: Field>(
    base_evals: &[F; SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, ProverError> {
    let interpolation_degree = PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT;
    if extended_evals.len() != interpolation_degree {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product uni-skip extended eval count mismatch: got {}, expected {interpolation_degree}",
            extended_evals.len()
        )));
    }

    let extended_size = 2 * interpolation_degree + 1;
    let mut t1_values = vec![F::zero(); extended_size];
    let base_start = centered_domain_start(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE)
        .map_err(invalid_sumcheck_output)?;
    for (index, &value) in base_evals.iter().enumerate() {
        let target = base_start
            + i64::try_from(index).map_err(|_| {
                invalid_sumcheck_output(format!("Stage 2 base index {index} is out of range"))
            })?;
        set_uniskip_value(&mut t1_values, interpolation_degree, target, value)?;
    }

    for (target, &value) in
        uniskip_targets(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, interpolation_degree)?
            .iter()
            .zip(extended_evals)
    {
        set_uniskip_value(&mut t1_values, interpolation_degree, *target, value)?;
    }

    let t1_coeffs = interpolate_to_coeffs(-(interpolation_degree as i64), &t1_values);
    let lagrange_values = centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)
        .map_err(invalid_sumcheck_output)?;
    let lagrange_coeffs = interpolate_to_coeffs(base_start, &lagrange_values);
    let mut coeffs = poly_mul(&lagrange_coeffs, &t1_coeffs);
    coeffs.resize(SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE + 1, F::zero());
    Ok(UnivariatePoly::new(coeffs))
}

fn set_uniskip_value<F: Field>(
    values: &mut [F],
    degree: usize,
    target: i64,
    value: F,
) -> Result<(), ProverError> {
    let position = usize::try_from(target + degree as i64).map_err(|_| {
        invalid_sumcheck_output(format!("Stage 2 uniskip target {target} is out of range"))
    })?;
    let Some(slot) = values.get_mut(position) else {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 uniskip target {target} is outside extended domain"
        )));
    };
    *slot = value;
    Ok(())
}

fn centered_domain_sum<F: Field>(
    poly: &UnivariatePoly<F>,
    domain_size: usize,
) -> Result<F, ProverError> {
    let start = centered_domain_start(domain_size).map_err(invalid_sumcheck_output)?;
    (0..domain_size)
        .map(|offset| {
            let target = start
                + i64::try_from(offset).map_err(|_| {
                    invalid_sumcheck_output(format!(
                        "Stage 2 centered-domain offset {offset} is out of range"
                    ))
                })?;
            Ok(poly.evaluate(F::from_i64(target)))
        })
        .sum()
}

fn uniskip_targets(domain_size: usize, degree: usize) -> Result<Vec<i64>, ProverError> {
    let base_left = centered_domain_start(domain_size).map_err(invalid_sumcheck_output)?;
    let base_right = base_left
        + i64::try_from(domain_size).map_err(|_| {
            invalid_sumcheck_output(format!(
                "Stage 2 uniskip domain size {domain_size} is too large"
            ))
        })?
        - 1;
    let ext_left = -(degree as i64);
    let ext_right = degree as i64;
    let mut targets = Vec::with_capacity(degree);
    let mut left = base_left - 1;
    let mut right = base_right + 1;

    while targets.len() < degree && left >= ext_left && right <= ext_right {
        targets.push(left);
        if targets.len() < degree {
            targets.push(right);
        }
        left -= 1;
        right += 1;
    }
    while targets.len() < degree && left >= ext_left {
        targets.push(left);
        left -= 1;
    }
    while targets.len() < degree && right <= ext_right {
        targets.push(right);
        right += 1;
    }

    Ok(targets)
}

#[cfg(not(feature = "field-inline"))]
fn ordered_row_product_outputs<F: Field>(
    outputs: Vec<SumcheckLinearProductOutput<F>>,
    expected_count: usize,
) -> Result<Vec<F>, ProverError> {
    if outputs.len() != expected_count {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 row product backend returned {} outputs, expected {expected_count}",
            outputs.len()
        )));
    }
    let mut values = vec![None; expected_count];
    for output in outputs {
        let index = usize::try_from(output.slot.0).map_err(|_| {
            invalid_sumcheck_output(format!(
                "Stage 2 row product output slot {:?} is out of range",
                output.slot
            ))
        })?;
        let Some(slot) = values.get_mut(index) else {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 row product output slot {:?} exceeds expected count {expected_count}",
                output.slot
            )));
        };
        if slot.replace(output.value).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 2 row product output slot {:?}",
                output.slot
            )));
        }
    }
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                invalid_sumcheck_output(format!(
                    "missing Stage 2 row product output slot {:?}",
                    value_slot(index).unwrap_or(BackendValueSlot(u32::MAX))
                ))
            })
        })
        .collect()
}

#[cfg(not(feature = "field-inline"))]
fn value_slot(index: usize) -> Result<BackendValueSlot, ProverError> {
    Ok(BackendValueSlot(u32::try_from(index).map_err(|_| {
        invalid_sumcheck_output(format!(
            "Stage 2 query index {index} exceeds value slot range"
        ))
    })?))
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
