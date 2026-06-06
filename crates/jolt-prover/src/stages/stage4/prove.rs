use jolt_backends::{
    Stage4ReadWriteSumcheckBackend, SumcheckBackend, SumcheckRamReadWriteRow,
    SumcheckRamValCheckStateRequest, SumcheckRegisterRead, SumcheckRegisterWrite,
    SumcheckRegistersReadWriteRow, SumcheckRegistersReadWriteStateRequest,
};
#[cfg(feature = "field-inline")]
use jolt_backends::{
    SumcheckFieldRegisterRead, SumcheckFieldRegisterWrite, SumcheckFieldRegistersReadWriteRow,
    SumcheckFieldRegistersReadWriteStateRequest,
};
use jolt_claims::protocols::jolt::formulas::dimensions::REGISTER_ADDRESS_BITS;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::{try_eq_mle, LtPolynomial, Point, UnivariatePoly};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::{LabelWithCount, Transcript};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage4::inputs::{
    FieldInlineStage4Claims, FieldRegistersReadWriteOutputOpeningClaims,
};
use jolt_verifier::stages::stage4::inputs::{
    RamValCheckOutputOpeningClaims, RegistersReadWriteOutputOpeningClaims, Stage4Claims,
};
use jolt_verifier::stages::stage4::outputs::{
    RamValCheckInitialEvaluation, Stage4ClearOutput, Stage4PublicOutput,
    VerifiedRamValCheckAdviceContribution, VerifiedStage4Batch, VerifiedStage4Sumcheck,
};
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage3::Stage3ClearOutput};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
};
use jolt_witness::protocols::jolt_vm::{
    JoltVmRegisterReadWriteRow, JoltVmRegisterReadWriteRows, JoltVmStage2Rows, JoltVmStage2TraceRow,
};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckBuilder;
use crate::ProverError;

use super::input::Stage4ProverInput;
#[cfg(feature = "zk")]
use super::output::Stage4CommittedBoundaryOutput;
use super::output::{
    stage4_advice_claims_from_prefix, Stage4ProverOutput, Stage4RegularBatchExpectedOutputs,
    Stage4RegularBatchProofOutput,
};
use super::{
    input::Stage4ProverConfig,
    output::{
        stage4_output_openings_from_evaluations, Stage4RamValCheckInitialEvaluation,
        Stage4RegularBatchInputClaims, Stage4RegularBatchOutputOpeningClaims,
        Stage4RegularBatchPrefixOutput,
    },
    request::build_stage4_output_opening_evaluation_request,
};

#[cfg(feature = "field-inline")]
const STAGE4_FIELD_REGISTERS_READ_WRITE_OPTIMIZATION_IDS: &[&str] = &[
    "OPT-RW-001",
    "OPT-RW-002",
    "OPT-RW-003",
    "OPT-RW-004",
    "OPT-RW-005",
    "OPT-RW-006",
    "OPT-RW-007",
    "OPT-RW-008",
    "OPT-RW-009",
    "OPT-RW-010",
    "OPT-FLD-002",
    "OPT-FLD-003",
];

/// Canonical Stage 4 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage4/verify.rs` in prover order: derive
/// the register/RAM value-check gammas (with the `ram_val_check_gamma` domain
/// separator), prove the registers read-write + RAM value-check batched
/// sumcheck, evaluate output openings, and assemble the verifier-owned
/// `stage4_sumcheck_proof`, [`Stage4Claims`], and [`Stage4ClearOutput`] for
/// Stage 5 and later stages.
///
/// `ram_val_check_init` is supplied via the input bundle; computing it prover-side
/// from preprocessing and the advice witness is a tracked self-containment
/// follow-up. Field-inline and ZK Stage 4 prover paths are not yet implemented.
#[cfg(not(feature = "field-inline"))]
pub fn prove<F, W, B, T, C>(
    input: Stage4ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage4ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 4 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }
    if input.checked.ram_K != (1usize << input.config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 checked RAM K {} does not match log_k {}",
                input.checked.ram_K, input.config.log_k
            ),
        });
    }

    let prefix = derive_stage4_regular_batch_prefix(
        input.config,
        input.stage2,
        input.stage3,
        input.ram_val_check_init.clone(),
        transcript,
    )?;
    let proof_output = prove_stage4_transparent_sumchecks::<F, W, B, T, C>(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage3,
        &prefix,
        transcript,
    )?;

    let claims = Stage4Claims {
        advice: proof_output.output_openings.advice.clone(),
        registers_read_write: proof_output.output_openings.registers_read_write.clone(),
        ram_val_check: proof_output.output_openings.ram_val_check.clone(),
    };
    let public = Stage4PublicOutput {
        challenges: proof_output.sumcheck_point.clone(),
        batching_coefficients: proof_output.batching_coefficients.clone(),
        registers_gamma: prefix.registers_gamma,
        ram_val_check_gamma: prefix.ram_val_check_gamma,
    };
    let verifier_output = Stage4ClearOutput {
        public,
        output_claims: claims.clone(),
        ram_val_check_init: stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init),
        batch: VerifiedStage4Batch {
            batching_coefficients: proof_output.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(proof_output.sumcheck_point.clone()),
            sumcheck_final_claim: proof_output.sumcheck_final_claim,
            expected_final_claim: proof_output.expected_final_claim,
            registers_read_write: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.registers_read_write,
                sumcheck_point: proof_output.registers_read_write_sumcheck_point.clone(),
                opening_point: proof_output.registers_read_write_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.registers_read_write,
            },
            ram_val_check: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.ram_val_check,
                sumcheck_point: proof_output.ram_val_check_sumcheck_point.clone(),
                opening_point: proof_output.ram_val_check_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.ram_val_check,
            },
        },
    };

    Ok(Stage4ProverOutput {
        stage4_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
pub fn prove_committed_boundary<F, W, B, T, VC>(
    input: Stage4ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage4CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage4_committed_checked(input.config, input.checked)?;
    let prefix = derive_stage4_regular_batch_prefix(
        input.config,
        input.stage2,
        input.stage3,
        input.ram_val_check_init.clone(),
        transcript,
    )?;
    let output = prove_stage4_committed_specialized_regular_batch_sumcheck::<F, W, B, T, VC>(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage3,
        &prefix,
        transcript,
        vc_setup,
    )?;
    Ok(output)
}

#[cfg(feature = "field-inline")]
pub fn prove<F, W, FI, B, T, C>(
    input: Stage4ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage4ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 4 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }
    if input.checked.ram_K != (1usize << input.config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 checked RAM K {} does not match log_k {}",
                input.checked.ram_K, input.config.log_k
            ),
        });
    }

    let prefix = derive_stage4_regular_batch_prefix(
        input.config,
        input.stage2,
        input.stage3,
        input.ram_val_check_init.clone(),
        transcript,
    )?;
    let proof_output = prove_stage4_transparent_sumchecks::<F, W, FI, B, T, C>(
        input.config,
        input.witness,
        input.field_inline_witness,
        backend,
        input.stage2,
        input.stage3,
        &prefix,
        transcript,
    )?;

    let claims = Stage4Claims {
        advice: proof_output.output_openings.advice.clone(),
        registers_read_write: proof_output.output_openings.registers_read_write.clone(),
        field_inline: proof_output.output_openings.field_inline.clone(),
        ram_val_check: proof_output.output_openings.ram_val_check.clone(),
    };
    let public = Stage4PublicOutput {
        challenges: proof_output.sumcheck_point.clone(),
        batching_coefficients: proof_output.batching_coefficients.clone(),
        registers_gamma: prefix.registers_gamma,
        field_registers_gamma: prefix.field_registers_gamma,
        ram_val_check_gamma: prefix.ram_val_check_gamma,
    };
    let verifier_output = Stage4ClearOutput {
        public,
        output_claims: claims.clone(),
        ram_val_check_init: stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init),
        batch: VerifiedStage4Batch {
            batching_coefficients: proof_output.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(proof_output.sumcheck_point.clone()),
            sumcheck_final_claim: proof_output.sumcheck_final_claim,
            expected_final_claim: proof_output.expected_final_claim,
            registers_read_write: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.registers_read_write,
                sumcheck_point: proof_output.registers_read_write_sumcheck_point.clone(),
                opening_point: proof_output.registers_read_write_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.registers_read_write,
            },
            field_registers_read_write: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.field_registers_read_write,
                sumcheck_point: proof_output
                    .field_registers_read_write_sumcheck_point
                    .clone(),
                opening_point: proof_output
                    .field_registers_read_write_opening_point
                    .clone(),
                expected_output_claim: proof_output.expected_outputs.field_registers_read_write,
            },
            ram_val_check: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.ram_val_check,
                sumcheck_point: proof_output.ram_val_check_sumcheck_point.clone(),
                opening_point: proof_output.ram_val_check_opening_point.clone(),
                expected_output_claim: proof_output.expected_outputs.ram_val_check,
            },
        },
    };

    Ok(Stage4ProverOutput {
        stage4_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
pub fn prove_committed_boundary<F, W, FI, B, T, VC>(
    input: Stage4ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage4CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage4_committed_checked(input.config, input.checked)?;
    let prefix = derive_stage4_regular_batch_prefix(
        input.config,
        input.stage2,
        input.stage3,
        input.ram_val_check_init.clone(),
        transcript,
    )?;
    let output = prove_stage4_committed_specialized_regular_batch_sumcheck::<F, W, FI, B, T, VC>(
        input.config,
        input.witness,
        input.field_inline_witness,
        backend,
        input.stage2,
        input.stage3,
        &prefix,
        transcript,
        vc_setup,
    )?;
    Ok(output)
}

fn stage4_ram_val_check_init_to_verifier<F: Field>(
    init: &Stage4RamValCheckInitialEvaluation<F>,
) -> RamValCheckInitialEvaluation<F> {
    RamValCheckInitialEvaluation {
        public_eval: init.public_eval,
        advice_contributions: init
            .advice_contributions
            .iter()
            .map(|contribution| VerifiedRamValCheckAdviceContribution {
                kind: contribution.kind,
                selector: contribution.selector,
                opening_claim: contribution.opening_claim,
                opening_point: contribution.opening_point.clone(),
            })
            .collect(),
        full_eval: init.full_eval,
    }
}

pub fn derive_stage4_regular_batch_prefix<F, T>(
    config: Stage4ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
    transcript: &mut T,
) -> Result<Stage4RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    validate_stage4_dependencies(config, stage2, stage3)?;

    let registers_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_registers_gamma = transcript.challenge_scalar();

    append_ram_val_check_gamma_domain_separator(transcript);
    let ram_val_check_gamma = transcript.challenge_scalar();

    let registers_gamma2 = registers_gamma * registers_gamma;
    #[cfg(feature = "field-inline")]
    let field_registers_gamma2 = field_registers_gamma * field_registers_gamma;

    let input_claims = Stage4RegularBatchInputClaims {
        registers_read_write: stage3
            .output_claims
            .registers_claim_reduction
            .rd_write_value
            + registers_gamma * stage3.output_claims.registers_claim_reduction.rs1_value
            + registers_gamma2 * stage3.output_claims.registers_claim_reduction.rs2_value,
        #[cfg(feature = "field-inline")]
        field_registers_read_write: stage2.output_claims.field_inline.product.field_rd_value
            + field_registers_gamma * stage2.output_claims.field_inline.product.field_rs1_value
            + field_registers_gamma2 * stage2.output_claims.field_inline.product.field_rs2_value,
        ram_val_check: stage2.output_claims.ram_read_write.val
            + ram_val_check_gamma * stage2.output_claims.ram_output_check
            - (F::one() + ram_val_check_gamma) * ram_val_check_init.full_eval,
    };

    Ok(Stage4RegularBatchPrefixOutput {
        input_claims,
        registers_gamma,
        #[cfg(feature = "field-inline")]
        field_registers_gamma,
        ram_val_check_gamma,
        ram_val_check_init,
    })
}

pub fn evaluate_stage4_output_openings<F, W, B>(
    config: Stage4ProverConfig,
    witness: &W,
    backend: &mut B,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    registers_read_write_opening_point: Vec<F>,
    ram_val_check_opening_point: Vec<F>,
) -> Result<Stage4RegularBatchOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let request = build_stage4_output_opening_evaluation_request(
        config,
        witness,
        registers_read_write_opening_point,
        ram_val_check_opening_point,
    )?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    stage4_output_openings_from_evaluations(prefix, &request, evaluations)
}

#[cfg(not(feature = "field-inline"))]
pub fn prove_stage4_transparent_sumchecks<F, W, B, T, C>(
    config: Stage4ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage4RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let proof_output = prove_stage4_specialized_regular_batch_sumcheck::<F, W, B, T, C>(
        config, witness, backend, stage2, stage3, prefix, transcript,
    )?;
    append_stage4_opening_claims(transcript, &proof_output.output_openings);
    Ok(proof_output)
}

#[cfg(feature = "field-inline")]
#[expect(clippy::too_many_arguments)]
pub fn prove_stage4_transparent_sumchecks<F, W, FI, B, T, C>(
    config: Stage4ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage4RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let proof_output = prove_stage4_specialized_regular_batch_sumcheck::<F, W, FI, B, T, C>(
        config,
        witness,
        field_inline_witness,
        backend,
        stage2,
        stage3,
        prefix,
        transcript,
    )?;
    append_stage4_opening_claims(transcript, &proof_output.output_openings);
    Ok(proof_output)
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchFrontierProof<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub output_claim: F,
}

#[cfg(all(feature = "frontier-harness", not(feature = "field-inline")))]
pub fn prove_stage4_regular_batch_sumcheck_for_frontier<F, W, B, T, C>(
    input: &Stage4ProverInput<'_, F, W>,
    backend: &mut B,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage4RegularBatchFrontierProof<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let proof_output = prove_stage4_specialized_regular_batch_sumcheck::<F, W, B, T, C>(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage3,
        prefix,
        transcript,
    )?;
    Ok(Stage4RegularBatchFrontierProof {
        proof: proof_output.proof,
        challenges: proof_output.sumcheck_point,
        batching_coefficients: proof_output.batching_coefficients,
        output_claim: proof_output.sumcheck_final_claim,
    })
}

#[cfg(not(feature = "field-inline"))]
fn prove_stage4_specialized_regular_batch_sumcheck<F, W, B, T, C>(
    config: Stage4ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage4RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let register_dimensions = config
        .rw_config
        .register_dimensions(config.log_t, REGISTER_ADDRESS_BITS);
    let fixed_register_cycle_point = stage3.batch.registers_claim_reduction.opening_point.clone();
    if fixed_register_cycle_point.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 fixed register cycle point has {} variables, expected {}",
            fixed_register_cycle_point.len(),
            config.log_t
        )));
    }
    let (fixed_ram_address_point, fixed_ram_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);

    let register_request = SumcheckRegistersReadWriteStateRequest::new(
        "stage4.registers_read_write",
        stage4_register_rows(witness)?,
        fixed_register_cycle_point.clone(),
        prefix.registers_gamma,
        prefix.input_claims.registers_read_write,
        config.log_t,
        REGISTER_ADDRESS_BITS,
        register_dimensions.phase1_num_rounds(),
        register_dimensions.phase2_num_rounds(),
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let ram_request = SumcheckRamValCheckStateRequest::new(
        "stage4.ram_val_check",
        stage4_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_cycle_point.to_vec(),
        prefix.ram_val_check_gamma,
        prefix.input_claims.ram_val_check,
        config.log_t,
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let mut register_state =
        backend.materialize_sumcheck_registers_read_write_state(&register_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_val_check_state(&ram_request)?;

    append_sumcheck_claim(transcript, &prefix.input_claims.registers_read_write);
    append_sumcheck_claim(transcript, &prefix.input_claims.ram_val_check);
    let batching_coefficients = [transcript.challenge_scalar(), transcript.challenge_scalar()];
    let max_num_rounds = config.log_t + REGISTER_ADDRESS_BITS;
    let ram_offset = REGISTER_ADDRESS_BITS;
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut individual_claims = [
        prefix.input_claims.registers_read_write,
        prefix.input_claims.ram_val_check.mul_pow_2(ram_offset),
    ];
    let mut running_claim = individual_claims[0] * batching_coefficients[0]
        + individual_claims[1] * batching_coefficients[1];
    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let register_poly = backend
            .evaluate_sumcheck_registers_read_write_round(&register_state, individual_claims[0])?;
        let ram_poly = if round < ram_offset {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend.evaluate_sumcheck_ram_val_check_round(&ram_state, individual_claims[1])?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&register_poly * batching_coefficients[0]);
        round_poly += &(&ram_poly * batching_coefficients[1]);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 4 specialized regular batch round {round} sumcheck invariant failed"
            )));
        }

        CompressedLabeledRoundPoly::sumcheck(&round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = round_poly.evaluate(challenge);
        challenges.push(challenge);
        individual_claims[0] = register_poly.evaluate(challenge);
        individual_claims[1] = if round < ram_offset {
            individual_claims[1] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_registers_read_write_state(&mut register_state, challenge)?;
        if round >= ram_offset {
            backend.bind_sumcheck_ram_val_check_state(&mut ram_state, challenge)?;
        }
        round_polynomials.push(round_poly.compress());
    }

    let registers_read_write_sumcheck_point = challenges.clone();
    let registers_read_write_opening_point = register_dimensions
        .read_write_opening_point(&registers_read_write_sumcheck_point)
        .map(|point| point.opening_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let ram_val_check_sumcheck_point = challenges[ram_offset..].to_vec();
    let ram_val_check_cycle_point = ram_val_check_sumcheck_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let ram_val_check_opening_point = [
        fixed_ram_address_point,
        ram_val_check_cycle_point.as_slice(),
    ]
    .concat();
    let registers_output = backend.output_sumcheck_registers_read_write_state(
        &register_state,
        &registers_read_write_opening_point,
    )?;
    let ram_output = backend.output_sumcheck_ram_val_check_state(&ram_state)?;
    let output_openings = Stage4RegularBatchOutputOpeningClaims {
        advice: stage4_advice_claims_from_prefix(prefix)?,
        registers_read_write: RegistersReadWriteOutputOpeningClaims {
            registers_val: registers_output.registers_val,
            rs1_ra: registers_output.rs1_ra,
            rs2_ra: registers_output.rs2_ra,
            rd_wa: registers_output.rd_wa,
            rd_inc: registers_output.rd_inc,
        },
        ram_val_check: RamValCheckOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
            ram_inc: ram_output.ram_inc,
        },
    };
    let (_, registers_cycle_point) =
        registers_read_write_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let eq_cycle = try_eq_mle(&fixed_register_cycle_point, registers_cycle_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let registers_expected = eq_cycle
        * (output_openings.registers_read_write.rd_wa
            * (output_openings.registers_read_write.rd_inc
                + output_openings.registers_read_write.registers_val)
            + prefix.registers_gamma
                * output_openings.registers_read_write.rs1_ra
                * output_openings.registers_read_write.registers_val
            + prefix.registers_gamma
                * prefix.registers_gamma
                * output_openings.registers_read_write.rs2_ra
                * output_openings.registers_read_write.registers_val);
    let ram_lt = LtPolynomial::evaluate(&ram_val_check_cycle_point, fixed_ram_cycle_point);
    let ram_expected = (ram_lt + prefix.ram_val_check_gamma)
        * output_openings.ram_val_check.ram_inc
        * output_openings.ram_val_check.ram_ra;
    let expected_final_claim =
        batching_coefficients[0] * registers_expected + batching_coefficients[1] * ram_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 4 batch final claim did not match output openings",
        ));
    }

    Ok(Stage4RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        output_openings,
        expected_outputs: Stage4RegularBatchExpectedOutputs {
            registers_read_write: registers_expected,
            ram_val_check: ram_expected,
        },
        batching_coefficients: batching_coefficients.to_vec(),
        sumcheck_point: challenges,
        sumcheck_final_claim: running_claim,
        expected_final_claim,
        registers_read_write_sumcheck_point,
        registers_read_write_opening_point,
        ram_val_check_sumcheck_point,
        ram_val_check_opening_point,
    })
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
#[expect(clippy::too_many_arguments)]
fn prove_stage4_committed_specialized_regular_batch_sumcheck<F, W, B, T, VC>(
    config: Stage4ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage4CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let register_dimensions = config
        .rw_config
        .register_dimensions(config.log_t, REGISTER_ADDRESS_BITS);
    let fixed_register_cycle_point = stage3.batch.registers_claim_reduction.opening_point.clone();
    if fixed_register_cycle_point.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 fixed register cycle point has {} variables, expected {}",
            fixed_register_cycle_point.len(),
            config.log_t
        )));
    }
    let (fixed_ram_address_point, fixed_ram_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);

    let register_request = SumcheckRegistersReadWriteStateRequest::new(
        "stage4.registers_read_write",
        stage4_register_rows(witness)?,
        fixed_register_cycle_point.clone(),
        prefix.registers_gamma,
        prefix.input_claims.registers_read_write,
        config.log_t,
        REGISTER_ADDRESS_BITS,
        register_dimensions.phase1_num_rounds(),
        register_dimensions.phase2_num_rounds(),
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let ram_request = SumcheckRamValCheckStateRequest::new(
        "stage4.ram_val_check",
        stage4_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_cycle_point.to_vec(),
        prefix.ram_val_check_gamma,
        prefix.input_claims.ram_val_check,
        config.log_t,
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let mut register_state =
        backend.materialize_sumcheck_registers_read_write_state(&register_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_val_check_state(&ram_request)?;

    let batching_coefficients = [transcript.challenge_scalar(), transcript.challenge_scalar()];
    let max_num_rounds = config.log_t + REGISTER_ADDRESS_BITS;
    let ram_offset = REGISTER_ADDRESS_BITS;
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut individual_claims = [
        prefix.input_claims.registers_read_write,
        prefix.input_claims.ram_val_check.mul_pow_2(ram_offset),
    ];
    let mut running_claim = individual_claims[0] * batching_coefficients[0]
        + individual_claims[1] * batching_coefficients[1];
    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, max_num_rounds)?;

    for round in 0..max_num_rounds {
        let register_poly = backend
            .evaluate_sumcheck_registers_read_write_round(&register_state, individual_claims[0])?;
        let ram_poly = if round < ram_offset {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend.evaluate_sumcheck_ram_val_check_round(&ram_state, individual_claims[1])?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&register_poly * batching_coefficients[0]);
        round_poly += &(&ram_poly * batching_coefficients[1]);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 4 committed regular batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = builder.commit_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        challenges.push(challenge);
        individual_claims[0] = register_poly.evaluate(challenge);
        individual_claims[1] = if round < ram_offset {
            individual_claims[1] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_registers_read_write_state(&mut register_state, challenge)?;
        if round >= ram_offset {
            backend.bind_sumcheck_ram_val_check_state(&mut ram_state, challenge)?;
        }
    }

    let registers_read_write_sumcheck_point = challenges.clone();
    let registers_read_write_opening_point = register_dimensions
        .read_write_opening_point(&registers_read_write_sumcheck_point)
        .map(|point| point.opening_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let ram_val_check_sumcheck_point = challenges[ram_offset..].to_vec();
    let ram_val_check_cycle_point = ram_val_check_sumcheck_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let ram_val_check_opening_point = [
        fixed_ram_address_point,
        ram_val_check_cycle_point.as_slice(),
    ]
    .concat();
    let registers_output = backend.output_sumcheck_registers_read_write_state(
        &register_state,
        &registers_read_write_opening_point,
    )?;
    let ram_output = backend.output_sumcheck_ram_val_check_state(&ram_state)?;
    let output_openings = Stage4RegularBatchOutputOpeningClaims {
        advice: stage4_advice_claims_from_prefix(prefix)?,
        registers_read_write: RegistersReadWriteOutputOpeningClaims {
            registers_val: registers_output.registers_val,
            rs1_ra: registers_output.rs1_ra,
            rs2_ra: registers_output.rs2_ra,
            rd_wa: registers_output.rd_wa,
            rd_inc: registers_output.rd_inc,
        },
        ram_val_check: RamValCheckOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
            ram_inc: ram_output.ram_inc,
        },
    };
    let (_, registers_cycle_point) =
        registers_read_write_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let eq_cycle = try_eq_mle(&fixed_register_cycle_point, registers_cycle_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let registers_expected = eq_cycle
        * (output_openings.registers_read_write.rd_wa
            * (output_openings.registers_read_write.rd_inc
                + output_openings.registers_read_write.registers_val)
            + prefix.registers_gamma
                * output_openings.registers_read_write.rs1_ra
                * output_openings.registers_read_write.registers_val
            + prefix.registers_gamma
                * prefix.registers_gamma
                * output_openings.registers_read_write.rs2_ra
                * output_openings.registers_read_write.registers_val);
    let ram_lt = LtPolynomial::evaluate(&ram_val_check_cycle_point, fixed_ram_cycle_point);
    let ram_expected = (ram_lt + prefix.ram_val_check_gamma)
        * output_openings.ram_val_check.ram_inc
        * output_openings.ram_val_check.ram_ra;
    let expected_final_claim =
        batching_coefficients[0] * registers_expected + batching_coefficients[1] * ram_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 4 batch final claim did not match output openings",
        ));
    }

    let output_claim_values = stage4_committed_output_claim_values(&output_openings);
    let claims = Stage4Claims {
        advice: output_openings.advice.clone(),
        registers_read_write: output_openings.registers_read_write.clone(),
        ram_val_check: output_openings.ram_val_check.clone(),
    };
    let public = Stage4PublicOutput {
        challenges: challenges.clone(),
        batching_coefficients: batching_coefficients.to_vec(),
        registers_gamma: prefix.registers_gamma,
        ram_val_check_gamma: prefix.ram_val_check_gamma,
    };
    let verifier_output = Stage4ClearOutput {
        public: public.clone(),
        output_claims: claims,
        ram_val_check_init: stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init),
        batch: VerifiedStage4Batch {
            batching_coefficients: batching_coefficients.to_vec(),
            sumcheck_point: Point::high_to_low(challenges.clone()),
            sumcheck_final_claim: running_claim,
            expected_final_claim,
            registers_read_write: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.registers_read_write,
                sumcheck_point: registers_read_write_sumcheck_point.clone(),
                opening_point: registers_read_write_opening_point.clone(),
                expected_output_claim: registers_expected,
            },
            ram_val_check: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.ram_val_check,
                sumcheck_point: ram_val_check_sumcheck_point.clone(),
                opening_point: ram_val_check_opening_point.clone(),
                expected_output_claim: ram_expected,
            },
        },
    };
    let built = builder.finish(&output_claim_values, transcript)?;
    Ok(Stage4CommittedBoundaryOutput {
        stage4_sumcheck_proof: built.proof,
        public,
        verifier_output,
        output_claim_values,
        committed_witness: built.witness,
    })
}

#[cfg(feature = "field-inline")]
#[expect(clippy::too_many_arguments)]
fn prove_stage4_specialized_regular_batch_sumcheck<F, W, FI, B, T, C>(
    config: Stage4ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage4RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let register_dimensions = config
        .rw_config
        .register_dimensions(config.log_t, REGISTER_ADDRESS_BITS);
    let field_register_dimensions = config.field_inline.read_write_dimensions(config.log_t);
    let fixed_register_cycle_point = stage3.batch.registers_claim_reduction.opening_point.clone();
    if fixed_register_cycle_point.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 fixed register cycle point has {} variables, expected {}",
            fixed_register_cycle_point.len(),
            config.log_t
        )));
    }
    let fixed_field_register_cycle_point = stage2
        .batch
        .field_registers_claim_reduction
        .opening_point
        .clone();
    if fixed_field_register_cycle_point.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 fixed field-register cycle point has {} variables, expected {}",
            fixed_field_register_cycle_point.len(),
            config.log_t
        )));
    }
    let (fixed_ram_address_point, fixed_ram_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);

    let register_request = SumcheckRegistersReadWriteStateRequest::new(
        "stage4.registers_read_write",
        stage4_register_rows(witness)?,
        fixed_register_cycle_point.clone(),
        prefix.registers_gamma,
        prefix.input_claims.registers_read_write,
        config.log_t,
        REGISTER_ADDRESS_BITS,
        register_dimensions.phase1_num_rounds(),
        register_dimensions.phase2_num_rounds(),
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let field_register_request = SumcheckFieldRegistersReadWriteStateRequest::new(
        "stage4.field_registers_read_write",
        stage4_field_register_rows(field_inline_witness)?,
        fixed_field_register_cycle_point.clone(),
        prefix.field_registers_gamma,
        prefix.input_claims.field_registers_read_write,
        config.log_t,
        field_register_dimensions.log_k(),
        field_register_dimensions.phase1_num_rounds(),
        field_register_dimensions.phase2_num_rounds(),
    )
    .with_optimization_ids(STAGE4_FIELD_REGISTERS_READ_WRITE_OPTIMIZATION_IDS);
    let ram_request = SumcheckRamValCheckStateRequest::new(
        "stage4.ram_val_check",
        stage4_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_cycle_point.to_vec(),
        prefix.ram_val_check_gamma,
        prefix.input_claims.ram_val_check,
        config.log_t,
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let mut register_state =
        backend.materialize_sumcheck_registers_read_write_state(&register_request)?;
    let mut field_register_state =
        backend.materialize_sumcheck_field_registers_read_write_state(&field_register_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_val_check_state(&ram_request)?;

    append_sumcheck_claim(transcript, &prefix.input_claims.registers_read_write);
    append_sumcheck_claim(transcript, &prefix.input_claims.field_registers_read_write);
    append_sumcheck_claim(transcript, &prefix.input_claims.ram_val_check);
    let batching_coefficients = [
        transcript.challenge_scalar(),
        transcript.challenge_scalar(),
        transcript.challenge_scalar(),
    ];
    let max_num_rounds = config.log_t + REGISTER_ADDRESS_BITS;
    let field_offset = max_num_rounds - field_register_dimensions.read_write_sumcheck().rounds;
    let ram_offset = REGISTER_ADDRESS_BITS;
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut individual_claims = [
        prefix.input_claims.registers_read_write,
        prefix
            .input_claims
            .field_registers_read_write
            .mul_pow_2(field_offset),
        prefix.input_claims.ram_val_check.mul_pow_2(ram_offset),
    ];
    let mut running_claim = individual_claims[0] * batching_coefficients[0]
        + individual_claims[1] * batching_coefficients[1]
        + individual_claims[2] * batching_coefficients[2];
    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);

    for round in 0..max_num_rounds {
        let register_poly = backend
            .evaluate_sumcheck_registers_read_write_round(&register_state, individual_claims[0])?;
        let field_register_poly = if round < field_offset {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend.evaluate_sumcheck_field_registers_read_write_round(
                &field_register_state,
                individual_claims[1],
            )?
        };
        let ram_poly = if round < ram_offset {
            UnivariatePoly::new(vec![individual_claims[2] * two_inv])
        } else {
            backend.evaluate_sumcheck_ram_val_check_round(&ram_state, individual_claims[2])?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&register_poly * batching_coefficients[0]);
        round_poly += &(&field_register_poly * batching_coefficients[1]);
        round_poly += &(&ram_poly * batching_coefficients[2]);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 4 specialized regular batch round {round} sumcheck invariant failed"
            )));
        }

        CompressedLabeledRoundPoly::sumcheck(&round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = round_poly.evaluate(challenge);
        challenges.push(challenge);
        individual_claims[0] = register_poly.evaluate(challenge);
        individual_claims[1] = if round < field_offset {
            individual_claims[1] * two_inv
        } else {
            field_register_poly.evaluate(challenge)
        };
        individual_claims[2] = if round < ram_offset {
            individual_claims[2] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_registers_read_write_state(&mut register_state, challenge)?;
        if round >= field_offset {
            backend.bind_sumcheck_field_registers_read_write_state(
                &mut field_register_state,
                challenge,
            )?;
        }
        if round >= ram_offset {
            backend.bind_sumcheck_ram_val_check_state(&mut ram_state, challenge)?;
        }
        round_polynomials.push(round_poly.compress());
    }

    let registers_read_write_sumcheck_point = challenges.clone();
    let registers_read_write_opening_point = register_dimensions
        .read_write_opening_point(&registers_read_write_sumcheck_point)
        .map(|point| point.opening_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let field_registers_read_write_sumcheck_point = challenges[field_offset..].to_vec();
    let field_registers_opening = field_register_dimensions
        .read_write_opening_point(&field_registers_read_write_sumcheck_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let ram_val_check_sumcheck_point = challenges[ram_offset..].to_vec();
    let ram_val_check_cycle_point = ram_val_check_sumcheck_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let ram_val_check_opening_point = [
        fixed_ram_address_point,
        ram_val_check_cycle_point.as_slice(),
    ]
    .concat();
    let registers_output = backend.output_sumcheck_registers_read_write_state(
        &register_state,
        &registers_read_write_opening_point,
    )?;
    let field_registers_output = backend.output_sumcheck_field_registers_read_write_state(
        &field_register_state,
        &field_registers_opening.opening_point,
    )?;
    let ram_output = backend.output_sumcheck_ram_val_check_state(&ram_state)?;
    let output_openings = Stage4RegularBatchOutputOpeningClaims {
        advice: stage4_advice_claims_from_prefix(prefix)?,
        registers_read_write: RegistersReadWriteOutputOpeningClaims {
            registers_val: registers_output.registers_val,
            rs1_ra: registers_output.rs1_ra,
            rs2_ra: registers_output.rs2_ra,
            rd_wa: registers_output.rd_wa,
            rd_inc: registers_output.rd_inc,
        },
        field_inline: FieldInlineStage4Claims {
            field_registers_read_write: FieldRegistersReadWriteOutputOpeningClaims {
                field_registers_val: field_registers_output.registers_val,
                field_rs1_ra: field_registers_output.rs1_ra,
                field_rs2_ra: field_registers_output.rs2_ra,
                field_rd_wa: field_registers_output.rd_wa,
                field_rd_inc: field_registers_output.rd_inc,
            },
        },
        ram_val_check: RamValCheckOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
            ram_inc: ram_output.ram_inc,
        },
    };
    let (_, registers_cycle_point) =
        registers_read_write_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let eq_cycle = try_eq_mle(&fixed_register_cycle_point, registers_cycle_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let registers_expected = eq_cycle
        * (output_openings.registers_read_write.rd_wa
            * (output_openings.registers_read_write.rd_inc
                + output_openings.registers_read_write.registers_val)
            + prefix.registers_gamma
                * output_openings.registers_read_write.rs1_ra
                * output_openings.registers_read_write.registers_val
            + prefix.registers_gamma
                * prefix.registers_gamma
                * output_openings.registers_read_write.rs2_ra
                * output_openings.registers_read_write.registers_val);
    let field_claims = &output_openings.field_inline.field_registers_read_write;
    let field_eq_cycle = try_eq_mle(
        &fixed_field_register_cycle_point,
        &field_registers_opening.r_cycle,
    )
    .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let field_registers_expected = field_eq_cycle
        * (field_claims.field_rd_wa
            * (field_claims.field_rd_inc + field_claims.field_registers_val)
            + prefix.field_registers_gamma
                * field_claims.field_rs1_ra
                * field_claims.field_registers_val
            + prefix.field_registers_gamma
                * prefix.field_registers_gamma
                * field_claims.field_rs2_ra
                * field_claims.field_registers_val);
    let ram_lt = LtPolynomial::evaluate(&ram_val_check_cycle_point, fixed_ram_cycle_point);
    let ram_expected = (ram_lt + prefix.ram_val_check_gamma)
        * output_openings.ram_val_check.ram_inc
        * output_openings.ram_val_check.ram_ra;
    let expected_final_claim = batching_coefficients[0] * registers_expected
        + batching_coefficients[1] * field_registers_expected
        + batching_coefficients[2] * ram_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 4 batch final claim did not match output openings",
        ));
    }

    Ok(Stage4RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        output_openings,
        expected_outputs: Stage4RegularBatchExpectedOutputs {
            registers_read_write: registers_expected,
            field_registers_read_write: field_registers_expected,
            ram_val_check: ram_expected,
        },
        batching_coefficients: batching_coefficients.to_vec(),
        sumcheck_point: challenges,
        sumcheck_final_claim: running_claim,
        expected_final_claim,
        registers_read_write_sumcheck_point,
        registers_read_write_opening_point,
        field_registers_read_write_sumcheck_point,
        field_registers_read_write_opening_point: field_registers_opening.opening_point,
        ram_val_check_sumcheck_point,
        ram_val_check_opening_point,
    })
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
#[expect(clippy::too_many_arguments)]
fn prove_stage4_committed_specialized_regular_batch_sumcheck<F, W, FI, B, T, VC>(
    config: Stage4ProverConfig,
    witness: &W,
    field_inline_witness: &FI,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage4CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    FI: FieldInlineRegisterReadWriteRows<F>,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let register_dimensions = config
        .rw_config
        .register_dimensions(config.log_t, REGISTER_ADDRESS_BITS);
    let field_register_dimensions = config.field_inline.read_write_dimensions(config.log_t);
    let fixed_register_cycle_point = stage3.batch.registers_claim_reduction.opening_point.clone();
    if fixed_register_cycle_point.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 fixed register cycle point has {} variables, expected {}",
            fixed_register_cycle_point.len(),
            config.log_t
        )));
    }
    let fixed_field_register_cycle_point = stage2
        .batch
        .field_registers_claim_reduction
        .opening_point
        .clone();
    if fixed_field_register_cycle_point.len() != config.log_t {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 fixed field-register cycle point has {} variables, expected {}",
            fixed_field_register_cycle_point.len(),
            config.log_t
        )));
    }
    let (fixed_ram_address_point, fixed_ram_cycle_point) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);

    let register_request = SumcheckRegistersReadWriteStateRequest::new(
        "stage4.registers_read_write",
        stage4_register_rows(witness)?,
        fixed_register_cycle_point.clone(),
        prefix.registers_gamma,
        prefix.input_claims.registers_read_write,
        config.log_t,
        REGISTER_ADDRESS_BITS,
        register_dimensions.phase1_num_rounds(),
        register_dimensions.phase2_num_rounds(),
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let field_register_request = SumcheckFieldRegistersReadWriteStateRequest::new(
        "stage4.field_registers_read_write",
        stage4_field_register_rows(field_inline_witness)?,
        fixed_field_register_cycle_point.clone(),
        prefix.field_registers_gamma,
        prefix.input_claims.field_registers_read_write,
        config.log_t,
        field_register_dimensions.log_k(),
        field_register_dimensions.phase1_num_rounds(),
        field_register_dimensions.phase2_num_rounds(),
    )
    .with_optimization_ids(STAGE4_FIELD_REGISTERS_READ_WRITE_OPTIMIZATION_IDS);
    let ram_request = SumcheckRamValCheckStateRequest::new(
        "stage4.ram_val_check",
        stage4_ram_rows(witness)?,
        fixed_ram_address_point.to_vec(),
        fixed_ram_cycle_point.to_vec(),
        prefix.ram_val_check_gamma,
        prefix.input_claims.ram_val_check,
        config.log_t,
    )
    .with_optimization_ids(&["cpu_stage4_regular_batch_sumcheck"]);
    let mut register_state =
        backend.materialize_sumcheck_registers_read_write_state(&register_request)?;
    let mut field_register_state =
        backend.materialize_sumcheck_field_registers_read_write_state(&field_register_request)?;
    let mut ram_state = backend.materialize_sumcheck_ram_val_check_state(&ram_request)?;

    let batching_coefficients = [
        transcript.challenge_scalar(),
        transcript.challenge_scalar(),
        transcript.challenge_scalar(),
    ];
    let max_num_rounds = config.log_t + REGISTER_ADDRESS_BITS;
    let field_offset = max_num_rounds - field_register_dimensions.read_write_sumcheck().rounds;
    let ram_offset = REGISTER_ADDRESS_BITS;
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut individual_claims = [
        prefix.input_claims.registers_read_write,
        prefix
            .input_claims
            .field_registers_read_write
            .mul_pow_2(field_offset),
        prefix.input_claims.ram_val_check.mul_pow_2(ram_offset),
    ];
    let mut running_claim = individual_claims[0] * batching_coefficients[0]
        + individual_claims[1] * batching_coefficients[1]
        + individual_claims[2] * batching_coefficients[2];
    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, max_num_rounds)?;

    for round in 0..max_num_rounds {
        let register_poly = backend
            .evaluate_sumcheck_registers_read_write_round(&register_state, individual_claims[0])?;
        let field_register_poly = if round < field_offset {
            UnivariatePoly::new(vec![individual_claims[1] * two_inv])
        } else {
            backend.evaluate_sumcheck_field_registers_read_write_round(
                &field_register_state,
                individual_claims[1],
            )?
        };
        let ram_poly = if round < ram_offset {
            UnivariatePoly::new(vec![individual_claims[2] * two_inv])
        } else {
            backend.evaluate_sumcheck_ram_val_check_round(&ram_state, individual_claims[2])?
        };
        let mut round_poly = UnivariatePoly::zero();
        round_poly += &(&register_poly * batching_coefficients[0]);
        round_poly += &(&field_register_poly * batching_coefficients[1]);
        round_poly += &(&ram_poly * batching_coefficients[2]);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 4 committed regular batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = builder.commit_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        challenges.push(challenge);
        individual_claims[0] = register_poly.evaluate(challenge);
        individual_claims[1] = if round < field_offset {
            individual_claims[1] * two_inv
        } else {
            field_register_poly.evaluate(challenge)
        };
        individual_claims[2] = if round < ram_offset {
            individual_claims[2] * two_inv
        } else {
            ram_poly.evaluate(challenge)
        };
        backend.bind_sumcheck_registers_read_write_state(&mut register_state, challenge)?;
        if round >= field_offset {
            backend.bind_sumcheck_field_registers_read_write_state(
                &mut field_register_state,
                challenge,
            )?;
        }
        if round >= ram_offset {
            backend.bind_sumcheck_ram_val_check_state(&mut ram_state, challenge)?;
        }
    }

    let registers_read_write_sumcheck_point = challenges.clone();
    let registers_read_write_opening_point = register_dimensions
        .read_write_opening_point(&registers_read_write_sumcheck_point)
        .map(|point| point.opening_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let field_registers_read_write_sumcheck_point = challenges[field_offset..].to_vec();
    let field_registers_opening = field_register_dimensions
        .read_write_opening_point(&field_registers_read_write_sumcheck_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let ram_val_check_sumcheck_point = challenges[ram_offset..].to_vec();
    let ram_val_check_cycle_point = ram_val_check_sumcheck_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let ram_val_check_opening_point = [
        fixed_ram_address_point,
        ram_val_check_cycle_point.as_slice(),
    ]
    .concat();
    let registers_output = backend.output_sumcheck_registers_read_write_state(
        &register_state,
        &registers_read_write_opening_point,
    )?;
    let field_registers_output = backend.output_sumcheck_field_registers_read_write_state(
        &field_register_state,
        &field_registers_opening.opening_point,
    )?;
    let ram_output = backend.output_sumcheck_ram_val_check_state(&ram_state)?;
    let output_openings = Stage4RegularBatchOutputOpeningClaims {
        advice: stage4_advice_claims_from_prefix(prefix)?,
        registers_read_write: RegistersReadWriteOutputOpeningClaims {
            registers_val: registers_output.registers_val,
            rs1_ra: registers_output.rs1_ra,
            rs2_ra: registers_output.rs2_ra,
            rd_wa: registers_output.rd_wa,
            rd_inc: registers_output.rd_inc,
        },
        field_inline: FieldInlineStage4Claims {
            field_registers_read_write: FieldRegistersReadWriteOutputOpeningClaims {
                field_registers_val: field_registers_output.registers_val,
                field_rs1_ra: field_registers_output.rs1_ra,
                field_rs2_ra: field_registers_output.rs2_ra,
                field_rd_wa: field_registers_output.rd_wa,
                field_rd_inc: field_registers_output.rd_inc,
            },
        },
        ram_val_check: RamValCheckOutputOpeningClaims {
            ram_ra: ram_output.ram_ra,
            ram_inc: ram_output.ram_inc,
        },
    };
    let (_, registers_cycle_point) =
        registers_read_write_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let eq_cycle = try_eq_mle(&fixed_register_cycle_point, registers_cycle_point)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let registers_expected = eq_cycle
        * (output_openings.registers_read_write.rd_wa
            * (output_openings.registers_read_write.rd_inc
                + output_openings.registers_read_write.registers_val)
            + prefix.registers_gamma
                * output_openings.registers_read_write.rs1_ra
                * output_openings.registers_read_write.registers_val
            + prefix.registers_gamma
                * prefix.registers_gamma
                * output_openings.registers_read_write.rs2_ra
                * output_openings.registers_read_write.registers_val);
    let field_claims = &output_openings.field_inline.field_registers_read_write;
    let field_eq_cycle = try_eq_mle(
        &fixed_field_register_cycle_point,
        &field_registers_opening.r_cycle,
    )
    .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let field_registers_expected = field_eq_cycle
        * (field_claims.field_rd_wa
            * (field_claims.field_rd_inc + field_claims.field_registers_val)
            + prefix.field_registers_gamma
                * field_claims.field_rs1_ra
                * field_claims.field_registers_val
            + prefix.field_registers_gamma
                * prefix.field_registers_gamma
                * field_claims.field_rs2_ra
                * field_claims.field_registers_val);
    let ram_lt = LtPolynomial::evaluate(&ram_val_check_cycle_point, fixed_ram_cycle_point);
    let ram_expected = (ram_lt + prefix.ram_val_check_gamma)
        * output_openings.ram_val_check.ram_inc
        * output_openings.ram_val_check.ram_ra;
    let expected_final_claim = batching_coefficients[0] * registers_expected
        + batching_coefficients[1] * field_registers_expected
        + batching_coefficients[2] * ram_expected;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 4 batch final claim did not match output openings",
        ));
    }

    let output_claim_values = stage4_committed_output_claim_values(&output_openings);
    let claims = Stage4Claims {
        advice: output_openings.advice.clone(),
        registers_read_write: output_openings.registers_read_write.clone(),
        field_inline: output_openings.field_inline.clone(),
        ram_val_check: output_openings.ram_val_check.clone(),
    };
    let public = Stage4PublicOutput {
        challenges: challenges.clone(),
        batching_coefficients: batching_coefficients.to_vec(),
        registers_gamma: prefix.registers_gamma,
        field_registers_gamma: prefix.field_registers_gamma,
        ram_val_check_gamma: prefix.ram_val_check_gamma,
    };
    let verifier_output = Stage4ClearOutput {
        public: public.clone(),
        output_claims: claims,
        ram_val_check_init: stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init),
        batch: VerifiedStage4Batch {
            batching_coefficients: batching_coefficients.to_vec(),
            sumcheck_point: Point::high_to_low(challenges.clone()),
            sumcheck_final_claim: running_claim,
            expected_final_claim,
            registers_read_write: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.registers_read_write,
                sumcheck_point: registers_read_write_sumcheck_point.clone(),
                opening_point: registers_read_write_opening_point.clone(),
                expected_output_claim: registers_expected,
            },
            field_registers_read_write: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.field_registers_read_write,
                sumcheck_point: field_registers_read_write_sumcheck_point.clone(),
                opening_point: field_registers_opening.opening_point.clone(),
                expected_output_claim: field_registers_expected,
            },
            ram_val_check: VerifiedStage4Sumcheck {
                input_claim: prefix.input_claims.ram_val_check,
                sumcheck_point: ram_val_check_sumcheck_point.clone(),
                opening_point: ram_val_check_opening_point.clone(),
                expected_output_claim: ram_expected,
            },
        },
    };
    let built = builder.finish(&output_claim_values, transcript)?;
    Ok(Stage4CommittedBoundaryOutput {
        stage4_sumcheck_proof: built.proof,
        public,
        verifier_output,
        output_claim_values,
        committed_witness: built.witness,
    })
}

fn stage4_register_rows<W>(witness: &W) -> Result<Vec<SumcheckRegistersReadWriteRow>, ProverError>
where
    W: JoltVmRegisterReadWriteRows,
{
    Ok(witness
        .register_read_write_rows()?
        .into_iter()
        .map(stage4_register_row)
        .collect())
}

fn stage4_register_row(row: JoltVmRegisterReadWriteRow) -> SumcheckRegistersReadWriteRow {
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
fn stage4_field_register_rows<F, FI>(
    witness: &FI,
) -> Result<Vec<SumcheckFieldRegistersReadWriteRow<F>>, ProverError>
where
    F: Field,
    FI: FieldInlineRegisterReadWriteRows<F>,
{
    Ok(witness
        .field_inline_register_read_write_rows()?
        .into_iter()
        .map(stage4_field_register_row)
        .collect())
}

#[cfg(feature = "field-inline")]
fn stage4_field_register_row<F: Field>(
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

fn stage4_ram_rows<W>(witness: &W) -> Result<Vec<SumcheckRamReadWriteRow>, ProverError>
where
    W: JoltVmStage2Rows,
{
    Ok(witness
        .stage2_rows()?
        .into_iter()
        .map(stage4_ram_row)
        .collect())
}

fn stage4_ram_row(row: JoltVmStage2TraceRow) -> SumcheckRamReadWriteRow {
    SumcheckRamReadWriteRow {
        remapped_ram_address: row.remapped_ram_address,
        ram_read_value: row.ram_read_value,
        ram_write_value: row.ram_write_value,
        ram_increment: row.ram_increment,
    }
}

pub fn append_stage4_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage4RegularBatchOutputOpeningClaims<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if let Some(opening_claim) = claims.advice.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    if let Some(opening_claim) = claims.advice.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.registers_val);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rs1_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rs2_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rd_wa);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rd_inc);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_read_write;
        transcript.append_labeled(b"opening_claim", &field_claims.field_registers_val);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rs1_ra);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rs2_ra);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_wa);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_inc);
    }
    transcript.append_labeled(b"opening_claim", &claims.ram_val_check.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.ram_val_check.ram_inc);
}

#[cfg(feature = "zk")]
fn stage4_committed_output_claim_values<F: Field>(
    claims: &Stage4RegularBatchOutputOpeningClaims<F>,
) -> Vec<F> {
    let mut values = Vec::with_capacity(stage4_committed_output_claim_capacity(claims));
    if let Some(opening_claim) = claims.advice.untrusted {
        values.push(opening_claim);
    }
    if let Some(opening_claim) = claims.advice.trusted {
        values.push(opening_claim);
    }
    values.push(claims.registers_read_write.registers_val);
    values.push(claims.registers_read_write.rs1_ra);
    values.push(claims.registers_read_write.rs2_ra);
    values.push(claims.registers_read_write.rd_wa);
    values.push(claims.registers_read_write.rd_inc);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_read_write;
        values.push(field_claims.field_registers_val);
        values.push(field_claims.field_rs1_ra);
        values.push(field_claims.field_rs2_ra);
        values.push(field_claims.field_rd_wa);
        values.push(field_claims.field_rd_inc);
    }
    values.push(claims.ram_val_check.ram_ra);
    values.push(claims.ram_val_check.ram_inc);
    values
}

#[cfg(feature = "zk")]
fn stage4_committed_output_claim_capacity<F: Field>(
    claims: &Stage4RegularBatchOutputOpeningClaims<F>,
) -> usize {
    7 + usize::from(claims.advice.untrusted.is_some())
        + usize::from(claims.advice.trusted.is_some())
        + {
            #[cfg(feature = "field-inline")]
            {
                5
            }
            #[cfg(not(feature = "field-inline"))]
            {
                0
            }
        }
}

fn validate_stage4_dependencies<F: Field>(
    config: Stage4ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
) -> Result<(), ProverError> {
    let expected_ram_read_write_vars = config.log_k + config.log_t;
    if stage2.batch.ram_read_write.opening_point.len() != expected_ram_read_write_vars {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 RAM read-write opening point has {} variables, expected {expected_ram_read_write_vars}",
                stage2.batch.ram_read_write.opening_point.len()
            ),
        });
    }
    if stage2.batch.ram_output_check.opening_point.len() != config.log_k {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 RAM output-check opening point has {} variables, expected {}",
                stage2.batch.ram_output_check.opening_point.len(),
                config.log_k
            ),
        });
    }
    let (ram_address_point, _) = stage2
        .batch
        .ram_read_write
        .opening_point
        .split_at(config.log_k);
    if stage2.batch.ram_output_check.opening_point != ram_address_point {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 4 RAM value-check dependencies use different address opening points"
                .to_owned(),
        });
    }
    if stage3.output_claims.registers_claim_reduction.rs1_value
        != stage3.output_claims.instruction_input.rs1_value
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 4 register dependencies disagree on rs1 value".to_owned(),
        });
    }
    if stage3.output_claims.registers_claim_reduction.rs2_value
        != stage3.output_claims.instruction_input.rs2_value
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 4 register dependencies disagree on rs2 value".to_owned(),
        });
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn validate_stage4_committed_checked(
    config: Stage4ProverConfig,
    checked: &jolt_verifier::CheckedInputs,
) -> Result<(), ProverError> {
    if !checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 4 committed-boundary prover received transparent checked inputs"
                .to_owned(),
        });
    }
    if checked.trace_length != (1usize << config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 checked trace length {} does not match log_t {}",
                checked.trace_length, config.log_t
            ),
        });
    }
    if checked.ram_K != (1usize << config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 checked RAM K {} does not match log_k {}",
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

fn append_ram_val_check_gamma_domain_separator<T: Transcript>(transcript: &mut T) {
    transcript.append(&LabelWithCount(b"ram_val_check_gamma", 0));
    transcript.append_bytes(&[]);
}
