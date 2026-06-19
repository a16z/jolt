use common::constants::INSTRUCTION_PHASES_THRESHOLD_LOG_T;
use jolt_backends::{
    instruction_read_raf_rows, ram_read_write_rows, register_read_write_rows,
    Stage5ValueEvaluationSumcheckBackend, SumcheckBackend, SumcheckInstructionReadRafStateRequest,
    SumcheckRamRaClaimReductionStateRequest, SumcheckRegistersValEvaluationStateRequest,
};
use jolt_claims::protocols::jolt::formulas::{
    dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
    instruction::InstructionReadRafDimensions,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::{Point, UnivariatePoly};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage5::inputs::{
    InstructionReadRafOutputOpeningClaims, RamRaClaimReductionOutputOpeningClaims,
    RegistersValEvaluationOutputOpeningClaims, Stage5Claims,
};
use jolt_verifier::stages::stage5::outputs::{
    Stage5ClearOutput, Stage5PublicOutput, VerifiedInstructionReadRafSumcheck, VerifiedStage5Batch,
    VerifiedStage5Sumcheck,
};
use jolt_verifier::stages::stage5::stage5_output_claim_values;
use jolt_verifier::stages::stage5::{
    stage5_dependency_opening_points, stage5_expected_final_claim, stage5_expected_output_claims,
    stage5_input_claims, stage5_instruction_opening_points,
    stage5_instruction_read_raf_dependencies, stage5_value_opening_points,
    Stage5DependencyOpeningPointRequest, Stage5DependencyOpeningPoints, Stage5ExpectedOutputClaims,
    Stage5ExpectedOutputRequest, Stage5InputClaimRequest, Stage5InputClaims,
    Stage5InstructionReadRafDependencyRequest, Stage5ValueOpeningPointRequest,
    Stage5ValueOpeningPoints,
};
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage4::Stage4ClearOutput};
use jolt_verifier::CheckedInputs;
use jolt_witness::protocols::jolt_vm::{
    JoltVmRegisterReadWriteRows, JoltVmStage2Rows, JoltVmStage5InstructionReadRafRows,
};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::stages::invalid_sumcheck_output;
#[cfg(feature = "zk")]
use crate::stages::recorder::CommittedSumcheckRecorder;
use crate::stages::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::ProverError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5ProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub instruction_read_raf_dimensions: InstructionReadRafDimensions,
}

impl Stage5ProverConfig {
    pub const fn new(
        log_t: usize,
        log_k: usize,
        instruction_read_raf_dimensions: InstructionReadRafDimensions,
    ) -> Self {
        Self {
            log_t,
            log_k,
            instruction_read_raf_dimensions,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage5ProverInput<'a, F: Field, W> {
    pub config: Stage5ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage5ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage5ProverConfig,
        checked: &'a CheckedInputs,
        stage2: &'a Stage2ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage2,
            stage4,
            witness,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage5RegularBatchPrefixOutput<F: Field> {
    input_claims: Stage5InputClaims<F>,
    instruction_gamma: F,
    ram_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ProofComponent<F: Field, Proof> {
    pub stage5_sumcheck_proof: Proof,
    pub claims: Stage5Claims<F>,
    pub verifier_output: Stage5ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage5_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage5PublicOutput<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage5ClearOutput<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage5RegularBatchProofOutput<F: Field, C> {
    prefix: Stage5RegularBatchPrefixOutput<F>,
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
    output_openings: Stage5Claims<F>,
    expected_outputs: Stage5ExpectedOutputClaims<F>,
    batching_coefficients: Vec<F>,
    sumcheck_point: Vec<F>,
    sumcheck_final_claim: F,
    expected_final_claim: F,
    instruction_read_raf_sumcheck_point: Vec<F>,
    instruction_read_raf_r_address: Vec<F>,
    instruction_read_raf_r_cycle: Vec<F>,
    instruction_read_raf_full_opening_point: Vec<F>,
    instruction_lookup_table_flag_opening_point: Vec<F>,
    instruction_ra_opening_points: Vec<Vec<F>>,
    instruction_raf_flag_opening_point: Vec<F>,
    ram_ra_claim_reduction_sumcheck_point: Vec<F>,
    ram_ra_claim_reduction_opening_point: Vec<F>,
    registers_val_evaluation_sumcheck_point: Vec<F>,
    registers_val_evaluation_opening_point: Vec<F>,
}
const STAGE5_BATCH_COEFFICIENTS: usize = 3;

/// Canonical Stage 5 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage5/verify.rs` in prover order: derive
/// the instruction/RAM gammas, prove the instruction read-RAF, RAM-RA reduction,
/// and register value-evaluation batched sumcheck, then assemble the
/// verifier-owned Stage 5 proof, claims, and clear output for Stage 6 and later
/// stages. ZK Stage 5 prover assembly is a separate committed proof component
/// path.
pub fn prove<F, W, B, T, C>(
    input: Stage5ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage5ProofComponent<F, SumcheckProof<F, C>>, ProverError>
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

    Ok(Stage5ProofComponent {
        stage5_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage5ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage5CommittedProofComponent<F, VC>, ProverError>
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

fn derive_stage5_regular_batch_prefix<F, T>(
    config: Stage5ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage5RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    validate_stage5_dependencies(config, stage2)?;

    let instruction_gamma = transcript.challenge_scalar();
    let ram_gamma = transcript.challenge_scalar();

    Ok(Stage5RegularBatchPrefixOutput {
        input_claims: stage5_input_claims(Stage5InputClaimRequest {
            stage2,
            stage4,
            trace_dimensions: TraceDimensions::new(config.log_t),
            instruction_read_raf_dimensions: config.instruction_read_raf_dimensions,
            ram_log_k: config.log_k,
            instruction_gamma,
            ram_gamma,
        })
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: error.to_string(),
        })?,
        instruction_gamma,
        ram_gamma,
    })
}

struct Stage5ExpectedOutputInputs<'a, F: Field> {
    config: Stage5ProverConfig,
    prefix: &'a Stage5RegularBatchPrefixOutput<F>,
    instruction_fixed_cycle_point: &'a [F],
    instruction_r_address: &'a [F],
    instruction_r_cycle: &'a [F],
    ram_raf_fixed_cycle_point: &'a [F],
    ram_read_write_fixed_cycle_point: &'a [F],
    ram_val_check_fixed_cycle_point: &'a [F],
    ram_ra_claim_reduction_opening_point: &'a [F],
    registers_fixed_cycle_point: &'a [F],
    registers_val_evaluation_opening_point: &'a [F],
    output_openings: &'a Stage5Claims<F>,
}

fn stage5_expected_outputs<F: Field>(
    input: Stage5ExpectedOutputInputs<'_, F>,
) -> Result<Stage5ExpectedOutputClaims<F>, ProverError> {
    stage5_expected_output_claims(Stage5ExpectedOutputRequest {
        instruction_read_raf_dimensions: input.config.instruction_read_raf_dimensions,
        ram_log_k: input.config.log_k,
        instruction_gamma: input.prefix.instruction_gamma,
        ram_gamma: input.prefix.ram_gamma,
        instruction_fixed_cycle_point: input.instruction_fixed_cycle_point,
        instruction_r_address: input.instruction_r_address,
        instruction_r_cycle: input.instruction_r_cycle,
        ram_raf_fixed_cycle_point: input.ram_raf_fixed_cycle_point,
        ram_read_write_fixed_cycle_point: input.ram_read_write_fixed_cycle_point,
        ram_val_check_fixed_cycle_point: input.ram_val_check_fixed_cycle_point,
        ram_ra_claim_reduction_opening_point: input.ram_ra_claim_reduction_opening_point,
        registers_fixed_cycle_point: input.registers_fixed_cycle_point,
        registers_val_evaluation_opening_point: input.registers_val_evaluation_opening_point,
        claims: input.output_openings,
    })
    .map_err(|error| ProverError::InvalidStageRequest {
        reason: error.to_string(),
    })
}

fn stage5_expected_batch_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage5ExpectedOutputClaims<F>,
) -> Result<F, ProverError> {
    stage5_expected_final_claim(coefficients, expected_outputs)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
}

fn stage5_batch_value_opening_points<F: Field>(
    request: Stage5ValueOpeningPointRequest<'_, F>,
) -> Result<Stage5ValueOpeningPoints<F>, ProverError> {
    stage5_value_opening_points(request).map_err(|error| invalid_sumcheck_output(error.to_string()))
}

fn stage5_dependency_points<'a, F: Field>(
    config: Stage5ProverConfig,
    stage2: &'a Stage2ClearOutput<F>,
    stage4: &'a Stage4ClearOutput<F>,
) -> Result<Stage5DependencyOpeningPoints<'a, F>, ProverError> {
    stage5_dependency_opening_points(Stage5DependencyOpeningPointRequest {
        trace_dimensions: TraceDimensions::new(config.log_t),
        ram_log_k: config.log_k,
        ram_raf_opening_point: &stage2.batch.ram_raf_evaluation.opening_point,
        ram_read_write_opening_point: &stage2.batch.ram_read_write.opening_point,
        ram_val_check_opening_point: &stage4.batch.ram_val_check.opening_point,
        registers_read_write_opening_point: &stage4.batch.registers_read_write.opening_point,
    })
    .map_err(|error| ProverError::InvalidStageRequest {
        reason: error.to_string(),
    })
}

fn prove_stage5_transparent_sumchecks<F, W, B, T, C>(
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
    let proof_output = prove_stage5_specialized_regular_batch_sumcheck_with_recorder(
        config,
        witness,
        backend,
        stage2,
        stage4,
        prefix,
        transcript,
        ClearSumcheckRecorder::<F, C>::new(0),
    )?;
    Ok(proof_output)
}

#[expect(clippy::too_many_arguments)]
fn prove_stage5_specialized_regular_batch_sumcheck_with_recorder<F, W, B, T, S>(
    config: Stage5ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    prefix: &Stage5RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    mut proof_recorder: S,
) -> Result<Stage5RegularBatchProofOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage2Rows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage5ValueEvaluationSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: SumcheckRecorder<F>,
{
    let instruction_layout = config
        .instruction_read_raf_dimensions
        .u128_address_layout()
        .map_err(invalid_sumcheck_output)?;

    let instruction_rows = instruction_read_raf_rows(witness, config.log_t)?;
    let instruction_request = SumcheckInstructionReadRafStateRequest::new(
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
        instruction_layout.address_bits(),
        instruction_layout.virtual_ra_chunk_bits(),
        instruction_sumcheck_phases(config.log_t),
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);

    let dependency_points = stage5_dependency_points(config, stage2, stage4)?;

    let ram_rows = ram_read_write_rows(witness)?;
    let ram_request = SumcheckRamRaClaimReductionStateRequest::new(
        "stage5.ram_ra_claim_reduction",
        ram_rows,
        dependency_points.ram_address_point.to_vec(),
        dependency_points.ram_raf_fixed_cycle_point.to_vec(),
        dependency_points.ram_read_write_fixed_cycle_point.to_vec(),
        dependency_points.ram_val_check_fixed_cycle_point.to_vec(),
        prefix.ram_gamma,
        prefix.input_claims.ram_ra_claim_reduction,
        config.log_t,
        config.log_k,
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);
    let register_rows = register_read_write_rows(witness)?;
    let registers_request = SumcheckRegistersValEvaluationStateRequest::new(
        "stage5.registers_val_evaluation",
        register_rows,
        dependency_points.register_address_point.to_vec(),
        dependency_points.register_fixed_cycle_point.to_vec(),
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

    proof_recorder.absorb_input_claims(
        &[
            prefix.input_claims.instruction_read_raf,
            prefix.input_claims.ram_ra_claim_reduction,
            prefix.input_claims.registers_val_evaluation,
        ],
        transcript,
    );
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

    let front_padding_rounds = instruction_layout.address_bits();
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
                "Stage 5 batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = proof_recorder.absorb_round(&round_poly, transcript)?;
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

    let instruction_points =
        stage5_instruction_opening_points(config.instruction_read_raf_dimensions, &sumcheck_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let value_points = stage5_batch_value_opening_points(Stage5ValueOpeningPointRequest {
        trace_dimensions: TraceDimensions::new(config.log_t),
        ram_log_k: config.log_k,
        ram_ra_claim_reduction_sumcheck_point: &sumcheck_point[front_padding_rounds..],
        registers_val_evaluation_sumcheck_point: &sumcheck_point[front_padding_rounds..],
        ram_raf_opening_point: &stage2.batch.ram_raf_evaluation.opening_point,
        ram_read_write_opening_point: &stage2.batch.ram_read_write.opening_point,
        ram_val_check_opening_point: &stage4.batch.ram_val_check.opening_point,
        registers_read_write_opening_point: &stage4.batch.registers_read_write.opening_point,
    })?;

    let instruction_output =
        backend.output_sumcheck_instruction_read_raf_state(&instruction_state)?;
    let instruction_internal_final = instruction_output.final_claim;
    let ram_output = backend.output_sumcheck_ram_ra_claim_reduction_state(&ram_state)?;
    let registers_output =
        backend.output_sumcheck_registers_val_evaluation_state(&registers_state)?;
    let output_openings = Stage5Claims {
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
    let expected_outputs = stage5_expected_outputs(Stage5ExpectedOutputInputs {
        config,
        prefix,
        instruction_fixed_cycle_point: &stage2.batch.instruction_claim_reduction.opening_point,
        instruction_r_address: &instruction_points.r_address,
        instruction_r_cycle: &instruction_points.r_cycle,
        ram_raf_fixed_cycle_point: &value_points.ram_raf_fixed_cycle_point,
        ram_read_write_fixed_cycle_point: &value_points.ram_read_write_fixed_cycle_point,
        ram_val_check_fixed_cycle_point: &value_points.ram_val_check_fixed_cycle_point,
        ram_ra_claim_reduction_opening_point: &value_points.ram_ra_claim_reduction_opening_point,
        registers_fixed_cycle_point: &value_points.registers_fixed_cycle_point,
        registers_val_evaluation_opening_point: &value_points
            .registers_val_evaluation_opening_point,
        output_openings: &output_openings,
    })?;
    let instruction_expected = expected_outputs.instruction_read_raf;
    let ram_expected = expected_outputs.ram_ra_claim_reduction;
    let registers_expected = expected_outputs.registers_val_evaluation;
    let expected_final_claim =
        stage5_expected_batch_final_claim(&batching_coefficients, &expected_outputs)?;
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

    let output_claim_values = stage5_output_claim_values(&output_openings);
    let recorded = proof_recorder.finish(&output_claim_values, transcript)?;
    Ok(Stage5RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: recorded.proof,
        #[cfg(feature = "zk")]
        committed_witness: recorded.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: recorded.output_claim_values,
        output_openings,
        expected_outputs,
        batching_coefficients,
        sumcheck_point,
        sumcheck_final_claim: running_claim,
        expected_final_claim,
        instruction_read_raf_sumcheck_point: instruction_points.sumcheck_point,
        instruction_read_raf_r_address: instruction_points.r_address,
        instruction_read_raf_r_cycle: instruction_points.r_cycle,
        instruction_read_raf_full_opening_point: instruction_points.full_opening_point,
        instruction_lookup_table_flag_opening_point: instruction_points
            .lookup_table_flag_opening_point,
        instruction_ra_opening_points: instruction_points.instruction_ra_opening_points,
        instruction_raf_flag_opening_point: instruction_points.instruction_raf_flag_opening_point,
        ram_ra_claim_reduction_sumcheck_point: value_points.ram_ra_claim_reduction_sumcheck_point,
        ram_ra_claim_reduction_opening_point: value_points.ram_ra_claim_reduction_opening_point,
        registers_val_evaluation_sumcheck_point: value_points
            .registers_val_evaluation_sumcheck_point,
        registers_val_evaluation_opening_point: value_points.registers_val_evaluation_opening_point,
    })
}

#[cfg(feature = "zk")]
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
) -> Result<Stage5CommittedProofComponent<F, VC>, ProverError>
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
    let output = prove_stage5_specialized_regular_batch_sumcheck_with_recorder(
        config,
        witness,
        backend,
        stage2,
        stage4,
        prefix,
        transcript,
        CommittedSumcheckRecorder::<F, VC>::new(vc_setup)?,
    )?;

    let public = Stage5PublicOutput {
        challenges: output.sumcheck_point.clone(),
        batching_coefficients: output.batching_coefficients.clone(),
        instruction_gamma: prefix.instruction_gamma,
        ram_gamma: prefix.ram_gamma,
    };
    let verifier_output = Stage5ClearOutput {
        public: public.clone(),
        output_claims: output.output_openings,
        batch: VerifiedStage5Batch {
            batching_coefficients: public.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(public.challenges.clone()),
            sumcheck_final_claim: output.sumcheck_final_claim,
            expected_final_claim: output.expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: prefix.input_claims.instruction_read_raf,
                sumcheck_point: output.instruction_read_raf_sumcheck_point,
                r_address: output.instruction_read_raf_r_address,
                r_cycle: output.instruction_read_raf_r_cycle,
                full_opening_point: output.instruction_read_raf_full_opening_point,
                lookup_table_flag_opening_point: output.instruction_lookup_table_flag_opening_point,
                instruction_ra_opening_points: output.instruction_ra_opening_points,
                instruction_raf_flag_opening_point: output.instruction_raf_flag_opening_point,
                expected_output_claim: output.expected_outputs.instruction_read_raf,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.ram_ra_claim_reduction,
                sumcheck_point: output.ram_ra_claim_reduction_sumcheck_point,
                opening_point: output.ram_ra_claim_reduction_opening_point,
                expected_output_claim: output.expected_outputs.ram_ra_claim_reduction,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: prefix.input_claims.registers_val_evaluation,
                sumcheck_point: output.registers_val_evaluation_sumcheck_point,
                opening_point: output.registers_val_evaluation_opening_point,
                expected_output_claim: output.expected_outputs.registers_val_evaluation,
            },
        },
    };
    Ok(Stage5CommittedProofComponent {
        stage5_sumcheck_proof: output.proof,
        public,
        output_claim_values: output.output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 5 committed output claim values are missing")
        })?,
        verifier_output,
        committed_witness: output.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 5 committed witness material is missing")
        })?,
    })
}

const fn instruction_sumcheck_phases(log_t: usize) -> usize {
    if log_t < INSTRUCTION_PHASES_THRESHOLD_LOG_T {
        16
    } else {
        8
    }
}

fn validate_stage5_dependencies<F: Field>(
    config: Stage5ProverConfig,
    stage2: &Stage2ClearOutput<F>,
) -> Result<(), ProverError> {
    stage5_instruction_read_raf_dependencies(Stage5InstructionReadRafDependencyRequest {
        trace_dimensions: TraceDimensions::new(config.log_t),
        stage2,
    })
    .map_err(|error| ProverError::InvalidStageRequest {
        reason: error.to_string(),
    })
}

#[derive(Clone, Copy)]
struct Stage5BatchCoefficients<F: Field> {
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
}

#[cfg(feature = "zk")]
fn validate_stage5_committed_checked(
    config: Stage5ProverConfig,
    checked: &jolt_verifier::CheckedInputs,
) -> Result<(), ProverError> {
    if !checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 committed proof component prover received transparent checked inputs"
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
