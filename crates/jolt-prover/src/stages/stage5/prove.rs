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
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::SumcheckInstance;
use jolt_verifier::stages::stage5::outputs::{Stage5Challenges, Stage5ClearOutput};
use jolt_verifier::stages::stage5::{
    stage5_expected_final_claim, stage5_output_claims_with_points, InstructionReadRaf,
    InstructionReadRafInputClaims, InstructionReadRafOutputClaims, RamRaClaimReduction,
    RamRaClaimReductionInputClaims, RamRaClaimReductionOutputClaims, RegistersValEvaluation,
    RegistersValEvaluationInputClaims, RegistersValEvaluationOutputClaims, Stage5OutputClaims,
};
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage4::Stage4ClearOutput};
use jolt_verifier::{CheckedInputs, VerifierError};
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

/// The three stage 5 batch input claims, in canonical batch order (instruction
/// read-RAF, RAM-RA reduction, register value-evaluation). Computed once via the
/// shared relation objects so prover and verifier derive them identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage5InputClaims<F: Field> {
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
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
    pub claims: Stage5OutputClaims<F>,
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
    pub challenges: Stage5Challenges<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage5ClearOutput<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage5RegularBatchProofOutput<F: Field, C> {
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
    claims: Stage5OutputClaims<F>,
    verifier_output: Stage5ClearOutput<F>,
}

const STAGE5_BATCH_COEFFICIENTS: usize = 3;

#[derive(Clone, Copy)]
struct Stage5BatchCoefficients<F: Field> {
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
}

/// Build the three stage 5 relation objects from the batch gammas. Shared by the
/// prefix (input claims) and the batch sumcheck (opening points, expected output)
/// so prover and verifier drive identical relations.
fn stage5_relations<F: Field>(
    config: Stage5ProverConfig,
    instruction_gamma: F,
    ram_gamma: F,
) -> (
    InstructionReadRaf<F>,
    RamRaClaimReduction<F>,
    RegistersValEvaluation<F>,
) {
    let trace_dimensions = TraceDimensions::new(config.log_t);
    (
        InstructionReadRaf::new(config.instruction_read_raf_dimensions, instruction_gamma),
        RamRaClaimReduction::new(trace_dimensions, config.log_k, ram_gamma),
        RegistersValEvaluation::new(trace_dimensions),
    )
}

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
    let proof_output = prove_stage5_specialized_regular_batch_sumcheck_with_recorder(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage4,
        &prefix,
        transcript,
        ClearSumcheckRecorder::<F, C>::new(0),
    )?;

    Ok(Stage5ProofComponent {
        stage5_sumcheck_proof: proof_output.proof,
        claims: proof_output.claims,
        verifier_output: proof_output.verifier_output,
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
    let output = prove_stage5_specialized_regular_batch_sumcheck_with_recorder(
        input.config,
        input.witness,
        backend,
        input.stage2,
        input.stage4,
        &prefix,
        transcript,
        CommittedSumcheckRecorder::<F, VC>::new(vc_setup)?,
    )?;

    let challenges = output.verifier_output.challenges.clone();
    Ok(Stage5CommittedProofComponent {
        stage5_sumcheck_proof: output.proof,
        challenges,
        output_claim_values: output.output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 5 committed output claim values are missing")
        })?,
        verifier_output: output.verifier_output,
        committed_witness: output.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 5 committed witness material is missing")
        })?,
    })
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

    let (instruction_relation, ram_relation, registers_relation) =
        stage5_relations(config, instruction_gamma, ram_gamma);
    let to_prover_error = |error: VerifierError| invalid_sumcheck_output(error.to_string());
    let input_claims = Stage5InputClaims {
        instruction_read_raf: instruction_relation
            .input_claim(&InstructionReadRafInputClaims::from_upstream(stage2))
            .map_err(to_prover_error)?,
        ram_ra_claim_reduction: ram_relation
            .input_claim(&RamRaClaimReductionInputClaims::from_upstream(stage2, stage4))
            .map_err(to_prover_error)?,
        registers_val_evaluation: registers_relation
            .input_claim(&RegistersValEvaluationInputClaims::from_upstream(stage4))
            .map_err(to_prover_error)?,
    };

    Ok(Stage5RegularBatchPrefixOutput {
        input_claims,
        instruction_gamma,
        ram_gamma,
    })
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
    let (instruction_relation, ram_relation, registers_relation) =
        stage5_relations(config, prefix.instruction_gamma, prefix.ram_gamma);
    let to_prover_error = |error: VerifierError| invalid_sumcheck_output(error.to_string());

    let instruction_layout = config
        .instruction_read_raf_dimensions
        .u128_address_layout()
        .map_err(invalid_sumcheck_output)?;

    let instruction_rows = instruction_read_raf_rows(witness, config.log_t)?;
    let instruction_request = SumcheckInstructionReadRafStateRequest::new(
        "stage5.instruction_read_raf",
        instruction_rows,
        stage2
            .output_claims
            .instruction_claim_reduction_point()
            .to_vec(),
        prefix.instruction_gamma,
        prefix.input_claims.instruction_read_raf,
        config.log_t,
        instruction_layout.address_bits(),
        instruction_layout.virtual_ra_chunk_bits(),
        instruction_sumcheck_phases(config.log_t),
    )
    .with_optimization_ids(&["cpu_stage5_regular_batch_sumcheck"]);

    // Split the upstream RAM/register opening points into the fixed address prefix
    // and fixed cycle suffixes the backend witness materialization needs. The
    // address-agreement and length invariants are re-checked by each relation's
    // `derive_opening_points` when the output openings are paired with points below.
    let ram_read_write_opening_point = stage2.output_claims.ram_read_write_point();
    let ram_raf_opening_point = stage2.output_claims.ram_raf_evaluation_point();
    let ram_val_check_opening_point = stage4.output_claims.ram_val_check.ram_ra.point.as_slice();
    let registers_opening_point = stage4
        .output_claims
        .registers_read_write
        .registers_val
        .point
        .as_slice();
    let ram_address_point = &ram_read_write_opening_point[..config.log_k];
    let ram_raf_fixed_cycle_point = &ram_raf_opening_point[config.log_k..];
    let ram_read_write_fixed_cycle_point = &ram_read_write_opening_point[config.log_k..];
    let ram_val_check_fixed_cycle_point = &ram_val_check_opening_point[config.log_k..];
    let (register_address_point, register_fixed_cycle_point) =
        registers_opening_point.split_at(REGISTER_ADDRESS_BITS);

    let ram_rows = ram_read_write_rows(witness)?;
    let ram_request = SumcheckRamRaClaimReductionStateRequest::new(
        "stage5.ram_ra_claim_reduction",
        ram_rows,
        ram_address_point.to_vec(),
        ram_raf_fixed_cycle_point.to_vec(),
        ram_read_write_fixed_cycle_point.to_vec(),
        ram_val_check_fixed_cycle_point.to_vec(),
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
        register_address_point.to_vec(),
        register_fixed_cycle_point.to_vec(),
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

    let instruction_inputs = InstructionReadRafInputClaims::from_upstream(stage2);
    let ram_inputs = RamRaClaimReductionInputClaims::from_upstream(stage2, stage4);
    let registers_inputs = RegistersValEvaluationInputClaims::from_upstream(stage4);
    let points = Stage5OutputClaims {
        instruction_read_raf: instruction_relation
            .derive_opening_points(&sumcheck_point, &instruction_inputs)
            .map_err(to_prover_error)?,
        ram_ra_claim_reduction: ram_relation
            .derive_opening_points(&sumcheck_point[front_padding_rounds..], &ram_inputs)
            .map_err(to_prover_error)?,
        registers_val_evaluation: registers_relation
            .derive_opening_points(&sumcheck_point[front_padding_rounds..], &registers_inputs)
            .map_err(to_prover_error)?,
    };

    let instruction_output =
        backend.output_sumcheck_instruction_read_raf_state(&instruction_state)?;
    let instruction_internal_final = instruction_output.final_claim;
    let ram_output = backend.output_sumcheck_ram_ra_claim_reduction_state(&ram_state)?;
    let registers_output =
        backend.output_sumcheck_registers_val_evaluation_state(&registers_state)?;
    let output_openings = Stage5OutputClaims {
        instruction_read_raf: InstructionReadRafOutputClaims {
            lookup_table_flags: instruction_output.lookup_table_flags,
            instruction_ra: instruction_output.instruction_ra,
            instruction_raf_flag: instruction_output.instruction_raf_flag,
        },
        ram_ra_claim_reduction: RamRaClaimReductionOutputClaims {
            ram_ra: ram_output.ram_ra,
        },
        registers_val_evaluation: RegistersValEvaluationOutputClaims {
            rd_inc: registers_output.rd_inc,
            rd_wa: registers_output.rd_wa,
        },
    };

    let output_claims = stage5_output_claims_with_points(&output_openings, &points);
    let instruction_expected = instruction_relation
        .expected_output(&instruction_inputs, &output_claims.instruction_read_raf)
        .map_err(to_prover_error)?;
    let ram_expected = ram_relation
        .expected_output(&ram_inputs, &output_claims.ram_ra_claim_reduction)
        .map_err(to_prover_error)?;
    let registers_expected = registers_relation
        .expected_output(&registers_inputs, &output_claims.registers_val_evaluation)
        .map_err(to_prover_error)?;
    let expected_final_claim = stage5_expected_final_claim(
        &batching_coefficients,
        instruction_expected,
        ram_expected,
        registers_expected,
    )
    .map_err(to_prover_error)?;
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

    let instruction_r_address = output_claims.instruction_read_raf.r_address();
    let verifier_output = Stage5ClearOutput {
        challenges: Stage5Challenges {
            instruction_gamma: prefix.instruction_gamma,
            ram_gamma: prefix.ram_gamma,
        },
        output_claims,
        instruction_r_address,
    };

    let output_claim_values = output_openings.opening_values();
    let recorded = proof_recorder.finish(&output_claim_values, transcript)?;
    Ok(Stage5RegularBatchProofOutput {
        proof: recorded.proof,
        #[cfg(feature = "zk")]
        committed_witness: recorded.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: recorded.output_claim_values,
        claims: output_openings,
        verifier_output,
    })
}

const fn instruction_sumcheck_phases(log_t: usize) -> usize {
    if log_t < INSTRUCTION_PHASES_THRESHOLD_LOG_T {
        16
    } else {
        8
    }
}

/// Enforce the cross-relation opening aliases the instruction read-RAF input
/// wiring relies on: the product-remainder and instruction-claim-reduction share
/// the trace-length opening point and agree on the reduced lookup output. Mirrors
/// the verifier's clear-path check.
fn validate_stage5_dependencies<F: Field>(
    config: Stage5ProverConfig,
    stage2: &Stage2ClearOutput<F>,
) -> Result<(), ProverError> {
    let expected_trace_vars = config.log_t;
    let product_point = stage2.output_claims.product_remainder_point();
    let reduced_point = stage2.output_claims.instruction_claim_reduction_point();
    for (label, point) in [
        ("product remainder", product_point),
        ("instruction claim reduction", reduced_point),
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
    if product_point != reduced_point {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 instruction read-RAF dependencies use different opening points"
                .to_owned(),
        });
    }
    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output.value;
    let reduced_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .as_ref()
        .map_or(product_lookup_output, |claim| claim.value);
    if reduced_lookup_output != product_lookup_output {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 5 instruction read-RAF dependencies disagree on the reduced lookup output"
                .to_owned(),
        });
    }
    Ok(())
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
