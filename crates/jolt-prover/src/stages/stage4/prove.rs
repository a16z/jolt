use jolt_backends::{
    ram_read_write_rows, register_read_write_rows, Stage4ReadWriteSumcheckBackend, SumcheckBackend,
    SumcheckRamValCheckStateRequest, SumcheckRegistersReadWriteStateRequest,
};
use jolt_claims::protocols::jolt::{
    formulas::dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
    JoltAdviceKind, JoltReadWriteConfig,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::{OpeningClaim, OutputClaims, SumcheckInstance};
use jolt_verifier::stages::stage4::outputs::{
    RamValCheckInitialEvaluation, Stage4Challenges, Stage4ClearOutput,
    VerifiedRamValCheckAdviceContribution,
};
use jolt_verifier::stages::stage4::{
    append_ram_val_check_gamma_domain_separator, stage4_expected_final_claim,
    stage4_output_claims_with_points, RamValCheck, RamValCheckAdviceClaims, RamValCheckInputClaims,
    RamValCheckOutputClaims, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
    RegistersReadWriteOutputClaims, Stage4OutputClaims,
};
use jolt_verifier::stages::{stage2::Stage2ClearOutput, stage3::Stage3ClearOutput};
use jolt_verifier::CheckedInputs;
use jolt_witness::protocols::jolt_vm::{JoltVmRegisterReadWriteRows, JoltVmStage2Rows};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::stages::invalid_sumcheck_output;
#[cfg(feature = "zk")]
use crate::stages::recorder::CommittedSumcheckRecorder;
use crate::stages::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::ProverError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4ProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
}

impl Stage4ProverConfig {
    pub const fn new(log_t: usize, log_k: usize, rw_config: JoltReadWriteConfig) -> Self {
        Self {
            log_t,
            log_k,
            rw_config,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage4ProverInput<'a, F: Field, W> {
    pub config: Stage4ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3ClearOutput<F>,
    pub ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage4ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage4ProverConfig,
        checked: &'a CheckedInputs,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage2,
            stage3,
            ram_val_check_init,
            witness,
        }
    }
}

/// The two stage 4 batch input claims, in canonical batch order (registers
/// read-write, then RAM value-check). Computed once via the shared relation
/// objects and reused for the sumcheck setup and the verifier output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage4InputClaims<F: Field> {
    registers_read_write: F,
    ram_val_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage4RegularBatchPrefixOutput<F: Field> {
    input_claims: Stage4InputClaims<F>,
    registers_gamma: F,
    ram_val_check_gamma: F,
    ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    pub advice_contributions: Vec<Stage4RamValCheckAdviceContribution<F>>,
    pub full_eval: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage4RegularBatchProofOutput<F: Field, C> {
    prefix: Stage4RegularBatchPrefixOutput<F>,
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
    output_openings: Stage4OutputClaims<F>,
    registers_read_write_opening_point: Vec<F>,
    ram_val_check_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ProofComponent<F: Field, Proof> {
    pub stage4_sumcheck_proof: Proof,
    pub claims: Stage4OutputClaims<F>,
    pub verifier_output: Stage4ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage4_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub challenges: Stage4Challenges<F>,
    pub verifier_output: Stage4ClearOutput<F>,
    pub output_claim_values: Vec<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

fn stage4_advice_claims_from_prefix<F: Field>(
    prefix: &Stage4RegularBatchPrefixOutput<F>,
) -> Result<RamValCheckAdviceClaims<F>, ProverError> {
    let mut untrusted = None;
    let mut trusted = None;
    for contribution in &prefix.ram_val_check_init.advice_contributions {
        let target = match contribution.kind {
            JoltAdviceKind::Trusted => &mut trusted,
            JoltAdviceKind::Untrusted => &mut untrusted,
        };
        if target.replace(contribution.opening_claim).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 4 RAM value-check {:?} advice contribution",
                contribution.kind
            )));
        }
    }
    Ok(RamValCheckAdviceClaims { untrusted, trusted })
}

/// Canonical Stage 4 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage4/verify.rs` in prover order: derive
/// the register/RAM value-check gammas (with the `ram_val_check_gamma` domain
/// separator), prove the registers read-write + RAM value-check batched
/// sumcheck, evaluate output openings, and assemble the verifier-owned
/// `stage4_sumcheck_proof`, [`Stage4OutputClaims`], and [`Stage4ClearOutput`] for
/// Stage 5 and later stages.
///
/// `ram_val_check_init` is supplied via the input bundle; computing it prover-side
/// from preprocessing and the advice witness is a tracked self-containment
/// follow-up. The ZK Stage 4 prover path is not yet implemented.
pub fn prove<F, W, B, T, C>(
    input: Stage4ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage4ProofComponent<F, SumcheckProof<F, C>>, ProverError>
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

    let claims = Stage4OutputClaims {
        advice: proof_output.output_openings.advice.clone(),
        program_image_contribution: None,
        registers_read_write: proof_output.output_openings.registers_read_write.clone(),
        ram_val_check: proof_output.output_openings.ram_val_check.clone(),
    };
    let challenges = Stage4Challenges {
        registers_gamma: prefix.registers_gamma,
        ram_val_check_gamma: prefix.ram_val_check_gamma,
    };
    let ram_val_check_init = stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init);
    let output_claims = stage4_output_claims_with_points(
        &claims,
        &proof_output.registers_read_write_opening_point,
        &proof_output.ram_val_check_opening_point,
        &ram_val_check_init,
    );
    let verifier_output = Stage4ClearOutput {
        challenges,
        output_claims,
        ram_val_check_init,
    };

    Ok(Stage4ProofComponent {
        stage4_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage4ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage4CommittedProofComponent<F, VC>, ProverError>
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

fn stage4_ram_val_check_init_to_verifier<F: Field>(
    init: &Stage4RamValCheckInitialEvaluation<F>,
) -> RamValCheckInitialEvaluation<F> {
    RamValCheckInitialEvaluation {
        public_eval: init.public_eval,
        program_image_contribution: None,
        advice_contributions: init
            .advice_contributions
            .iter()
            .map(|contribution| VerifiedRamValCheckAdviceContribution {
                kind: contribution.kind,
                selector: contribution.selector,
                opening: OpeningClaim {
                    point: contribution.opening_point.clone(),
                    value: contribution.opening_claim,
                },
            })
            .collect(),
        full_eval: init.full_eval,
    }
}

fn derive_stage4_regular_batch_prefix<F, T>(
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

    append_ram_val_check_gamma_domain_separator(transcript);
    let ram_val_check_gamma = transcript.challenge_scalar();

    let register_dimensions = config
        .rw_config
        .register_dimensions(config.log_t, REGISTER_ADDRESS_BITS);
    let verifier_init = stage4_ram_val_check_init_to_verifier(&ram_val_check_init);
    let registers_relation = RegistersReadWriteChecking::new(register_dimensions, registers_gamma);
    let ram_relation = RamValCheck::new(
        TraceDimensions::new(config.log_t),
        config.log_k,
        ram_val_check_gamma,
        verifier_init.decomposition(),
    );
    let input_claims = Stage4InputClaims {
        registers_read_write: registers_relation
            .input_claim(&RegistersReadWriteInputClaims::from_upstream(stage3))
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?,
        ram_val_check: ram_relation
            .input_claim(&RamValCheckInputClaims::from_upstream(
                stage2,
                &verifier_init,
            ))
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?,
    };

    Ok(Stage4RegularBatchPrefixOutput {
        input_claims,
        registers_gamma,
        ram_val_check_gamma,
        ram_val_check_init,
    })
}

fn stage4_expected_batch_final_claim<F: Field>(
    coefficients: &[F],
    registers_read_write: F,
    ram_val_check: F,
) -> Result<F, ProverError> {
    stage4_expected_final_claim(coefficients, registers_read_write, ram_val_check)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
}

fn prove_stage4_transparent_sumchecks<F, W, B, T, C>(
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
    Ok(proof_output)
}

#[derive(Clone, Copy)]
struct Stage4RegularBatchRunInput<'a, F: Field> {
    config: Stage4ProverConfig,
    stage2: &'a Stage2ClearOutput<F>,
    stage3: &'a Stage3ClearOutput<F>,
    prefix: &'a Stage4RegularBatchPrefixOutput<F>,
}

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
    prove_stage4_specialized_regular_batch_sumcheck_with_recorder(
        Stage4RegularBatchRunInput {
            config,
            stage2,
            stage3,
            prefix,
        },
        witness,
        backend,
        transcript,
        ClearSumcheckRecorder::<F, C>::new(0),
    )
}

fn prove_stage4_specialized_regular_batch_sumcheck_with_recorder<F, W, B, T, S>(
    input: Stage4RegularBatchRunInput<'_, F>,
    witness: &W,
    backend: &mut B,
    transcript: &mut T,
    mut proof_recorder: S,
) -> Result<Stage4RegularBatchProofOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: SumcheckRecorder<F>,
{
    let config = input.config;
    let stage2 = input.stage2;
    let stage3 = input.stage3;
    let prefix = input.prefix;
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
        register_read_write_rows(witness)?,
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
        ram_read_write_rows(witness)?,
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

    proof_recorder.absorb_input_claims(
        &[
            prefix.input_claims.registers_read_write,
            prefix.input_claims.ram_val_check,
        ],
        transcript,
    );
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

        let challenge = proof_recorder.absorb_round(&round_poly, transcript)?;
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

    let verifier_init = stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init);
    let registers_relation =
        RegistersReadWriteChecking::new(register_dimensions, prefix.registers_gamma);
    let ram_relation = RamValCheck::new(
        TraceDimensions::new(config.log_t),
        config.log_k,
        prefix.ram_val_check_gamma,
        verifier_init.decomposition(),
    );
    let registers_inputs = RegistersReadWriteInputClaims::from_upstream(stage3);
    let ram_inputs = RamValCheckInputClaims::from_upstream(stage2, &verifier_init);

    let registers_read_write_opening_point = registers_relation
        .derive_opening_points(&challenges, &registers_inputs)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?
        .registers_val;
    let ram_val_check_opening_point = ram_relation
        .derive_opening_points(&challenges[ram_offset..], &ram_inputs)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?
        .ram_ra;

    let registers_output = backend.output_sumcheck_registers_read_write_state(
        &register_state,
        &registers_read_write_opening_point,
    )?;
    let ram_output = backend.output_sumcheck_ram_val_check_state(&ram_state)?;
    let output_openings = Stage4OutputClaims {
        advice: stage4_advice_claims_from_prefix(prefix)?,
        program_image_contribution: None,
        registers_read_write: RegistersReadWriteOutputClaims {
            registers_val: registers_output.registers_val,
            rs1_ra: registers_output.rs1_ra,
            rs2_ra: registers_output.rs2_ra,
            rd_wa: registers_output.rd_wa,
            rd_inc: registers_output.rd_inc,
        },
        ram_val_check: RamValCheckOutputClaims {
            ram_ra: ram_output.ram_ra,
            ram_inc: ram_output.ram_inc,
        },
    };

    // Pair the produced openings with their points via the shared helper for the
    // output-algebra check — the same form the verifier builds.
    let verifier_init = stage4_ram_val_check_init_to_verifier(&prefix.ram_val_check_init);
    let claims_with_points = stage4_output_claims_with_points(
        &output_openings,
        &registers_read_write_opening_point,
        &ram_val_check_opening_point,
        &verifier_init,
    );
    let registers_expected = registers_relation
        .expected_output(&registers_inputs, &claims_with_points.registers_read_write)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let ram_expected = ram_relation
        .expected_output(&ram_inputs, &claims_with_points.ram_val_check)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let expected_final_claim = stage4_expected_batch_final_claim(
        &batching_coefficients,
        registers_expected,
        ram_expected,
    )?;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 4 batch final claim did not match output openings",
        ));
    }
    let output_claim_values = output_openings.opening_values();
    let recorded = proof_recorder.finish(&output_claim_values, transcript)?;

    Ok(Stage4RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: recorded.proof,
        #[cfg(feature = "zk")]
        committed_witness: recorded.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: recorded.output_claim_values,
        output_openings,
        registers_read_write_opening_point,
        ram_val_check_opening_point,
    })
}

#[cfg(feature = "zk")]
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
) -> Result<Stage4CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage2Rows + JoltVmRegisterReadWriteRows,
    B: Stage4ReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let batch = prove_stage4_specialized_regular_batch_sumcheck_with_recorder(
        Stage4RegularBatchRunInput {
            config,
            stage2,
            stage3,
            prefix,
        },
        witness,
        backend,
        transcript,
        CommittedSumcheckRecorder::<F, VC>::new(vc_setup)?,
    )?;

    let claims = Stage4OutputClaims {
        advice: batch.output_openings.advice.clone(),
        program_image_contribution: None,
        registers_read_write: batch.output_openings.registers_read_write.clone(),
        ram_val_check: batch.output_openings.ram_val_check.clone(),
    };
    let challenges = Stage4Challenges {
        registers_gamma: batch.prefix.registers_gamma,
        ram_val_check_gamma: batch.prefix.ram_val_check_gamma,
    };
    let ram_val_check_init =
        stage4_ram_val_check_init_to_verifier(&batch.prefix.ram_val_check_init);
    let output_claims = stage4_output_claims_with_points(
        &claims,
        &batch.registers_read_write_opening_point,
        &batch.ram_val_check_opening_point,
        &ram_val_check_init,
    );
    let verifier_output = Stage4ClearOutput {
        challenges: challenges.clone(),
        output_claims,
        ram_val_check_init,
    };
    Ok(Stage4CommittedProofComponent {
        stage4_sumcheck_proof: batch.proof,
        challenges,
        verifier_output,
        output_claim_values: batch.output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 4 committed output claim values are missing".to_owned())
        })?,
        committed_witness: batch.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 4 committed witness material is missing".to_owned())
        })?,
    })
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
            reason: "Stage 4 committed proof component prover received transparent checked inputs"
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
