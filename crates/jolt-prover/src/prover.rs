#![expect(
    clippy::too_many_arguments,
    reason = "generated prover helpers mirror staged protocol ABIs"
)]

use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_field::Fr;
use jolt_kernels::{stage1, stage2, stage3, stage4, stage5, stage6, stage7};
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};
use jolt_verifier::{JoltEvaluationProof, JoltNamedEval, JoltProof, JoltStage2RamAccess, JoltStage2RamData, JoltStage2RamOutputLayout, JoltStage6BytecodeEntry, JoltStage6BytecodeReadRafData, JoltStage6VerifierData, JoltStageChallengeVector, JoltStageExecutionArtifacts, JoltStageOpeningInputValue, JoltStageProof, JoltSumcheckOutput};
use jolt_witness::{stage4_ram_val_init_opening, CycleInput, Stage45SparseTraceWitness, Stage6BytecodeEntry as WitnessStage6BytecodeEntry, Stage6WitnessParams, Stage6WitnessPolynomials, Stage6WitnessSlices};
use rayon::prelude::*;

use crate::stages::{commitment as commitment_stage, stage1_outer as stage1_outer_stage, stage2 as stage2_stage, stage3 as stage3_stage, stage4 as stage4_stage, stage5 as stage5_stage, stage6 as stage6_stage, stage7 as stage7_stage, stage8 as stage8_stage};

pub type DefaultJoltTranscript = Blake2bTranscript<Fr>;

pub struct JoltProverInputs<'a, CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor, Stage4Executor, Stage5Executor, Stage6Executor, Stage7Executor> {
    pub commitment_inputs: &'a mut CommitmentInputs,
    pub prover_setup: &'a DoryProverSetup,
    pub stage1_outer_executor: &'a mut Stage1OuterExecutor,
    pub stage2_executor: &'a mut Stage2Executor,
    pub stage3_executor: &'a mut Stage3Executor,
    pub stage4_executor: &'a mut Stage4Executor,
    pub stage5_executor: &'a mut Stage5Executor,
    pub stage6_executor: &'a mut Stage6Executor,
    pub stage7_executor: &'a mut Stage7Executor,
    pub stage7_openings: Option<&'a [stage7::Stage7OpeningInputValue<Fr>]>,
}

#[derive(Clone, Copy, Debug)]
pub struct JoltProverPrograms {
    pub commitment: &'static commitment_stage::CommitmentProverProgramPlan,
    pub stage1_outer: &'static stage1::Stage1CpuProgramPlan,
    pub stage2: &'static stage2::Stage2CpuProgramPlan,
    pub stage3: &'static stage3::Stage3CpuProgramPlan,
    pub stage4: &'static stage4::Stage4CpuProgramPlan,
    pub stage5: &'static stage5::Stage5CpuProgramPlan,
    pub stage6: &'static stage6::Stage6CpuProgramPlan,
    pub stage7: &'static stage7::Stage7CpuProgramPlan,
    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,
}

pub fn default_prover_programs() -> JoltProverPrograms {
    JoltProverPrograms {
        commitment: &commitment_stage::COMMITMENT_PROGRAM,
        stage1_outer: &stage1_outer_stage::STAGE1_PROGRAM,
        stage2: &stage2_stage::STAGE2_PROGRAM,
        stage3: &stage3_stage::STAGE3_PROGRAM,
        stage4: &stage4_stage::STAGE4_PROGRAM,
        stage5: &stage5_stage::STAGE5_PROGRAM,
        stage6: &stage6_stage::STAGE6_PROGRAM,
        stage7: &stage7_stage::STAGE7_PROGRAM,
        stage8: &stage8_stage::STAGE8_PROGRAM,
    }
}

#[derive(Clone, Debug)]
pub struct JoltProverArtifacts {
    pub commitment: commitment_stage::CommitmentArtifacts,
    pub stage1_outer: stage1::Stage1ExecutionArtifacts<Fr>,
    pub stage2: stage2::Stage2ExecutionArtifacts<Fr>,
    pub stage3: stage3::Stage3ExecutionArtifacts<Fr>,
    pub stage4: stage4::Stage4ExecutionArtifacts<Fr>,
    pub stage5: stage5::Stage5ExecutionArtifacts<Fr>,
    pub stage6: stage6::Stage6ExecutionArtifacts<Fr>,
    pub stage7: stage7::Stage7ExecutionArtifacts<Fr>,
}

#[derive(Debug)]
pub enum JoltProveError {
    Commitment(commitment_stage::CommitmentPhaseError),
    Stage1Outer(stage1::Stage1KernelError),
    Stage2(stage2::Stage2KernelError),
    Stage3(stage3::Stage3KernelError),
    Stage4(stage4::Stage4KernelError),
    Stage5(stage5::Stage5KernelError),
    Stage6(stage6::Stage6KernelError),
    Stage7(stage7::Stage7KernelError),
    Evaluation(JoltEvaluationProveError),
}

#[derive(Debug)]
pub enum JoltEvaluationProveError {
    MissingOracle { oracle: &'static str },
    MissingOpeningHint { oracle: &'static str },
    MissingStageEval { stage: &'static str, eval: &'static str },
    MissingStage7RaEval,
    MissingStage7EvaluationPoint,
    InvalidPointLength {
        artifact: &'static str,
        expected: usize,
        actual: usize,
    },
    TargetSizeOverflow { num_vars: usize },
}

#[derive(Debug)]
pub enum JoltOpeningInputError {
    MissingOpeningClaim { stage: &'static str, source_claim: &'static str },
    MissingStage6OpeningClaim { source_claim: &'static str },
    UnsupportedOpeningInputSource { stage: &'static str, symbol: &'static str, source_stage: &'static str },
    UnsupportedStage7InputSource { symbol: &'static str, source_stage: &'static str },
    InvalidPointLength {
        symbol: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl From<commitment_stage::CommitmentPhaseError> for JoltProveError {
    fn from(error: commitment_stage::CommitmentPhaseError) -> Self {
        Self::Commitment(error)
    }
}

impl From<stage1::Stage1KernelError> for JoltProveError {
    fn from(error: stage1::Stage1KernelError) -> Self {
        Self::Stage1Outer(error)
    }
}

impl From<stage2::Stage2KernelError> for JoltProveError {
    fn from(error: stage2::Stage2KernelError) -> Self {
        Self::Stage2(error)
    }
}

impl From<stage3::Stage3KernelError> for JoltProveError {
    fn from(error: stage3::Stage3KernelError) -> Self {
        Self::Stage3(error)
    }
}

impl From<stage4::Stage4KernelError> for JoltProveError {
    fn from(error: stage4::Stage4KernelError) -> Self {
        Self::Stage4(error)
    }
}

impl From<stage5::Stage5KernelError> for JoltProveError {
    fn from(error: stage5::Stage5KernelError) -> Self {
        Self::Stage5(error)
    }
}

impl From<stage6::Stage6KernelError> for JoltProveError {
    fn from(error: stage6::Stage6KernelError) -> Self {
        Self::Stage6(error)
    }
}

impl From<stage7::Stage7KernelError> for JoltProveError {
    fn from(error: stage7::Stage7KernelError) -> Self {
        Self::Stage7(error)
    }
}

impl From<JoltEvaluationProveError> for JoltProveError {
    fn from(error: JoltEvaluationProveError) -> Self {
        Self::Evaluation(error)
    }
}

pub fn prove_jolt<CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor, Stage4Executor, Stage5Executor, Stage6Executor, Stage7Executor, T>(
    inputs: JoltProverInputs<'_, CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor, Stage4Executor, Stage5Executor, Stage6Executor, Stage7Executor>,
    transcript: &mut T,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    Stage1OuterExecutor: stage1::Stage1KernelExecutor<Fr>,
    Stage2Executor: stage2::Stage2KernelExecutor<Fr>,
    Stage3Executor: stage3::Stage3KernelExecutor<Fr>,
    Stage4Executor: stage4::Stage4KernelExecutor<Fr>,
    Stage5Executor: stage5::Stage5KernelExecutor<Fr>,
    Stage6Executor: stage6::Stage6KernelExecutor<Fr>,
    Stage7Executor: stage7::Stage7KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    prove_jolt_with_programs(inputs, default_prover_programs(), transcript)
}

pub fn prove_jolt_with_programs<CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor, Stage4Executor, Stage5Executor, Stage6Executor, Stage7Executor, T>(
    inputs: JoltProverInputs<'_, CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor, Stage4Executor, Stage5Executor, Stage6Executor, Stage7Executor>,
    programs: JoltProverPrograms,
    transcript: &mut T,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    Stage1OuterExecutor: stage1::Stage1KernelExecutor<Fr>,
    Stage2Executor: stage2::Stage2KernelExecutor<Fr>,
    Stage3Executor: stage3::Stage3KernelExecutor<Fr>,
    Stage4Executor: stage4::Stage4KernelExecutor<Fr>,
    Stage5Executor: stage5::Stage5KernelExecutor<Fr>,
    Stage6Executor: stage6::Stage6KernelExecutor<Fr>,
    Stage7Executor: stage7::Stage7KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    let _prove_span = tracing::info_span!("bolt.prove").entered();
    let _commitment_span = tracing::info_span!("bolt.commitment").entered();
    let commitment = commitment_stage::prove_commitment_phase_with_program(
        programs.commitment, inputs.commitment_inputs,
        inputs.prover_setup,
        transcript,
    )?;
    drop(_commitment_span);
    let _stage1_outer_span = tracing::info_span!("bolt.stage1").entered();
    let stage1_outer = stage1_outer_stage::prove_stage1_outer_with_program(programs.stage1_outer, inputs.stage1_outer_executor, transcript)?;
    drop(_stage1_outer_span);
    let _stage2_span = tracing::info_span!("bolt.stage2").entered();
    let stage2 = stage2_stage::execute_stage2_prover_with_program(programs.stage2, inputs.stage2_executor, transcript)?;
    drop(_stage2_span);
    let _stage3_span = tracing::info_span!("bolt.stage3").entered();
    let stage3 = stage3_stage::execute_stage3_prover_with_program(programs.stage3, inputs.stage3_executor, transcript)?;
    drop(_stage3_span);
    let _stage4_span = tracing::info_span!("bolt.stage4").entered();
    let stage4 = stage4_stage::execute_stage4_prover_with_program(programs.stage4, inputs.stage4_executor, transcript)?;
    drop(_stage4_span);
    let _stage5_span = tracing::info_span!("bolt.stage5").entered();
    let stage5 = stage5_stage::execute_stage5_prover_with_program(programs.stage5, inputs.stage5_executor, transcript)?;
    drop(_stage5_span);
    let _stage6_span = tracing::info_span!("bolt.stage6").entered();
    let stage6 = stage6_stage::execute_stage6_prover_with_program(programs.stage6, inputs.stage6_executor, transcript)?;
    drop(_stage6_span);
    let _stage7_span = tracing::info_span!("bolt.stage7").entered();
    let stage7 = stage7_stage::execute_stage7_prover_with_program(programs.stage7, inputs.stage7_executor, transcript)?;
    drop(_stage7_span);
    let evaluation = if let Some(stage7_openings) = inputs.stage7_openings {
        let _stage8_span = tracing::info_span!("bolt.stage8").entered();
        let _evaluate_span = tracing::info_span!("bolt.evaluate").entered();
        Some(prove_jolt_evaluation_proof(
            programs.stage8,
            inputs.commitment_inputs,
            inputs.prover_setup,
            &commitment,
            &stage6,
            &stage7,
            stage7_openings,
            transcript,
        )?)
    } else {
        None
    };

    let proof = JoltProof {
        commitments: commitment.commitments.clone(),
        stage1_outer: stage1_outer_proof(&stage1_outer),
        stage2: stage2_proof(&stage2),
        stage3: stage3_proof(&stage3),
        stage4: stage4_proof(&stage4),
        stage5: stage5_proof(&stage5),
        stage6: stage6_proof(&stage6),
        stage7: stage7_proof(&stage7),
        evaluation,
    };
    let artifacts = JoltProverArtifacts {
        commitment,
        stage1_outer,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7,
    };
    Ok((proof, artifacts))
}

pub fn prove_jolt_evaluation_proof<I, T>(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    commitment_inputs: &mut I,
    prover_setup: &DoryProverSetup,
    commitments: &commitment_stage::CommitmentArtifacts,
    stage6: &stage6::Stage6ExecutionArtifacts<Fr>,
    stage7: &stage7::Stage7ExecutionArtifacts<Fr>,
    stage7_openings: &[stage7::Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<JoltEvaluationProof, JoltEvaluationProveError>
where
    I: commitment_stage::CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let _claims_span = tracing::info_span!("bolt.evaluate.claims").entered();
    let (sumcheck_address_point, stage7_values) = stage7_claim_values(program, stage7)?;
    let address_point = reverse_point(&sumcheck_address_point);
    let (opening_point, log_t) =
        stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<Fr>::zero_selector(&address_point);
    let claims = evaluation_claims(program, stage6, &stage7_values, lagrange_factor)?;
    drop(_claims_span);

    let _rlc_span = tracing::info_span!("bolt.evaluate.rlc_claims").entered();
    append_rlc_claims(transcript, &claims);
    let gamma_powers = gamma_powers(transcript, claims.len());
    let joint_claim = claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| claim.value * *gamma)
        .sum();
    drop(_rlc_span);
    let _materialize_span =
        tracing::info_span!("bolt.evaluate.materialize_joint_polynomial").entered();
    let joint_evals = materialize_joint_polynomial(
        commitment_inputs,
        &claims,
        &gamma_powers,
        log_t,
        opening_point.len(),
    )?;
    drop(_materialize_span);
    let joint_poly = Polynomial::new(joint_evals);
    let _hint_span = tracing::info_span!("bolt.evaluate.joint_opening_hint").entered();
    let joint_hint = joint_opening_hint(commitments, &claims, &gamma_powers)?;
    drop(_hint_span);
    let _dory_open_span = tracing::info_span!("bolt.evaluate.dory_open").entered();
    let joint_opening_proof = <jolt_dory::DoryScheme as CommitmentScheme>::open(
        &joint_poly,
        &opening_point,
        joint_claim,
        prover_setup,
        Some(joint_hint),
        transcript,
    );
    drop(_dory_open_span);
    let _bind_span = tracing::info_span!("bolt.evaluate.bind_opening_inputs").entered();
    <jolt_dory::DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &opening_point,
        &joint_claim,
    );
    drop(_bind_span);
    Ok(JoltEvaluationProof { joint_opening_proof })
}

struct EvaluationClaim {
    oracle: &'static str,
    source_stage: &'static str,
    value: Fr,
}

fn stage6_eval_claim(
    artifacts: &stage6::Stage6ExecutionArtifacts<Fr>,
    eval_name: &'static str,
) -> Result<Fr, JoltEvaluationProveError> {
    for output in &artifacts.sumchecks {
        if let Some(eval) = output.evals.iter().find(|eval| eval.name == eval_name) {
            return Ok(eval.value);
        }
    }
    Err(JoltEvaluationProveError::MissingStageEval {
        stage: "stage6",
        eval: eval_name,
    })
}

fn evaluation_claims(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    stage6: &stage6::Stage6ExecutionArtifacts<Fr>,
    stage7_values: &std::collections::BTreeMap<&'static str, Fr>,
    lagrange_factor: Fr,
) -> Result<Vec<EvaluationClaim>, JoltEvaluationProveError> {
    let mut claims = Vec::with_capacity(program.opening_claims.len());
    for plan in program.opening_claims {
        let value = match plan.source_stage {
            "stage6" => stage6_eval_claim(stage6, plan.source_claim)? * lagrange_factor,
            "stage7" => *stage7_values.get(plan.source_claim).ok_or(
                JoltEvaluationProveError::MissingStageEval {
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                },
            )?,
            _ => {
                return Err(JoltEvaluationProveError::MissingStageEval {
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                });
            }
        };
        claims.push(EvaluationClaim {
            oracle: plan.oracle,
            source_stage: plan.source_stage,
            value,
        });
    }
    Ok(claims)
}

fn stage7_claim_values(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    artifacts: &stage7::Stage7ExecutionArtifacts<Fr>,
) -> Result<(Vec<Fr>, std::collections::BTreeMap<&'static str, Fr>), JoltEvaluationProveError> {
    let stage7_plans = program
        .opening_claims
        .iter()
        .filter(|plan| plan.source_stage == "stage7")
        .collect::<Vec<_>>();
    for output in &artifacts.sumchecks {
        let mut values = std::collections::BTreeMap::new();
        for plan in &stage7_plans {
            if let Some(eval) = output.evals.iter().find(|eval| eval.name == plan.source_claim) {
                let _ = values.insert(plan.source_claim, eval.value);
            }
        }
        if values.len() == stage7_plans.len() {
            return Ok((output.point.clone(), values));
        }
    }
    Err(JoltEvaluationProveError::MissingStage7RaEval)
}

fn reverse_point(point: &[Fr]) -> Vec<Fr> {
    point.iter().rev().copied().collect()
}

fn stage7_evaluation_opening_point(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    address_point: &[Fr],
    stage7_openings: &[stage7::Stage7OpeningInputValue<Fr>],
) -> Result<(Vec<Fr>, usize), JoltEvaluationProveError> {
    let cycle_source_symbol = program.evaluation_point_source.source_claim;
    let cycle_source = stage7_openings
        .iter()
        .find(|input| input.symbol == cycle_source_symbol)
        .ok_or(JoltEvaluationProveError::MissingStage7EvaluationPoint)?;
    if cycle_source.point.len() < address_point.len() {
        return Err(JoltEvaluationProveError::InvalidPointLength {
            artifact: cycle_source_symbol,
            expected: address_point.len(),
            actual: cycle_source.point.len(),
        });
    }
    let cycle_len = cycle_source.point.len() - address_point.len();
    let mut point = Vec::with_capacity(cycle_source.point.len());
    point.extend_from_slice(address_point);
    point.extend_from_slice(&cycle_source.point[address_point.len()..]);
    Ok((point, cycle_len))
}

fn append_rlc_claims<T>(transcript: &mut T, claims: &[EvaluationClaim])
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in claims {
        claim.value.append_to_transcript(transcript);
    }
}

fn gamma_powers<T>(transcript: &mut T, count: usize) -> Vec<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let gamma = transcript.challenge();
    let mut powers = Vec::with_capacity(count);
    let mut power = Fr::from_u64(1);
    for _ in 0..count {
        powers.push(power);
        power *= gamma;
    }
    powers
}

fn materialize_joint_polynomial<I>(
    commitment_inputs: &mut I,
    claims: &[EvaluationClaim],
    gamma_powers: &[Fr],
    log_t: usize,
    main_num_vars: usize,
) -> Result<Vec<Fr>, JoltEvaluationProveError>
where
    I: commitment_stage::CommitmentInputProvider,
{
    let trace_len = target_len(log_t)?;
    let main_len = target_len(main_num_vars)?;
    let mut joint = vec![Fr::from_u64(0); main_len];
    for (claim, gamma) in claims.iter().zip(gamma_powers) {
        if claim.source_stage == "stage6" {
            add_oracle_scaled(commitment_inputs, &mut joint, claim.oracle, log_t, trace_len, *gamma)?;
        } else {
            add_oracle_scaled(
                commitment_inputs,
                &mut joint,
                claim.oracle,
                main_num_vars,
                main_len,
                *gamma,
            )?;
        }
    }
    Ok(joint)
}

fn add_oracle_scaled<I>(
    commitment_inputs: &mut I,
    joint: &mut [Fr],
    oracle: &'static str,
    num_vars: usize,
    limit: usize,
    scalar: Fr,
) -> Result<(), JoltEvaluationProveError>
where
    I: commitment_stage::CommitmentInputProvider,
{
    if commitment_inputs.add_scaled_to_joint(oracle, joint, num_vars, limit, scalar) {
        return Ok(());
    }
    let target_len = target_len(num_vars)?;
    let data = commitment_inputs
        .materialize_with_num_vars(oracle, num_vars)
        .ok_or(JoltEvaluationProveError::MissingOracle { oracle })?;
    if data.len() > target_len {
        return Err(JoltEvaluationProveError::InvalidPointLength {
            artifact: oracle,
            expected: target_len,
            actual: data.len(),
        });
    }
    let zero = Fr::from_u64(0);
    let one = Fr::from_u64(1);
    let len = limit.min(joint.len()).min(data.len());
    if len >= 1 << 15 {
        joint[..len]
            .par_iter_mut()
            .zip(data[..len].par_iter())
            .for_each(|(dst, value)| {
                if *value == zero {
                    return;
                }
                if *value == one {
                    *dst += scalar;
                } else {
                    *dst += *value * scalar;
                }
            });
    } else {
        for (dst, value) in joint.iter_mut().take(len).zip(data.iter()) {
            if *value == zero {
                continue;
            }
            if *value == one {
                *dst += scalar;
            } else {
                *dst += *value * scalar;
            }
        }
    }
    Ok(())
}

fn joint_opening_hint(
    commitments: &commitment_stage::CommitmentArtifacts,
    claims: &[EvaluationClaim],
    gamma_powers: &[Fr],
) -> Result<DoryHint, JoltEvaluationProveError> {
    let mut coefficients = std::collections::BTreeMap::<&'static str, Fr>::new();
    for (claim, gamma) in claims.iter().zip(gamma_powers) {
        let coefficient = coefficients.entry(claim.oracle).or_insert(Fr::from_u64(0));
        *coefficient += *gamma;
    }

    let mut hints = Vec::with_capacity(coefficients.len());
    let mut scalars = Vec::with_capacity(coefficients.len());
    for (oracle, coefficient) in coefficients {
        hints.push(opening_hint_for_oracle(commitments, oracle)?);
        scalars.push(coefficient);
    }

    Ok(DoryScheme::combine_hint_refs(&hints, &scalars))
}

fn opening_hint_for_oracle<'a>(
    commitments: &'a commitment_stage::CommitmentArtifacts,
    oracle: &'static str,
) -> Result<&'a DoryHint, JoltEvaluationProveError> {
    commitments
        .hints
        .iter()
        .find(|hint| hint.oracle == oracle)
        .map(|hint| &hint.hint)
        .ok_or(JoltEvaluationProveError::MissingOpeningHint { oracle })
}

fn target_len(num_vars: usize) -> Result<usize, JoltEvaluationProveError> {
    if num_vars >= usize::BITS as usize {
        return Err(JoltEvaluationProveError::TargetSizeOverflow { num_vars });
    }
    Ok(1usize << num_vars)
}

pub struct JoltProverStageInputs<'a, CommitmentInputs> {
    pub commitment_inputs: &'a mut CommitmentInputs,
    pub prover_setup: &'a DoryProverSetup,
    pub stage1_outer: stage1::Stage1ProverInputs<'a, Fr>,
    pub stage2: stage2::Stage2ProverInputs<'a, Fr>,
    pub stage3: stage3::Stage3ProverInputs<'a, Fr>,
    pub stage4: stage4::Stage4ProverInputs<'a, Fr>,
    pub stage5: stage5::Stage5ProverInputs<'a, Fr>,
    pub stage6: stage6::Stage6ProverInputs<'a, Fr>,
    pub stage7: stage7::Stage7ProverInputs<'a, Fr>,
    pub stage7_openings: Option<&'a [stage7::Stage7OpeningInputValue<Fr>]>,
}

pub fn prove_jolt_with_stage_inputs<CommitmentInputs, T>(
    inputs: JoltProverStageInputs<'_, CommitmentInputs>,
    programs: JoltProverPrograms,
    transcript: &mut T,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let JoltProverStageInputs {
        commitment_inputs,
        prover_setup,
        stage1_outer,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7,
        stage7_openings,
    } = inputs;
    let mut stage1_outer_executor = stage1::Stage1ProverKernelExecutor::new(stage1_outer);
    let mut stage2_executor = stage2::Stage2ProverKernelExecutor::new(stage2);
    let mut stage3_executor = stage3::Stage3ProverKernelExecutor::new(stage3);
    let mut stage4_executor = stage4::Stage4ProverKernelExecutor::new(stage4);
    let mut stage5_executor = stage5::Stage5ProverKernelExecutor::new(stage5);
    let mut stage6_executor = stage6::Stage6ProverKernelExecutor::new(stage6);
    let mut stage7_executor = stage7::Stage7ProverKernelExecutor::new(stage7);
    prove_jolt_with_programs(
        JoltProverInputs {
            commitment_inputs,
            prover_setup,
            stage1_outer_executor: &mut stage1_outer_executor,
            stage2_executor: &mut stage2_executor,
            stage3_executor: &mut stage3_executor,
            stage4_executor: &mut stage4_executor,
            stage5_executor: &mut stage5_executor,
            stage6_executor: &mut stage6_executor,
            stage7_executor: &mut stage7_executor,
            stage7_openings,
        },
        programs,
        transcript,
    )
}

pub struct JoltProverWitnessInputs<'a, CommitmentInputs> {
    pub commitment_inputs: &'a mut CommitmentInputs,
    pub prover_setup: &'a DoryProverSetup,
    pub stage1_trace_num_vars: usize,
    pub stage1_outer_evaluator: &'a dyn stage1::Stage1OuterRemainingEvaluator<Fr>,
    pub stage2_openings: &'a [stage2::Stage2OpeningInputValue<Fr>],
    pub product_virtual_cycles: &'a [stage2::Stage2ProductVirtualCycle],
    pub instruction_lookup_cycles: &'a [stage2::Stage2InstructionLookupCycle],
    pub ram: &'a stage2::Stage2RamData<'a>,
    pub stage3_openings: &'a [stage3::Stage3OpeningInputValue<Fr>],
    pub stage3_cycles: &'a [stage3::Stage3Cycle],
    pub stage4_openings: &'a [stage4::Stage4OpeningInputValue<Fr>],
    pub register_count: usize,
    pub trace_len: usize,
    pub ram_k: usize,
    pub register_accesses: &'a [stage4::Stage4RegisterAccess],
    pub stage5_openings: &'a [stage5::Stage5OpeningInputValue<Fr>],
    pub lookup_indices: &'a [u128],
    pub lookup_table_indices: &'a [Option<usize>],
    pub is_interleaved_operands: &'a [bool],
    pub ra_virtual_log_k_chunk: usize,
    pub stage6_openings: &'a [stage6::Stage6OpeningInputValue<Fr>],
    pub stage6_bytecode_data: stage6::Stage6BytecodeReadRafData<'a, Fr>,
    pub stage6_witness_params: Stage6WitnessParams,
    pub cycle_inputs: &'a [CycleInput],
    pub instruction_ra_virtual_d: usize,
    pub stage7_openings: &'a [stage7::Stage7OpeningInputValue<Fr>],
    pub evaluation_openings: Option<&'a [stage7::Stage7OpeningInputValue<Fr>]>,
    pub field_reg_replay: Option<&'a jolt_witness::field_reg::FieldRegReplay>,
}

pub fn prove_jolt_with_witness_inputs<CommitmentInputs, T>(
    inputs: JoltProverWitnessInputs<'_, CommitmentInputs>,
    programs: JoltProverPrograms,
    transcript: &mut T,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let _input_span = tracing::info_span!("bolt.prove.inputs").entered();
    let _stage1_input_span = tracing::info_span!("bolt.prove.inputs.stage1").entered();
    let stage1_outer =
        stage1_outer_prover_inputs(inputs.stage1_trace_num_vars, inputs.stage1_outer_evaluator);
    drop(_stage1_input_span);
    let _stage2_input_span = tracing::info_span!("bolt.prove.inputs.stage2").entered();
    let stage2 = stage2_prover_inputs(
        inputs.stage2_openings,
        inputs.product_virtual_cycles,
        inputs.instruction_lookup_cycles,
        inputs.ram,
    )?;
    drop(_stage2_input_span);
    let _stage3_input_span = tracing::info_span!("bolt.prove.inputs.stage3").entered();
    let stage3 = stage3_prover_inputs(inputs.stage3_openings, inputs.stage3_cycles);
    drop(_stage3_input_span);
    let _stage45_witness_span = tracing::info_span!("bolt.prove.inputs.stage45_witness").entered();
    let mut stage45_witness = stage4::stage4_5_sparse_trace_witness_from_accesses(
        inputs.register_accesses,
        inputs.ram.accesses,
    );
    if let Some(replay) = inputs.field_reg_replay {
        stage45_witness = stage45_witness.with_field_reg_replay(replay);
    }
    drop(_stage45_witness_span);
    let _stage4_input_span = tracing::info_span!("bolt.prove.inputs.stage4").entered();
    let stage4 = stage4_prover_inputs(
        inputs.stage4_openings,
        inputs.register_count,
        inputs.trace_len,
        inputs.ram_k,
        inputs.register_accesses,
        &stage45_witness,
    );
    drop(_stage4_input_span);
    let _stage5_input_span = tracing::info_span!("bolt.prove.inputs.stage5").entered();
    let stage5 = stage5_prover_inputs(
        inputs.stage5_openings,
        inputs.trace_len,
        inputs.ram_k,
        inputs.register_count,
        inputs.lookup_indices,
        inputs.lookup_table_indices,
        inputs.is_interleaved_operands,
        inputs.ra_virtual_log_k_chunk,
        &stage45_witness,
    );
    drop(_stage5_input_span);
    let _stage6_witness_span = tracing::info_span!("bolt.prove.inputs.stage6_witness").entered();
    let stage6_witness = stage6_witness_from_opening_inputs(
        inputs.stage6_witness_params,
        inputs.cycle_inputs,
        inputs.stage6_openings,
    );
    let stage6_witness_slices = stage6_witness.slices();
    drop(_stage6_witness_span);
    let _stage6_input_span = tracing::info_span!("bolt.prove.inputs.stage6").entered();
    let stage6 = stage6_prover_inputs(
        inputs.stage6_openings,
        inputs.stage6_bytecode_data,
        &stage6_witness,
        &stage6_witness_slices,
        inputs.instruction_ra_virtual_d,
    );
    drop(_stage6_input_span);
    let _stage7_input_span = tracing::info_span!("bolt.prove.inputs.stage7").entered();
    let stage7 = stage7_prover_inputs(inputs.stage7_openings, &stage6_witness_slices);
    drop(_stage7_input_span);
    drop(_input_span);
    prove_jolt_with_stage_inputs(
        JoltProverStageInputs {
            commitment_inputs: inputs.commitment_inputs,
            prover_setup: inputs.prover_setup,
            stage1_outer,
            stage2,
            stage3,
            stage4,
            stage5,
            stage6,
            stage7,
            stage7_openings: inputs.evaluation_openings,
        },
        programs,
        transcript,
    )
}

pub fn stage1_outer_prover_inputs(
    trace_num_vars: usize,
    evaluator: &dyn stage1::Stage1OuterRemainingEvaluator<Fr>,
) -> stage1::Stage1ProverInputs<'_, Fr> {
    stage1::Stage1ProverInputs::empty(trace_num_vars).with_outer_remaining_evaluator(evaluator)
}

pub fn prove_stage1_outer_inputs_with_program<T>(
    program: &'static stage1::Stage1CpuProgramPlan,
    inputs: stage1::Stage1ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage1::Stage1ExecutionArtifacts<Fr>, stage1::Stage1KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage1::Stage1ProverKernelExecutor::new(inputs);
    stage1_outer_stage::prove_stage1_outer_with_program(program, &mut executor, transcript)
}

pub fn prove_stage1_outer_with_witness_inputs<T>(
    program: &'static stage1::Stage1CpuProgramPlan,
    trace_num_vars: usize,
    evaluator: &dyn stage1::Stage1OuterRemainingEvaluator<Fr>,
    transcript: &mut T,
) -> Result<stage1::Stage1ExecutionArtifacts<Fr>, stage1::Stage1KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage1_outer_prover_inputs(trace_num_vars, evaluator);
    prove_stage1_outer_inputs_with_program(program, inputs, transcript)
}

pub fn replay_stage1_outer_proof_with_program<T>(
    program: &'static stage1::Stage1CpuProgramPlan,
    proof: &stage1::Stage1Proof<Fr>,
    transcript: &mut T,
) -> Result<stage1::Stage1ExecutionArtifacts<Fr>, stage1::Stage1KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage1::Stage1VerifierKernelExecutor::new(proof);
    stage1::execute_stage1_program(
        program,
        stage1::Stage1ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}

pub fn stage1_outer_proof_from_kernel_proof(
    proof: &stage1::Stage1Proof<Fr>,
) -> JoltStageProof {
    JoltStageProof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(stage1_outer_sumcheck)
            .collect(),
    }
}

pub fn stage2_prover_inputs<'a>(
    opening_inputs: &'a [stage2::Stage2OpeningInputValue<Fr>],
    product_virtual_cycles: &'a [stage2::Stage2ProductVirtualCycle],
    instruction_lookup_cycles: &'a [stage2::Stage2InstructionLookupCycle],
    ram: &'a stage2::Stage2RamData<'a>,
) -> Result<stage2::Stage2ProverInputs<'a, Fr>, stage2::Stage2KernelError> {
    Ok(stage2::Stage2ProverInputs::new(opening_inputs)
        .with_product_virtual_witness(product_virtual_cycles)?
        .with_instruction_lookup_cycles(instruction_lookup_cycles)
        .with_ram_data(ram))
}

pub struct JoltStage2RamDataStorage<'a> {
    log_k: usize,
    start_address: u64,
    initial_ram: &'a [u64],
    final_ram: &'a [u64],
    accesses: Vec<JoltStage2RamAccess>,
    output_layout: Option<JoltStage2RamOutputLayout>,
}

impl<'a> JoltStage2RamDataStorage<'a> {
    pub fn from_kernel(ram: &stage2::Stage2RamData<'a>) -> Self {
        Self {
            log_k: ram.log_k,
            start_address: ram.start_address,
            initial_ram: ram.initial_ram,
            final_ram: ram.final_ram,
            accesses: ram
                .accesses
                .iter()
                .map(|access| JoltStage2RamAccess {
                    remapped_address: access.remapped_address,
                    read_value: access.read_value,
                    write_value: access.write_value,
                })
                .collect(),
            output_layout: ram.output_layout.map(|layout| JoltStage2RamOutputLayout {
                io_start: layout.io_start,
                io_end: layout.io_end,
            }),
        }
    }

    pub fn as_input(&self) -> JoltStage2RamData<'_> {
        JoltStage2RamData {
            log_k: self.log_k,
            start_address: self.start_address,
            initial_ram: self.initial_ram,
            final_ram: self.final_ram,
            accesses: &self.accesses,
            output_layout: self.output_layout,
        }
    }
}

pub fn stage2_verifier_ram_data<'a>(
    ram: &stage2::Stage2RamData<'a>,
) -> JoltStage2RamDataStorage<'a> {
    JoltStage2RamDataStorage::from_kernel(ram)
}

pub trait JoltKernelOpeningInput {
    fn symbol(&self) -> &'static str;
    fn point(&self) -> &[Fr];
    fn eval(&self) -> Fr;
}

macro_rules! impl_jolt_kernel_opening_input {
    ($opening:ty) => {
        impl JoltKernelOpeningInput for $opening {
            fn symbol(&self) -> &'static str {
                self.symbol
            }

            fn point(&self) -> &[Fr] {
                &self.point
            }

            fn eval(&self) -> Fr {
                self.eval
            }
        }
    };
}

impl_jolt_kernel_opening_input!(stage2::Stage2OpeningInputValue<Fr>);
impl_jolt_kernel_opening_input!(stage3::Stage3OpeningInputValue<Fr>);
impl_jolt_kernel_opening_input!(stage4::Stage4OpeningInputValue<Fr>);

pub fn verifier_opening_inputs_from_kernel<I>(inputs: &[I]) -> Vec<JoltStageOpeningInputValue>
where
    I: JoltKernelOpeningInput,
{
    inputs
        .iter()
        .map(|input| JoltStageOpeningInputValue {
            symbol: input.symbol(),
            point: input.point().to_vec(),
            eval: input.eval(),
        })
        .collect()
}

pub fn prove_stage2_inputs_with_program<T>(
    program: &'static stage2::Stage2CpuProgramPlan,
    inputs: stage2::Stage2ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage2::Stage2ExecutionArtifacts<Fr>, stage2::Stage2KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage2::Stage2ProverKernelExecutor::new(inputs);
    stage2_stage::execute_stage2_prover_with_program(program, &mut executor, transcript)
}

pub fn prove_stage2_with_witness_inputs<'a, T>(
    program: &'static stage2::Stage2CpuProgramPlan,
    opening_inputs: &'a [stage2::Stage2OpeningInputValue<Fr>],
    product_virtual_cycles: &'a [stage2::Stage2ProductVirtualCycle],
    instruction_lookup_cycles: &'a [stage2::Stage2InstructionLookupCycle],
    ram: &'a stage2::Stage2RamData<'a>,
    transcript: &mut T,
) -> Result<stage2::Stage2ExecutionArtifacts<Fr>, stage2::Stage2KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage2_prover_inputs(
        opening_inputs,
        product_virtual_cycles,
        instruction_lookup_cycles,
        ram,
    )?;
    prove_stage2_inputs_with_program(program, inputs, transcript)
}

pub fn stage2_opening_inputs_from_artifacts(
    program: &'static stage2::Stage2CpuProgramPlan,
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
) -> Result<Vec<stage2::Stage2OpeningInputValue<Fr>>, JoltOpeningInputError> {
    program
        .opening_inputs
        .iter()
        .map(|input| {
            let (point, eval) = match input.source_stage {
                "stage1" => stage1_opening_claim(stage1_artifacts, input.source_claim)?,
                source_stage => {
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {
                        stage: "stage2",
                        symbol: input.symbol,
                        source_stage,
                    });
                }
            };
            validate_point_len(input.symbol, input.point_arity, point.len())?;
            Ok(stage2::Stage2OpeningInputValue {
                symbol: input.symbol,
                point,
                eval,
            })
        })
        .collect()
}

pub fn replay_stage2_proof_with_program<'a, T>(
    program: &'static stage2::Stage2CpuProgramPlan,
    proof: &'a stage2::Stage2Proof<Fr>,
    opening_inputs: &'a [stage2::Stage2OpeningInputValue<Fr>],
    ram: Option<&'a stage2::Stage2RamData<'a>>,
    transcript: &mut T,
) -> Result<stage2::Stage2ExecutionArtifacts<Fr>, stage2::Stage2KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage2::Stage2VerifierKernelExecutor::new(proof, opening_inputs);
    if let Some(ram) = ram {
        executor = executor.with_ram_data(ram);
    }
    stage2::execute_stage2_program(
        program,
        stage2::Stage2ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}

pub fn stage3_prover_inputs<'a>(
    opening_inputs: &'a [stage3::Stage3OpeningInputValue<Fr>],
    cycles: &'a [stage3::Stage3Cycle],
) -> stage3::Stage3ProverInputs<'a, Fr> {
    stage3::Stage3ProverInputs::new(opening_inputs).with_cycles(cycles)
}

pub fn prove_stage3_inputs_with_program<T>(
    program: &'static stage3::Stage3CpuProgramPlan,
    inputs: stage3::Stage3ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage3::Stage3ExecutionArtifacts<Fr>, stage3::Stage3KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage3::Stage3ProverKernelExecutor::new(inputs);
    stage3_stage::execute_stage3_prover_with_program(program, &mut executor, transcript)
}

pub fn prove_stage3_with_witness_inputs<T>(
    program: &'static stage3::Stage3CpuProgramPlan,
    opening_inputs: &[stage3::Stage3OpeningInputValue<Fr>],
    cycles: &[stage3::Stage3Cycle],
    transcript: &mut T,
) -> Result<stage3::Stage3ExecutionArtifacts<Fr>, stage3::Stage3KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage3_prover_inputs(opening_inputs, cycles);
    prove_stage3_inputs_with_program(program, inputs, transcript)
}

pub fn stage3_opening_inputs_from_artifacts(
    program: &'static stage3::Stage3CpuProgramPlan,
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
) -> Result<Vec<stage3::Stage3OpeningInputValue<Fr>>, JoltOpeningInputError> {
    program
        .opening_inputs
        .iter()
        .map(|input| {
            let (point, eval) = match input.source_stage {
                "stage1" => stage1_opening_claim(stage1_artifacts, input.source_claim)?,
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                source_stage => {
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {
                        stage: "stage3",
                        symbol: input.symbol,
                        source_stage,
                    });
                }
            };
            validate_point_len(input.symbol, input.point_arity, point.len())?;
            Ok(stage3::Stage3OpeningInputValue {
                symbol: input.symbol,
                point,
                eval,
            })
        })
        .collect()
}

pub fn replay_stage3_proof_with_program<T>(
    program: &'static stage3::Stage3CpuProgramPlan,
    proof: &stage3::Stage3Proof<Fr>,
    opening_inputs: &[stage3::Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<stage3::Stage3ExecutionArtifacts<Fr>, stage3::Stage3KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage3::Stage3VerifierKernelExecutor::new(proof, opening_inputs);
    stage3::execute_stage3_program(
        program,
        stage3::Stage3ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}

pub fn stage4_prover_inputs<'a>(
    opening_inputs: &'a [stage4::Stage4OpeningInputValue<Fr>],
    register_count: usize,
    trace_len: usize,
    ram_k: usize,
    register_accesses: &'a [stage4::Stage4RegisterAccess],
    witness: &'a Stage45SparseTraceWitness<Fr>,
) -> stage4::Stage4ProverInputs<'a, Fr> {
    stage4::Stage4ProverInputs::new(opening_inputs).with_stage45_sparse_trace_witness(
        register_count,
        trace_len,
        ram_k,
        register_accesses,
        witness,
    )
}

pub fn prove_stage4_inputs_with_program<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    inputs: stage4::Stage4ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<Fr>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage4::Stage4ProverKernelExecutor::new(inputs);
    stage4_stage::execute_stage4_prover_with_program(program, &mut executor, transcript)
}

pub fn prove_stage4_with_witness_inputs<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    opening_inputs: &[stage4::Stage4OpeningInputValue<Fr>],
    register_count: usize,
    trace_len: usize,
    ram_k: usize,
    register_accesses: &[stage4::Stage4RegisterAccess],
    witness: &Stage45SparseTraceWitness<Fr>,
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<Fr>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage4_prover_inputs(
        opening_inputs,
        register_count,
        trace_len,
        ram_k,
        register_accesses,
        witness,
    );
    prove_stage4_inputs_with_program(program, inputs, transcript)
}

pub fn prove_stage4_with_trace_witness_inputs<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    opening_inputs: &[stage4::Stage4OpeningInputValue<Fr>],
    register_count: usize,
    trace_len: usize,
    ram_k: usize,
    register_accesses: &[stage4::Stage4RegisterAccess],
    ram_accesses: &[stage2::Stage2RamAccess],
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<Fr>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let witness = stage4::stage4_5_sparse_trace_witness_from_accesses(
        register_accesses,
        ram_accesses,
    );
    prove_stage4_with_witness_inputs(
        program,
        opening_inputs,
        register_count,
        trace_len,
        ram_k,
        register_accesses,
        &witness,
        transcript,
    )
}

pub fn stage4_opening_inputs_from_artifacts(
    program: &'static stage4::Stage4CpuProgramPlan,
    initial_ram_state: &[u64],
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<Fr>,
) -> Result<Vec<stage4::Stage4OpeningInputValue<Fr>>, JoltOpeningInputError> {
    program
        .opening_inputs
        .iter()
        .map(|input| {
            let (point, eval) = match input.source_stage {
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                "stage3" => stage3_opening_claim(stage3_artifacts, input.source_claim)?,
                "stage4_precomputed" => {
                    let (point, _) = stage2_opening_claim(
                        stage2_artifacts,
                        "stage2.ram_output.opening.RamValFinal",
                    )?;
                    stage4_ram_val_init_opening(initial_ram_state, &point)
                }
                source_stage => {
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {
                        stage: "stage4",
                        symbol: input.symbol,
                        source_stage,
                    });
                }
            };
            opening_input_value(input.symbol, input.point_arity, point, eval)
        })
        .collect()
}

pub fn replay_stage4_proof_with_program<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    proof: &stage4::Stage4Proof<Fr>,
    opening_inputs: &[stage4::Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<Fr>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage4::Stage4VerifierKernelExecutor::new(proof, opening_inputs);
    stage4::execute_stage4_program(
        program,
        stage4::Stage4ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}

pub fn stage5_prover_inputs<'a>(
    opening_inputs: &'a [stage5::Stage5OpeningInputValue<Fr>],
    trace_len: usize,
    ram_k: usize,
    register_count: usize,
    lookup_indices: &'a [u128],
    lookup_table_indices: &'a [Option<usize>],
    is_interleaved_operands: &'a [bool],
    ra_virtual_log_k_chunk: usize,
    witness: &'a Stage45SparseTraceWitness<Fr>,
) -> stage5::Stage5ProverInputs<'a, Fr> {
    stage5::Stage5ProverInputs::new(opening_inputs).with_stage45_sparse_trace_witness(
        trace_len,
        ram_k,
        register_count,
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
        ra_virtual_log_k_chunk,
        witness,
    )
}

pub fn prove_stage5_inputs_with_program<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    inputs: stage5::Stage5ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<Fr>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage5::Stage5ProverKernelExecutor::new(inputs);
    stage5_stage::execute_stage5_prover_with_program(program, &mut executor, transcript)
}

pub fn prove_stage5_with_witness_inputs<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    opening_inputs: &[stage5::Stage5OpeningInputValue<Fr>],
    trace_len: usize,
    ram_k: usize,
    register_count: usize,
    lookup_indices: &[u128],
    lookup_table_indices: &[Option<usize>],
    is_interleaved_operands: &[bool],
    ra_virtual_log_k_chunk: usize,
    witness: &Stage45SparseTraceWitness<Fr>,
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<Fr>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage5_prover_inputs(
        opening_inputs,
        trace_len,
        ram_k,
        register_count,
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
        ra_virtual_log_k_chunk,
        witness,
    );
    prove_stage5_inputs_with_program(program, inputs, transcript)
}

pub fn prove_stage5_with_trace_witness_inputs<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    opening_inputs: &[stage5::Stage5OpeningInputValue<Fr>],
    trace_len: usize,
    ram_k: usize,
    register_count: usize,
    lookup_indices: &[u128],
    lookup_table_indices: &[Option<usize>],
    is_interleaved_operands: &[bool],
    ra_virtual_log_k_chunk: usize,
    register_accesses: &[stage4::Stage4RegisterAccess],
    ram_accesses: &[stage2::Stage2RamAccess],
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<Fr>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let witness = stage4::stage4_5_sparse_trace_witness_from_accesses(
        register_accesses,
        ram_accesses,
    );
    prove_stage5_with_witness_inputs(
        program,
        opening_inputs,
        trace_len,
        ram_k,
        register_count,
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
        ra_virtual_log_k_chunk,
        &witness,
        transcript,
    )
}

pub fn stage5_opening_inputs_from_artifacts(
    program: &'static stage5::Stage5CpuProgramPlan,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<Fr>,
) -> Result<Vec<stage5::Stage5OpeningInputValue<Fr>>, JoltOpeningInputError> {
    program
        .opening_inputs
        .iter()
        .map(|input| {
            let (point, eval) = match input.source_stage {
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                "stage4" => stage4_opening_claim(stage4_artifacts, input.source_claim)?,
                source_stage => {
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {
                        stage: "stage5",
                        symbol: input.symbol,
                        source_stage,
                    });
                }
            };
            opening_input_value(input.symbol, input.point_arity, point, eval)
        })
        .collect()
}

pub fn stage5_kernel_proof(
    artifacts: &stage5::Stage5ExecutionArtifacts<Fr>,
) -> stage5::Stage5Proof<Fr> {
    stage5::Stage5Proof {
        sumchecks: artifacts.sumchecks.clone(),
    }
}

pub fn jolt_proof_through_stage5(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &JoltStageProof,
) -> JoltProof {
    JoltProof {
        commitments: commitments.to_vec(),
        stage1_outer: stage1_outer_proof(stage1_artifacts),
        stage2: stage2_proof(stage2_artifacts),
        stage3: stage3_proof(stage3_artifacts),
        stage4: stage4_proof(stage4_artifacts),
        stage5: stage5_proof.clone(),
        stage6: JoltStageProof::default(),
        stage7: JoltStageProof::default(),
        evaluation: None,
    }
}

pub fn jolt_proof_through_stage6(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
) -> JoltProof {
    let mut proof = jolt_proof_through_stage5(
        commitments,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
    );
    proof.stage6 = stage6_proof.clone();
    proof
}

pub fn jolt_proof_through_stage7(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<Fr>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
    stage7_proof: &JoltStageProof,
) -> JoltProof {
    let mut proof = jolt_proof_through_stage6(
        commitments,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
        stage6_proof,
    );
    proof.stage7 = stage7_proof.clone();
    proof
}

pub fn replay_stage5_proof_with_program<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    proof: &stage5::Stage5Proof<Fr>,
    opening_inputs: &[stage5::Stage5OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<Fr>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage5::Stage5ProofCarryingKernelExecutor::new(proof, opening_inputs);
    stage5_stage::execute_stage5_prover_with_program(program, &mut executor, transcript)
}

pub fn stage6_witness_from_opening_inputs(
    params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    opening_inputs: &[stage6::Stage6OpeningInputValue<Fr>],
) -> Stage6WitnessPolynomials<Fr> {
    stage6::stage6_witness_from_opening_inputs(params, cycle_inputs, opening_inputs)
}

pub fn stage6_bytecode_read_raf_data_from_witness_entries(
    entries: &[WitnessStage6BytecodeEntry<Fr>],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
) -> stage6::Stage6BytecodeReadRafDataStorage<Fr> {
    stage6::Stage6BytecodeReadRafDataStorage::from_witness_entries(
        entries,
        entry_bytecode_index,
        num_lookup_tables,
    )
}

pub fn stage6_verifier_data_from_witness_entries(
    entries: &[WitnessStage6BytecodeEntry<Fr>],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
) -> JoltStage6VerifierData {
    JoltStage6VerifierData {
        bytecode_read_raf: Some(JoltStage6BytecodeReadRafData {
            entries: entries
                .iter()
                .map(|entry| JoltStage6BytecodeEntry {
                    address: entry.address,
                    imm: entry.imm,
                    circuit_flags: entry.circuit_flags,
                    rd: entry.rd,
                    rs1: entry.rs1,
                    rs2: entry.rs2,
                    lookup_table: entry.lookup_table,
                    is_interleaved: entry.is_interleaved,
                    is_branch: entry.is_branch,
                    left_is_rs1: entry.left_is_rs1,
                    left_is_pc: entry.left_is_pc,
                    right_is_rs2: entry.right_is_rs2,
                    right_is_imm: entry.right_is_imm,
                    is_noop: entry.is_noop,
                })
                .collect(),
            entry_bytecode_index,
            num_lookup_tables,
        }),
    }
}

pub fn stage6_prover_inputs<'a>(
    opening_inputs: &'a [stage6::Stage6OpeningInputValue<Fr>],
    bytecode_data: stage6::Stage6BytecodeReadRafData<'a, Fr>,
    witness: &'a Stage6WitnessPolynomials<Fr>,
    slices: &'a Stage6WitnessSlices<'a, Fr>,
    instruction_ra_virtual_d: usize,
) -> stage6::Stage6ProverInputs<'a, Fr> {
    stage6::Stage6ProverInputs::new(opening_inputs).with_stage6_witness(
        bytecode_data,
        witness,
        slices,
        instruction_ra_virtual_d,
    )
}

pub fn prove_stage6_inputs_with_program<T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    inputs: stage6::Stage6ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<Fr>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage6::Stage6ProverKernelExecutor::new(inputs);
    stage6_stage::execute_stage6_prover_with_program(program, &mut executor, transcript)
}

pub fn prove_stage6_with_witness_inputs<T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    opening_inputs: &[stage6::Stage6OpeningInputValue<Fr>],
    bytecode_data: stage6::Stage6BytecodeReadRafData<'_, Fr>,
    witness: &Stage6WitnessPolynomials<Fr>,
    slices: &Stage6WitnessSlices<'_, Fr>,
    instruction_ra_virtual_d: usize,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<Fr>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage6_prover_inputs(
        opening_inputs,
        bytecode_data,
        witness,
        slices,
        instruction_ra_virtual_d,
    );
    prove_stage6_inputs_with_program(program, inputs, transcript)
}

pub fn prove_stage6_with_trace_witness_inputs<T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    opening_inputs: &[stage6::Stage6OpeningInputValue<Fr>],
    bytecode_data: stage6::Stage6BytecodeReadRafData<'_, Fr>,
    witness_params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    instruction_ra_virtual_d: usize,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<Fr>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let witness = stage6_witness_from_opening_inputs(witness_params, cycle_inputs, opening_inputs);
    let slices = witness.slices();
    prove_stage6_with_witness_inputs(
        program,
        opening_inputs,
        bytecode_data,
        &witness,
        &slices,
        instruction_ra_virtual_d,
        transcript,
    )
}

pub fn stage6_opening_inputs_from_artifacts(
    program: &'static stage6::Stage6CpuProgramPlan,
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<Fr>,
    stage5_artifacts: &stage5::Stage5ExecutionArtifacts<Fr>,
) -> Result<Vec<stage6::Stage6OpeningInputValue<Fr>>, JoltOpeningInputError> {
    program
        .opening_inputs
        .iter()
        .map(|input| {
            let (point, eval) = match input.source_stage {
                "stage1" => stage1_opening_claim(stage1_artifacts, input.source_claim)?,
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                "stage3" => stage3_opening_claim(stage3_artifacts, input.source_claim)?,
                "stage4" => stage4_opening_claim(stage4_artifacts, input.source_claim)?,
                "stage5" => stage5_opening_claim(stage5_artifacts, input.source_claim)?,
                source_stage => {
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {
                        stage: "stage6",
                        symbol: input.symbol,
                        source_stage,
                    });
                }
            };
            opening_input_value(input.symbol, input.point_arity, point, eval)
        })
        .collect()
}

pub fn stage6_kernel_proof(proof: &JoltStageProof) -> stage6::Stage6Proof<Fr> {
    stage6::Stage6Proof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(stage6_kernel_sumcheck_output)
            .collect(),
    }
}

fn stage6_kernel_sumcheck_output(
    output: &JoltSumcheckOutput,
) -> stage6::Stage6SumcheckOutput<Fr> {
    stage6::Stage6SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage6_kernel_eval).collect(),
        opening_claims: Vec::new(),
        proof: output.proof.clone(),
    }
}

fn stage6_kernel_eval(eval: &JoltNamedEval) -> stage6::Stage6NamedEval<Fr> {
    stage6::Stage6NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage6_execution_artifacts(
    artifacts: &stage6::Stage6ExecutionArtifacts<Fr>,
) -> JoltStageExecutionArtifacts {
    JoltStageExecutionArtifacts {
        challenge_vectors: artifacts
            .challenge_vectors
            .iter()
            .map(|challenge| JoltStageChallengeVector {
                symbol: challenge.symbol,
                values: challenge.values.clone(),
            })
            .collect(),
        sumchecks: stage6_proof(artifacts).sumchecks,
        opening_batches: Vec::new(),
    }
}

pub fn replay_stage6_proof_with_program<'a, T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    proof: &'a stage6::Stage6Proof<Fr>,
    opening_inputs: &'a [stage6::Stage6OpeningInputValue<Fr>],
    bytecode_data: Option<stage6::Stage6BytecodeReadRafData<'a, Fr>>,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<Fr>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage6::Stage6ProofCarryingKernelExecutor::new(proof, opening_inputs);
    if let Some(bytecode_data) = bytecode_data {
        executor = executor.with_bytecode_read_raf_data(bytecode_data);
    }
    stage6_stage::execute_stage6_prover_with_program(program, &mut executor, transcript)
}

pub fn stage7_prover_inputs<'a>(
    opening_inputs: &'a [stage7::Stage7OpeningInputValue<Fr>],
    slices: &'a Stage6WitnessSlices<'a, Fr>,
) -> stage7::Stage7ProverInputs<'a, Fr> {
    stage7::Stage7ProverInputs::new(opening_inputs).with_stage6_witness_indices(slices)
}

pub fn prove_stage7_inputs_with_program<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    inputs: stage7::Stage7ProverInputs<'_, Fr>,
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<Fr>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage7::Stage7ProverKernelExecutor::new(inputs);
    stage7_stage::execute_stage7_prover_with_program(program, &mut executor, transcript)
}

pub fn prove_stage7_with_witness_inputs<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    opening_inputs: &[stage7::Stage7OpeningInputValue<Fr>],
    slices: &Stage6WitnessSlices<'_, Fr>,
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<Fr>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = stage7_prover_inputs(opening_inputs, slices);
    prove_stage7_inputs_with_program(program, inputs, transcript)
}

pub fn prove_stage7_with_trace_witness_inputs<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    opening_inputs: &[stage7::Stage7OpeningInputValue<Fr>],
    witness_params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    stage6_openings: &[stage6::Stage6OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<Fr>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let witness = stage6_witness_from_opening_inputs(witness_params, cycle_inputs, stage6_openings);
    let slices = witness.slices();
    prove_stage7_with_witness_inputs(program, opening_inputs, &slices, transcript)
}

pub fn stage7_kernel_proof(proof: &JoltStageProof) -> stage7::Stage7Proof<Fr> {
    stage7::Stage7Proof {
        sumchecks: proof
            .sumchecks
            .iter()
            .map(stage7_kernel_sumcheck_output)
            .collect(),
    }
}

fn stage7_kernel_sumcheck_output(
    output: &JoltSumcheckOutput,
) -> stage7::Stage7SumcheckOutput<Fr> {
    stage7::Stage7SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage7_kernel_eval).collect(),
        opening_claims: Vec::new(),
        proof: output.proof.clone(),
    }
}

fn stage7_kernel_eval(eval: &JoltNamedEval) -> stage7::Stage7NamedEval<Fr> {
    stage7::Stage7NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage7_execution_artifacts(
    artifacts: &stage7::Stage7ExecutionArtifacts<Fr>,
) -> JoltStageExecutionArtifacts {
    JoltStageExecutionArtifacts {
        challenge_vectors: artifacts
            .challenge_vectors
            .iter()
            .map(|challenge| JoltStageChallengeVector {
                symbol: challenge.symbol,
                values: challenge.values.clone(),
            })
            .collect(),
        sumchecks: stage7_proof(artifacts).sumchecks,
        opening_batches: Vec::new(),
    }
}

pub fn replay_stage7_proof_with_program<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    proof: &stage7::Stage7Proof<Fr>,
    opening_inputs: &[stage7::Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<Fr>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut executor = stage7::Stage7ProofCarryingKernelExecutor::new(proof, opening_inputs);
    stage7_stage::execute_stage7_prover_with_program(program, &mut executor, transcript)
}

pub fn stage7_opening_inputs_from_stage6_artifacts(
    artifacts: &stage6::Stage6ExecutionArtifacts<Fr>,
) -> Result<Vec<stage7::Stage7OpeningInputValue<Fr>>, JoltOpeningInputError> {
    stage7_opening_inputs_from_stage6_artifacts_with_program(&stage7_stage::STAGE7_PROGRAM, artifacts)
}

pub fn stage7_opening_inputs_from_stage6_artifacts_with_program(
    program: &'static stage7::Stage7CpuProgramPlan,
    artifacts: &stage6::Stage6ExecutionArtifacts<Fr>,
) -> Result<Vec<stage7::Stage7OpeningInputValue<Fr>>, JoltOpeningInputError> {
    program
        .opening_inputs
        .iter()
        .map(|input| {
            let (point, eval) = stage6_opening_claim(artifacts, input.symbol, input.source_stage, input.source_claim, input.point_arity)?;
            Ok(stage7::Stage7OpeningInputValue {
                symbol: input.symbol,
                point,
                eval,
            })
        })
        .collect()
}

fn stage6_opening_claim(
    artifacts: &stage6::Stage6ExecutionArtifacts<Fr>,
    symbol: &'static str,
    source_stage: &'static str,
    source_claim: &'static str,
    point_arity: usize,
) -> Result<(Vec<Fr>, Fr), JoltOpeningInputError> {
    if source_stage != "stage6" {
        return Err(JoltOpeningInputError::UnsupportedStage7InputSource {
            symbol,
            source_stage,
        });
    }
    let opening = artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .ok_or(JoltOpeningInputError::MissingStage6OpeningClaim { source_claim })?;
    if opening.point.len() != point_arity {
        return Err(JoltOpeningInputError::InvalidPointLength {
            symbol,
            expected: point_arity,
            actual: opening.point.len(),
        });
    }
    Ok((opening.point.clone(), opening.eval))
}

fn opening_input_value(
    symbol: &'static str,
    point_arity: usize,
    point: Vec<Fr>,
    eval: Fr,
) -> Result<stage4::Stage4OpeningInputValue<Fr>, JoltOpeningInputError> {
    validate_point_len(symbol, point_arity, point.len())?;
    Ok(stage4::Stage4OpeningInputValue {
        symbol,
        point,
        eval,
    })
}

fn validate_point_len(
    symbol: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), JoltOpeningInputError> {
    if actual != expected {
        return Err(JoltOpeningInputError::InvalidPointLength {
            symbol,
            expected,
            actual,
        });
    }
    Ok(())
}

fn stage1_opening_claim(
    artifacts: &stage1::Stage1ExecutionArtifacts<Fr>,
    source_claim: &'static str,
) -> Result<(Vec<Fr>, Fr), JoltOpeningInputError> {
    let opening = artifacts.opening_value(source_claim).ok_or(
        JoltOpeningInputError::MissingOpeningClaim {
            stage: "stage1",
            source_claim,
        },
    )?;
    Ok((opening.point.clone(), opening.eval))
}

fn stage2_opening_claim(
    artifacts: &stage2::Stage2ExecutionArtifacts<Fr>,
    source_claim: &'static str,
) -> Result<(Vec<Fr>, Fr), JoltOpeningInputError> {
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {
            stage: "stage2",
            source_claim,
        })
}

fn stage3_opening_claim(
    artifacts: &stage3::Stage3ExecutionArtifacts<Fr>,
    source_claim: &'static str,
) -> Result<(Vec<Fr>, Fr), JoltOpeningInputError> {
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {
            stage: "stage3",
            source_claim,
        })
}

fn stage4_opening_claim(
    artifacts: &stage4::Stage4ExecutionArtifacts<Fr>,
    source_claim: &'static str,
) -> Result<(Vec<Fr>, Fr), JoltOpeningInputError> {
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {
            stage: "stage4",
            source_claim,
        })
}

fn stage5_opening_claim(
    artifacts: &stage5::Stage5ExecutionArtifacts<Fr>,
    source_claim: &'static str,
) -> Result<(Vec<Fr>, Fr), JoltOpeningInputError> {
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {
            stage: "stage5",
            source_claim,
        })
}

pub fn stage1_outer_proof(artifacts: &stage1::Stage1ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage1_outer_sumcheck).collect(),
    }
}

fn stage1_outer_sumcheck(output: &stage1::Stage1SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage1_outer_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage1_outer_eval(eval: &stage1::Stage1NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage2_proof(artifacts: &stage2::Stage2ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage2_sumcheck).collect(),
    }
}

fn stage2_sumcheck(output: &stage2::Stage2SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage2_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage2_eval(eval: &stage2::Stage2NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage3_proof(artifacts: &stage3::Stage3ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage3_sumcheck).collect(),
    }
}

fn stage3_sumcheck(output: &stage3::Stage3SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage3_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage3_eval(eval: &stage3::Stage3NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage4_proof(artifacts: &stage4::Stage4ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage4_sumcheck).collect(),
    }
}

fn stage4_sumcheck(output: &stage4::Stage4SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage4_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage4_eval(eval: &stage4::Stage4NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage5_proof(artifacts: &stage5::Stage5ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage5_sumcheck).collect(),
    }
}

fn stage5_sumcheck(output: &stage5::Stage5SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage5_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage5_eval(eval: &stage5::Stage5NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage6_proof(artifacts: &stage6::Stage6ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage6_sumcheck).collect(),
    }
}

fn stage6_sumcheck(output: &stage6::Stage6SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage6_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage6_eval(eval: &stage6::Stage6NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

pub fn stage7_proof(artifacts: &stage7::Stage7ExecutionArtifacts<Fr>) -> JoltStageProof {
    JoltStageProof {
        sumchecks: artifacts.sumchecks.iter().map(stage7_sumcheck).collect(),
    }
}

fn stage7_sumcheck(output: &stage7::Stage7SumcheckOutput<Fr>) -> JoltSumcheckOutput {
    JoltSumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage7_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage7_eval(eval: &stage7::Stage7NamedEval<Fr>) -> JoltNamedEval {
    JoltNamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}
