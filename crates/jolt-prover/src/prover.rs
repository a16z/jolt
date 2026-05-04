use jolt_dory::{DoryHint, DoryProverSetup, DoryScheme};
use jolt_field::{Field, Fr};
use jolt_kernels::{stage1, stage2, stage3, stage4, stage5, stage6, stage7};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};
use jolt_verifier::{JoltEvaluationProof, JoltNamedEval, JoltProof, JoltStageProof, JoltSumcheckOutput};
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
    let (sumcheck_address_point, stage7_values) = stage7_claim_values(program, stage7)?;
    let address_point = reverse_point(&sumcheck_address_point);
    let (opening_point, log_t) =
        stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<Fr>::zero_selector(&address_point);
    let claims = evaluation_claims(program, stage6, &stage7_values, lagrange_factor)?;

    append_rlc_claims(transcript, &claims);
    let gamma_powers = gamma_powers(transcript, claims.len());
    let joint_claim = claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| claim.value * *gamma)
        .sum();
    let joint_evals = materialize_joint_polynomial(
        commitment_inputs,
        &claims,
        &gamma_powers,
        log_t,
        opening_point.len(),
    )?;
    let joint_poly = Polynomial::new(joint_evals);
    let joint_hint = joint_opening_hint(commitments, &claims, &gamma_powers)?;
    let joint_opening_proof = <jolt_dory::DoryScheme as CommitmentScheme>::open(
        &joint_poly,
        &opening_point,
        joint_claim,
        prover_setup,
        Some(joint_hint),
        transcript,
    );
    <jolt_dory::DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &opening_point,
        &joint_claim,
    );
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

    Ok(<DoryScheme as AdditivelyHomomorphic>::combine_hints(
        hints, &scalars,
    ))
}

fn opening_hint_for_oracle(
    commitments: &commitment_stage::CommitmentArtifacts,
    oracle: &'static str,
) -> Result<DoryHint, JoltEvaluationProveError> {
    commitments
        .hints
        .iter()
        .find(|hint| hint.oracle == oracle)
        .map(|hint| hint.hint.clone())
        .ok_or(JoltEvaluationProveError::MissingOpeningHint { oracle })
}

fn target_len(num_vars: usize) -> Result<usize, JoltEvaluationProveError> {
    if num_vars >= usize::BITS as usize {
        return Err(JoltEvaluationProveError::TargetSizeOverflow { num_vars });
    }
    Ok(1usize << num_vars)
}

fn stage1_outer_proof(artifacts: &stage1::Stage1ExecutionArtifacts<Fr>) -> JoltStageProof {
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

fn stage2_proof(artifacts: &stage2::Stage2ExecutionArtifacts<Fr>) -> JoltStageProof {
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

fn stage3_proof(artifacts: &stage3::Stage3ExecutionArtifacts<Fr>) -> JoltStageProof {
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

fn stage4_proof(artifacts: &stage4::Stage4ExecutionArtifacts<Fr>) -> JoltStageProof {
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

fn stage5_proof(artifacts: &stage5::Stage5ExecutionArtifacts<Fr>) -> JoltStageProof {
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

fn stage6_proof(artifacts: &stage6::Stage6ExecutionArtifacts<Fr>) -> JoltStageProof {
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

fn stage7_proof(artifacts: &stage7::Stage7ExecutionArtifacts<Fr>) -> JoltStageProof {
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

