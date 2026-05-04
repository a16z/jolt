use std::collections::BTreeMap;

use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field, Fr};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::stages::{commitment as commitment_stage, stage1_outer as stage1_outer_stage, stage2 as stage2_stage, stage3 as stage3_stage, stage4 as stage4_stage, stage5 as stage5_stage, stage6 as stage6_stage, stage7 as stage7_stage, stage8 as stage8_stage};

#[derive(Clone, Debug)]
pub struct JoltNamedEval {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: Fr,
}

#[derive(Clone, Debug)]
pub struct JoltSumcheckOutput {
    pub driver: &'static str,
    pub point: Vec<Fr>,
    pub evals: Vec<JoltNamedEval>,
    pub proof: SumcheckProof<Fr>,
}

#[derive(Clone, Debug, Default)]
pub struct JoltStageProof {
    pub sumchecks: Vec<JoltSumcheckOutput>,
}

#[derive(Clone, Debug)]
pub struct JoltProof {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub stage1_outer: JoltStageProof,
    pub stage2: JoltStageProof,
    pub stage3: JoltStageProof,
    pub stage4: JoltStageProof,
    pub stage5: JoltStageProof,
    pub stage6: JoltStageProof,
    pub stage7: JoltStageProof,
    pub evaluation: Option<JoltEvaluationProof>,
}

#[derive(Clone, Debug)]
pub struct JoltEvaluationProof {
    pub joint_opening_proof: DoryProof,
}

#[derive(Clone, Copy)]
pub struct JoltVerifierInputs<'a> {
    pub stage2_openings: &'a [stage2_stage::Stage2OpeningInputValue<Fr>],
    pub stage2_ram: Option<&'a stage2_stage::Stage2RamData<'a>>,
    pub stage3_openings: &'a [stage3_stage::Stage3OpeningInputValue<Fr>],
    pub stage4_openings: &'a [stage4_stage::Stage4OpeningInputValue<Fr>],
    pub stage5_openings: &'a [stage5_stage::Stage5OpeningInputValue<Fr>],
    pub stage6_openings: &'a [stage6_stage::Stage6OpeningInputValue<Fr>],
    pub stage6_data: Option<&'a stage6_stage::Stage6VerifierData>,
    pub stage7_openings: &'a [stage7_stage::Stage7OpeningInputValue<Fr>],
    pub evaluation_setup: Option<&'a DoryVerifierSetup>,
}

#[derive(Clone, Copy, Debug)]
pub struct JoltVerifierPrograms {
    pub commitment: &'static commitment_stage::CommitmentVerifierProgramPlan,
    pub stage1_outer: &'static stage1_outer_stage::Stage1VerifierProgramPlan,
    pub stage2: &'static stage2_stage::Stage2VerifierProgramPlan,
    pub stage3: &'static stage3_stage::Stage3VerifierProgramPlan,
    pub stage4: &'static stage4_stage::Stage4VerifierProgramPlan,
    pub stage5: &'static stage5_stage::Stage5VerifierProgramPlan,
    pub stage6: &'static stage6_stage::Stage6VerifierProgramPlan,
    pub stage7: &'static stage7_stage::Stage7VerifierProgramPlan,
    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,
}

pub fn default_verifier_programs() -> JoltVerifierPrograms {
    JoltVerifierPrograms {
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
pub struct JoltVerificationArtifacts {
    pub commitment: commitment_stage::CommitmentArtifacts,
    pub stage1_outer: stage1_outer_stage::Stage1ExecutionArtifacts<Fr>,
    pub stage2: stage2_stage::Stage2ExecutionArtifacts<Fr>,
    pub stage3: stage3_stage::Stage3ExecutionArtifacts<Fr>,
    pub stage4: stage4_stage::Stage4ExecutionArtifacts<Fr>,
    pub stage5: stage5_stage::Stage5ExecutionArtifacts<Fr>,
    pub stage6: stage6_stage::Stage6ExecutionArtifacts<Fr>,
    pub stage7: stage7_stage::Stage7ExecutionArtifacts<Fr>,
}

#[derive(Debug)]
pub enum JoltVerifyError {
    Commitment(commitment_stage::CommitmentPhaseError),
    Stage1Outer(stage1_outer_stage::VerifyStage1Error),
    Stage2(stage2_stage::VerifyStage2Error),
    Stage3(stage3_stage::VerifyStage3Error),
    Stage4(stage4_stage::VerifyStage4Error),
    Stage5(stage5_stage::VerifyStage5Error),
    Stage6(stage6_stage::VerifyStage6Error),
    Stage7(stage7_stage::VerifyStage7Error),
    Evaluation(JoltEvaluationProofError),
}

#[derive(Debug)]
pub enum JoltEvaluationProofError {
    MissingProof,
    MissingVerifierSetup,
    MissingStageEval { stage: &'static str, eval: &'static str },
    MissingStage7RaEval,
    MissingStage7EvaluationPoint,
    MissingCommitment { oracle: &'static str },
    InvalidPointLength {
        artifact: &'static str,
        expected: usize,
        actual: usize,
    },
    Opening(OpeningsError),
}

impl From<commitment_stage::CommitmentPhaseError> for JoltVerifyError {
    fn from(error: commitment_stage::CommitmentPhaseError) -> Self {
        Self::Commitment(error)
    }
}

impl From<stage1_outer_stage::VerifyStage1Error> for JoltVerifyError {
    fn from(error: stage1_outer_stage::VerifyStage1Error) -> Self {
        Self::Stage1Outer(error)
    }
}

impl From<stage2_stage::VerifyStage2Error> for JoltVerifyError {
    fn from(error: stage2_stage::VerifyStage2Error) -> Self {
        Self::Stage2(error)
    }
}

impl From<stage3_stage::VerifyStage3Error> for JoltVerifyError {
    fn from(error: stage3_stage::VerifyStage3Error) -> Self {
        Self::Stage3(error)
    }
}

impl From<stage4_stage::VerifyStage4Error> for JoltVerifyError {
    fn from(error: stage4_stage::VerifyStage4Error) -> Self {
        Self::Stage4(error)
    }
}

impl From<stage5_stage::VerifyStage5Error> for JoltVerifyError {
    fn from(error: stage5_stage::VerifyStage5Error) -> Self {
        Self::Stage5(error)
    }
}

impl From<stage6_stage::VerifyStage6Error> for JoltVerifyError {
    fn from(error: stage6_stage::VerifyStage6Error) -> Self {
        Self::Stage6(error)
    }
}

impl From<stage7_stage::VerifyStage7Error> for JoltVerifyError {
    fn from(error: stage7_stage::VerifyStage7Error) -> Self {
        Self::Stage7(error)
    }
}

impl From<JoltEvaluationProofError> for JoltVerifyError {
    fn from(error: JoltEvaluationProofError) -> Self {
        Self::Evaluation(error)
    }
}

impl From<OpeningsError> for JoltEvaluationProofError {
    fn from(error: OpeningsError) -> Self {
        Self::Opening(error)
    }
}

pub fn verify_jolt<T>(
    proof: &JoltProof,
    inputs: JoltVerifierInputs<'_>,
    transcript: &mut T,
) -> Result<JoltVerificationArtifacts, JoltVerifyError>
where
    T: Transcript<Challenge = Fr>,
{
    verify_jolt_with_programs(proof, inputs, default_verifier_programs(), transcript)
}

pub fn verify_jolt_with_programs<T>(
    proof: &JoltProof,
    inputs: JoltVerifierInputs<'_>,
    programs: JoltVerifierPrograms,
    transcript: &mut T,
) -> Result<JoltVerificationArtifacts, JoltVerifyError>
where
    T: Transcript<Challenge = Fr>,
{
    let commitment = commitment_stage::verify_commitment_phase_with_program(programs.commitment, &proof.commitments, transcript)?;
    let stage1_outer_proof = stage1_outer_proof(&proof.stage1_outer);
    let stage2_proof = stage2_proof(&proof.stage2);
    let stage3_proof = stage3_proof(&proof.stage3);
    let stage4_proof = stage4_proof(&proof.stage4);
    let stage5_proof = stage5_proof(&proof.stage5);
    let stage6_proof = stage6_proof(&proof.stage6);
    let stage7_proof = stage7_proof(&proof.stage7);

    let stage1_outer = stage1_outer_stage::verify_stage1_outer_with_program(programs.stage1_outer, &stage1_outer_proof, transcript)?;
    let stage2 = stage2_stage::verify_stage2_with_program(programs.stage2, &stage2_proof, inputs.stage2_openings, inputs.stage2_ram, transcript)?;
    let stage3 = stage3_stage::verify_stage3_with_program(programs.stage3, &stage3_proof, inputs.stage3_openings, transcript)?;
    let stage4 = stage4_stage::verify_stage4_with_program(programs.stage4, &stage4_proof, inputs.stage4_openings, transcript)?;
    let stage5 = stage5_stage::verify_stage5_with_program(programs.stage5, &stage5_proof, inputs.stage5_openings, transcript)?;
    let stage6 = stage6_stage::verify_stage6_with_program(programs.stage6, &stage6_proof, inputs.stage6_openings, inputs.stage6_data, transcript)?;
    let stage7 = stage7_stage::verify_stage7_with_program(programs.stage7, &stage7_proof, inputs.stage7_openings, transcript)?;
    match (&proof.evaluation, inputs.evaluation_setup) {
        (Some(evaluation), Some(setup)) => {
            verify_jolt_evaluation_proof(
                programs.stage8,
                evaluation,
                &commitment,
                &proof.stage6,
                &proof.stage7,
                inputs.stage7_openings,
                setup,
                transcript,
            )?;
        }
        (Some(_), None) => return Err(JoltEvaluationProofError::MissingVerifierSetup.into()),
        (None, Some(_)) => return Err(JoltEvaluationProofError::MissingProof.into()),
        (None, None) => {}
    }

    Ok(JoltVerificationArtifacts {
        commitment,
        stage1_outer,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7,
    })
}

pub fn verify_jolt_evaluation_proof<T>(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    proof: &JoltEvaluationProof,
    commitments: &commitment_stage::CommitmentArtifacts,
    stage6: &JoltStageProof,
    stage7: &JoltStageProof,
    stage7_openings: &[stage7_stage::Stage7OpeningInputValue<Fr>],
    verifier_setup: &DoryVerifierSetup,
    transcript: &mut T,
) -> Result<(), JoltEvaluationProofError>
where
    T: Transcript<Challenge = Fr>,
{
    let state =
        evaluation_proof_state(program, commitments, stage6, stage7, stage7_openings, transcript)?;
    <DoryScheme as CommitmentScheme>::verify(
        &state.joint_commitment,
        &state.opening_point,
        state.joint_claim,
        &proof.joint_opening_proof,
        verifier_setup,
        transcript,
    )?;
    <DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &state.opening_point,
        &state.joint_claim,
    );
    Ok(())
}

struct EvaluationProofState {
    opening_point: Vec<Fr>,
    joint_claim: Fr,
    joint_commitment: DoryCommitment,
}

struct EvaluationClaim {
    oracle: &'static str,
    value: Fr,
}

fn evaluation_proof_state<T>(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    commitments: &commitment_stage::CommitmentArtifacts,
    stage6: &JoltStageProof,
    stage7: &JoltStageProof,
    stage7_openings: &[stage7_stage::Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<EvaluationProofState, JoltEvaluationProofError>
where
    T: Transcript<Challenge = Fr>,
{
    let (sumcheck_address_point, stage7_values) = stage7_claim_values(program, stage7)?;
    let address_point = reverse_point(&sumcheck_address_point);
    let opening_point = stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<Fr>::zero_selector(&address_point);
    let claims = evaluation_claims(program, stage6, &stage7_values, lagrange_factor)?;

    append_rlc_claims(transcript, &claims);
    let gamma_powers = gamma_powers(transcript, claims.len());
    let joint_claim = claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| claim.value * *gamma)
        .sum();
    let joint_commitment = joint_commitment(commitments, &claims, &gamma_powers)?;

    Ok(EvaluationProofState {
        opening_point,
        joint_claim,
        joint_commitment,
    })
}

fn stage_eval(
    proof: &JoltStageProof,
    stage: &'static str,
    eval_name: &'static str,
) -> Result<Fr, JoltEvaluationProofError> {
    for output in &proof.sumchecks {
        if let Some(eval) = output.evals.iter().find(|eval| eval.name == eval_name) {
            return Ok(eval.value);
        }
    }
    Err(JoltEvaluationProofError::MissingStageEval {
        stage,
        eval: eval_name,
    })
}

fn evaluation_claims(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    stage6: &JoltStageProof,
    stage7_values: &BTreeMap<&'static str, Fr>,
    lagrange_factor: Fr,
) -> Result<Vec<EvaluationClaim>, JoltEvaluationProofError> {
    let mut claims = Vec::with_capacity(program.opening_claims.len());
    for plan in program.opening_claims {
        let value = match plan.source_stage {
            "stage6" => stage_eval(stage6, plan.source_stage, plan.source_claim)? * lagrange_factor,
            "stage7" => *stage7_values.get(plan.source_claim).ok_or(
                JoltEvaluationProofError::MissingStageEval {
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                },
            )?,
            _ => {
                return Err(JoltEvaluationProofError::MissingStageEval {
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                });
            }
        };
        claims.push(EvaluationClaim {
            oracle: plan.oracle,
            value,
        });
    }
    Ok(claims)
}

fn stage7_claim_values(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    proof: &JoltStageProof,
) -> Result<(Vec<Fr>, BTreeMap<&'static str, Fr>), JoltEvaluationProofError> {
    let stage7_plans = program
        .opening_claims
        .iter()
        .filter(|plan| plan.source_stage == "stage7")
        .collect::<Vec<_>>();
    for output in &proof.sumchecks {
        let mut values = BTreeMap::new();
        for plan in &stage7_plans {
            if let Some(eval) = output.evals.iter().find(|eval| eval.name == plan.source_claim) {
                let _ = values.insert(plan.source_claim, eval.value);
            }
        }
        if values.len() == stage7_plans.len() {
            return Ok((output.point.clone(), values));
        }
    }
    Err(JoltEvaluationProofError::MissingStage7RaEval)
}

fn reverse_point(point: &[Fr]) -> Vec<Fr> {
    point.iter().rev().copied().collect()
}

fn stage7_evaluation_opening_point(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    address_point: &[Fr],
    stage7_openings: &[stage7_stage::Stage7OpeningInputValue<Fr>],
) -> Result<Vec<Fr>, JoltEvaluationProofError> {
    let cycle_source_symbol = program.evaluation_point_source.source_claim;
    let cycle_source = stage7_openings
        .iter()
        .find(|input| input.symbol == cycle_source_symbol)
        .ok_or(JoltEvaluationProofError::MissingStage7EvaluationPoint)?;
    if cycle_source.point.len() < address_point.len() {
        return Err(JoltEvaluationProofError::InvalidPointLength {
            artifact: cycle_source_symbol,
            expected: address_point.len(),
            actual: cycle_source.point.len(),
        });
    }
    let mut point = Vec::with_capacity(cycle_source.point.len());
    point.extend_from_slice(address_point);
    point.extend_from_slice(&cycle_source.point[address_point.len()..]);
    Ok(point)
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

fn joint_commitment(
    commitments: &commitment_stage::CommitmentArtifacts,
    claims: &[EvaluationClaim],
    gamma_powers: &[Fr],
) -> Result<DoryCommitment, JoltEvaluationProofError> {
    let mut coefficients = BTreeMap::<&'static str, Fr>::new();
    for (claim, gamma) in claims.iter().zip(gamma_powers) {
        let coefficient = coefficients.entry(claim.oracle).or_insert(Fr::from_u64(0));
        *coefficient += *gamma;
    }
    let mut commitment_values = Vec::with_capacity(coefficients.len());
    let mut scalars = Vec::with_capacity(coefficients.len());
    for (oracle, coefficient) in coefficients {
        commitment_values.push(commitment_for_oracle(commitments, oracle)?);
        scalars.push(coefficient);
    }
    Ok(<DoryScheme as AdditivelyHomomorphic>::combine(
        &commitment_values,
        &scalars,
    ))
}

fn commitment_for_oracle(
    commitments: &commitment_stage::CommitmentArtifacts,
    oracle: &'static str,
) -> Result<DoryCommitment, JoltEvaluationProofError> {
    for (record, commitment) in commitments.records.iter().zip(&commitments.commitments) {
        if record.oracle == oracle {
            return commitment
                .clone()
                .ok_or(JoltEvaluationProofError::MissingCommitment { oracle });
        }
    }
    Err(JoltEvaluationProofError::MissingCommitment { oracle })
}

fn stage1_outer_proof(proof: &JoltStageProof) -> stage1_outer_stage::Stage1Proof<Fr> {
    stage1_outer_stage::Stage1Proof {
        sumchecks: proof.sumchecks.iter().map(stage1_outer_sumcheck).collect(),
    }
}

fn stage1_outer_sumcheck(output: &JoltSumcheckOutput) -> stage1_outer_stage::Stage1SumcheckOutput<Fr> {
    stage1_outer_stage::Stage1SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage1_outer_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage1_outer_eval(eval: &JoltNamedEval) -> stage1_outer_stage::Stage1NamedEval<Fr> {
    stage1_outer_stage::Stage1NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

fn stage2_proof(proof: &JoltStageProof) -> stage2_stage::Stage2Proof<Fr> {
    stage2_stage::Stage2Proof {
        sumchecks: proof.sumchecks.iter().map(stage2_sumcheck).collect(),
    }
}

fn stage2_sumcheck(output: &JoltSumcheckOutput) -> stage2_stage::Stage2SumcheckOutput<Fr> {
    stage2_stage::Stage2SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage2_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage2_eval(eval: &JoltNamedEval) -> stage2_stage::Stage2NamedEval<Fr> {
    stage2_stage::Stage2NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

fn stage3_proof(proof: &JoltStageProof) -> stage3_stage::Stage3Proof<Fr> {
    stage3_stage::Stage3Proof {
        sumchecks: proof.sumchecks.iter().map(stage3_sumcheck).collect(),
    }
}

fn stage3_sumcheck(output: &JoltSumcheckOutput) -> stage3_stage::Stage3SumcheckOutput<Fr> {
    stage3_stage::Stage3SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage3_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage3_eval(eval: &JoltNamedEval) -> stage3_stage::Stage3NamedEval<Fr> {
    stage3_stage::Stage3NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

fn stage4_proof(proof: &JoltStageProof) -> stage4_stage::Stage4Proof<Fr> {
    stage4_stage::Stage4Proof {
        sumchecks: proof.sumchecks.iter().map(stage4_sumcheck).collect(),
    }
}

fn stage4_sumcheck(output: &JoltSumcheckOutput) -> stage4_stage::Stage4SumcheckOutput<Fr> {
    stage4_stage::Stage4SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage4_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage4_eval(eval: &JoltNamedEval) -> stage4_stage::Stage4NamedEval<Fr> {
    stage4_stage::Stage4NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

fn stage5_proof(proof: &JoltStageProof) -> stage5_stage::Stage5Proof<Fr> {
    stage5_stage::Stage5Proof {
        sumchecks: proof.sumchecks.iter().map(stage5_sumcheck).collect(),
    }
}

fn stage5_sumcheck(output: &JoltSumcheckOutput) -> stage5_stage::Stage5SumcheckOutput<Fr> {
    stage5_stage::Stage5SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage5_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage5_eval(eval: &JoltNamedEval) -> stage5_stage::Stage5NamedEval<Fr> {
    stage5_stage::Stage5NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

fn stage6_proof(proof: &JoltStageProof) -> stage6_stage::Stage6Proof<Fr> {
    stage6_stage::Stage6Proof {
        sumchecks: proof.sumchecks.iter().map(stage6_sumcheck).collect(),
    }
}

fn stage6_sumcheck(output: &JoltSumcheckOutput) -> stage6_stage::Stage6SumcheckOutput<Fr> {
    stage6_stage::Stage6SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage6_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage6_eval(eval: &JoltNamedEval) -> stage6_stage::Stage6NamedEval<Fr> {
    stage6_stage::Stage6NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

fn stage7_proof(proof: &JoltStageProof) -> stage7_stage::Stage7Proof<Fr> {
    stage7_stage::Stage7Proof {
        sumchecks: proof.sumchecks.iter().map(stage7_sumcheck).collect(),
    }
}

fn stage7_sumcheck(output: &JoltSumcheckOutput) -> stage7_stage::Stage7SumcheckOutput<Fr> {
    stage7_stage::Stage7SumcheckOutput {
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage7_eval).collect(),
        proof: output.proof.clone(),
    }
}

fn stage7_eval(eval: &JoltNamedEval) -> stage7_stage::Stage7NamedEval<Fr> {
    stage7_stage::Stage7NamedEval {
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }
}

