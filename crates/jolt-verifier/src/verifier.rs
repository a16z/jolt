use jolt_dory::DoryCommitment;
use jolt_field::Fr;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;

use crate::stages::{commitment as commitment_stage, stage1_outer as stage1_outer_stage, stage2 as stage2_stage, stage3 as stage3_stage};

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
}

#[derive(Clone, Copy, Debug)]
pub struct JoltVerifierInputs<'a> {
    pub stage2_openings: &'a [stage2_stage::Stage2OpeningInputValue<Fr>],
    pub stage2_ram: Option<&'a stage2_stage::Stage2RamData<'a>>,
    pub stage3_openings: &'a [stage3_stage::Stage3OpeningInputValue<Fr>],
}

#[derive(Clone, Debug)]
pub struct JoltVerificationArtifacts {
    pub commitment: commitment_stage::CommitmentArtifacts,
    pub stage1_outer: stage1_outer_stage::Stage1ExecutionArtifacts<Fr>,
    pub stage2: stage2_stage::Stage2ExecutionArtifacts<Fr>,
    pub stage3: stage3_stage::Stage3ExecutionArtifacts<Fr>,
}

#[derive(Debug)]
pub enum JoltVerifyError {
    Commitment(commitment_stage::CommitmentPhaseError),
    Stage1Outer(stage1_outer_stage::VerifyStage1Error),
    Stage2(stage2_stage::VerifyStage2Error),
    Stage3(stage3_stage::VerifyStage3Error),
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

pub fn verify_jolt<T>(
    proof: &JoltProof,
    inputs: JoltVerifierInputs<'_>,
    transcript: &mut T,
) -> Result<JoltVerificationArtifacts, JoltVerifyError>
where
    T: Transcript<Challenge = Fr>,
{
    let commitment = commitment_stage::verify_commitment_phase(&proof.commitments, transcript)?;
    let stage1_outer_proof = stage1_outer_proof(&proof.stage1_outer);
    let stage2_proof = stage2_proof(&proof.stage2);
    let stage3_proof = stage3_proof(&proof.stage3);

    let stage1_outer = stage1_outer_stage::verify_stage1_outer(&stage1_outer_proof, transcript)?;
    let stage2 = stage2_stage::verify_stage2(&stage2_proof, inputs.stage2_openings, inputs.stage2_ram, transcript)?;
    let stage3 = stage3_stage::verify_stage3(&stage3_proof, inputs.stage3_openings, transcript)?;

    Ok(JoltVerificationArtifacts {
        commitment,
        stage1_outer,
        stage2,
        stage3,
    })
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

