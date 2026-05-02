use jolt_dory::DoryProverSetup;
use jolt_field::Fr;
use jolt_kernels::{stage1, stage2, stage3};
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::{JoltNamedEval, JoltProof, JoltStageProof, JoltSumcheckOutput};

use crate::stages::{commitment as commitment_stage, stage1_outer as stage1_outer_stage, stage2 as stage2_stage, stage3 as stage3_stage};

pub type DefaultJoltTranscript = Blake2bTranscript<Fr>;

pub struct JoltProverInputs<'a, CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor> {
    pub commitment_inputs: &'a mut CommitmentInputs,
    pub prover_setup: &'a DoryProverSetup,
    pub stage1_outer_executor: &'a mut Stage1OuterExecutor,
    pub stage2_executor: &'a mut Stage2Executor,
    pub stage3_executor: &'a mut Stage3Executor,
}

#[derive(Clone, Debug)]
pub struct JoltProverArtifacts {
    pub commitment: commitment_stage::CommitmentArtifacts,
    pub stage1_outer: stage1::Stage1ExecutionArtifacts<Fr>,
    pub stage2: stage2::Stage2ExecutionArtifacts<Fr>,
    pub stage3: stage3::Stage3ExecutionArtifacts<Fr>,
}

#[derive(Debug)]
pub enum JoltProveError {
    Commitment(commitment_stage::CommitmentPhaseError),
    Stage1Outer(stage1::Stage1KernelError),
    Stage2(stage2::Stage2KernelError),
    Stage3(stage3::Stage3KernelError),
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

pub fn prove_jolt<CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor>(
    inputs: JoltProverInputs<'_, CommitmentInputs, Stage1OuterExecutor, Stage2Executor, Stage3Executor>,
    transcript: &mut DefaultJoltTranscript,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    Stage1OuterExecutor: stage1::Stage1KernelExecutor<Fr>,
    Stage2Executor: stage2::Stage2KernelExecutor<Fr>,
    Stage3Executor: stage3::Stage3KernelExecutor<Fr>,
{
    let commitment = commitment_stage::prove_commitment_phase(
        inputs.commitment_inputs,
        inputs.prover_setup,
        transcript,
    )?;
    let stage1_outer = stage1_outer_stage::prove_stage1_outer(inputs.stage1_outer_executor, transcript)?;
    let stage2 = stage2_stage::execute_stage2_prover(inputs.stage2_executor, transcript)?;
    let stage3 = stage3_stage::execute_stage3_prover(inputs.stage3_executor, transcript)?;

    let proof = JoltProof {
        commitments: commitment.commitments.clone(),
        stage1_outer: stage1_outer_proof(&stage1_outer),
        stage2: stage2_proof(&stage2),
        stage3: stage3_proof(&stage3),
    };
    let artifacts = JoltProverArtifacts {
        commitment,
        stage1_outer,
        stage2,
        stage3,
    };
    Ok((proof, artifacts))
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

