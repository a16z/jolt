use std::collections::BTreeMap;

use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::Fr;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};
use jolt_poly::EqPolynomial;
use jolt_transcript::Transcript;

use crate::stages::{commitment as commitment_stage, stage1_outer as stage1_outer_stage, stage2 as stage2_stage, stage3 as stage3_stage, stage4 as stage4_stage, stage5 as stage5_stage, stage6 as stage6_stage, stage7 as stage7_stage, stage8 as stage8_stage};

pub type JoltNamedEval = bolt_verifier_runtime::StageNamedEval<Fr>;
pub type JoltSumcheckOutput = bolt_verifier_runtime::StageSumcheckOutput<Fr>;
pub type JoltStageProof = bolt_verifier_runtime::StageProof<Fr>;

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

pub type JoltStage2RamAccess = crate::stages::stage2::Stage2RamAccess;
pub type JoltStage2RamOutputLayout = crate::stages::stage2::Stage2RamOutputLayout;
pub type JoltStage2RamData<'a> = crate::stages::stage2::Stage2RamData<'a>;
pub type JoltStageChallengeVector = bolt_verifier_runtime::StageChallengeVector<Fr>;
pub type JoltStageExecutionArtifacts = bolt_verifier_runtime::StageExecutionArtifacts<Fr>;
pub type JoltStageOpeningInputValue = bolt_verifier_runtime::StageOpeningInputValue<Fr>;

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
    pub verifier: &'static JoltVerifierProgramPlan,
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
        verifier: &VERIFIER_PROGRAM,
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
    Program(JoltVerifierProgramError),
    Evaluation(JoltEvaluationProofError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltProofSlot {
    Commitments,
    Stage1Outer,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Evaluation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltVerifierStepPlan {
    ReceiveCommitments { slot: JoltProofSlot },
    VerifySumcheckStage { slot: JoltProofSlot },
    VerifyPcsOpening { slot: JoltProofSlot },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltVerifierCheckpoint {
    AfterStage5,
    AfterStage6,
    AfterStage7,
    AfterEvaluation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltEvaluationPolicy {
    Skip,
    VerifyIfPresent,
    Required,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltVerifierTarget {
    pub checkpoint: JoltVerifierCheckpoint,
    pub evaluation: JoltEvaluationPolicy,
}

pub const JOLT_TARGET_THROUGH_STAGE5: JoltVerifierTarget = JoltVerifierTarget {
    checkpoint: JoltVerifierCheckpoint::AfterStage5,
    evaluation: JoltEvaluationPolicy::Skip,
};

pub const JOLT_TARGET_THROUGH_STAGE6: JoltVerifierTarget = JoltVerifierTarget {
    checkpoint: JoltVerifierCheckpoint::AfterStage6,
    evaluation: JoltEvaluationPolicy::Skip,
};

pub const JOLT_TARGET_THROUGH_STAGE7: JoltVerifierTarget = JoltVerifierTarget {
    checkpoint: JoltVerifierCheckpoint::AfterStage7,
    evaluation: JoltEvaluationPolicy::VerifyIfPresent,
};

pub const JOLT_TARGET_FULL: JoltVerifierTarget = JoltVerifierTarget {
    checkpoint: JoltVerifierCheckpoint::AfterEvaluation,
    evaluation: JoltEvaluationPolicy::Required,
};

#[derive(Clone, Copy, Debug)]
pub struct JoltVerifierTargetPlan {
    pub target: JoltVerifierTarget,
    pub steps: &'static [JoltVerifierStepPlan],
}

#[derive(Clone, Copy, Debug)]
pub struct JoltVerifierProgramPlan {
    pub targets: &'static [JoltVerifierTargetPlan],
}

pub const JOLT_VERIFY_THROUGH_STAGE5_STEPS: &[JoltVerifierStepPlan] = &[
    JoltVerifierStepPlan::ReceiveCommitments {
        slot: JoltProofSlot::Commitments,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage1Outer,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage2,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage3,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage4,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage5,
    },
];

pub const JOLT_VERIFY_THROUGH_STAGE6_STEPS: &[JoltVerifierStepPlan] = &[
    JoltVerifierStepPlan::ReceiveCommitments {
        slot: JoltProofSlot::Commitments,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage1Outer,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage2,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage3,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage4,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage5,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage6,
    },
];

pub const JOLT_VERIFY_THROUGH_STAGE7_STEPS: &[JoltVerifierStepPlan] = &[
    JoltVerifierStepPlan::ReceiveCommitments {
        slot: JoltProofSlot::Commitments,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage1Outer,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage2,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage3,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage4,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage5,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage6,
    },
    JoltVerifierStepPlan::VerifySumcheckStage {
        slot: JoltProofSlot::Stage7,
    },
    JoltVerifierStepPlan::VerifyPcsOpening {
        slot: JoltProofSlot::Evaluation,
    },
];

pub const JOLT_VERIFY_FULL_STEPS: &[JoltVerifierStepPlan] = JOLT_VERIFY_THROUGH_STAGE7_STEPS;

pub const JOLT_VERIFIER_TARGETS: &[JoltVerifierTargetPlan] = &[
    JoltVerifierTargetPlan {
        target: JOLT_TARGET_THROUGH_STAGE5,
        steps: JOLT_VERIFY_THROUGH_STAGE5_STEPS,
    },
    JoltVerifierTargetPlan {
        target: JOLT_TARGET_THROUGH_STAGE6,
        steps: JOLT_VERIFY_THROUGH_STAGE6_STEPS,
    },
    JoltVerifierTargetPlan {
        target: JOLT_TARGET_THROUGH_STAGE7,
        steps: JOLT_VERIFY_THROUGH_STAGE7_STEPS,
    },
    JoltVerifierTargetPlan {
        target: JOLT_TARGET_FULL,
        steps: JOLT_VERIFY_FULL_STEPS,
    },
];

pub const VERIFIER_PROGRAM: JoltVerifierProgramPlan = JoltVerifierProgramPlan {
    targets: JOLT_VERIFIER_TARGETS,
};

#[derive(Debug)]
pub enum JoltVerifierProgramError {
    MissingTarget { target: JoltVerifierTarget },
    MissingArtifact { slot: JoltProofSlot },
    UnsupportedStep {
        step: JoltVerifierStepPlan,
        reason: &'static str,
    },
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

macro_rules! define_jolt_verify_error_from {
    ($module:ident, $error_ty:ident, $variant:ident) => {
        impl From<$module::$error_ty> for JoltVerifyError {
            fn from(error: $module::$error_ty) -> Self {
                Self::$variant(error)
            }
        }
    };
}

define_jolt_verify_error_from!(commitment_stage, CommitmentPhaseError, Commitment);
define_jolt_verify_error_from!(stage1_outer_stage, VerifyStage1Error, Stage1Outer);
define_jolt_verify_error_from!(stage2_stage, VerifyStage2Error, Stage2);
define_jolt_verify_error_from!(stage3_stage, VerifyStage3Error, Stage3);
define_jolt_verify_error_from!(stage4_stage, VerifyStage4Error, Stage4);
define_jolt_verify_error_from!(stage5_stage, VerifyStage5Error, Stage5);
define_jolt_verify_error_from!(stage6_stage, VerifyStage6Error, Stage6);
define_jolt_verify_error_from!(stage7_stage, VerifyStage7Error, Stage7);

impl From<JoltVerifierProgramError> for JoltVerifyError {
    fn from(error: JoltVerifierProgramError) -> Self {
        Self::Program(error)
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

pub fn verify_jolt<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> {
    verify_jolt_with_programs(proof, inputs, default_verifier_programs(), transcript)
}

pub fn verify_jolt_prefix<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_prefix_with_programs(proof, inputs, default_verifier_programs(), transcript) }

pub fn verify_jolt_through_stage5<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage5_with_programs(proof, inputs, default_verifier_programs(), transcript) }

pub fn verify_jolt_through_stage6<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage6_with_programs(proof, inputs, default_verifier_programs(), transcript) }

pub fn verify_jolt_through_stage7<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage7_with_programs(proof, inputs, default_verifier_programs(), transcript) }

pub fn verify_jolt_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> {
    verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JOLT_TARGET_FULL)
}

pub fn verify_jolt_through_stage5_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JOLT_TARGET_THROUGH_STAGE5) }

pub fn verify_jolt_through_stage6_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JOLT_TARGET_THROUGH_STAGE6) }

pub fn verify_jolt_through_stage7_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JOLT_TARGET_THROUGH_STAGE7) }

pub fn verify_jolt_prefix_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage7_with_programs(proof, inputs, programs, transcript) }

fn verify_jolt_with_programs_inner<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T, target: JoltVerifierTarget) -> Result<JoltVerificationArtifacts, JoltVerifyError> {
    let _verify_span = tracing::info_span!("bolt.verify").entered();
    execute_jolt_verifier_program(proof, inputs, programs, transcript, target)
}

fn execute_jolt_verifier_program<T>(
    proof: &JoltProof,
    inputs: JoltVerifierInputs<'_>,
    programs: JoltVerifierPrograms,
    transcript: &mut T,
    target: JoltVerifierTarget,
) -> Result<JoltVerificationArtifacts, JoltVerifyError>
where
    T: Transcript<Challenge = Fr>,
{
    let target_plan = programs
        .verifier
        .targets
        .iter()
        .find(|plan| plan.target == target)
        .ok_or(JoltVerifierProgramError::MissingTarget { target })?;
    let mut artifacts = JoltArtifactStore::default();
    for step in target_plan.steps {
        execute_jolt_verifier_step(
            *step,
            proof,
            inputs,
            programs,
            transcript,
            target_plan,
            &mut artifacts,
        )?;
    }
    artifacts.into_public_artifacts().map_err(JoltVerifyError::from)
}

fn execute_jolt_verifier_step<T>(
    step: JoltVerifierStepPlan,
    proof: &JoltProof,
    inputs: JoltVerifierInputs<'_>,
    programs: JoltVerifierPrograms,
    transcript: &mut T,
    target_plan: &JoltVerifierTargetPlan,
    artifacts: &mut JoltArtifactStore,
) -> Result<(), JoltVerifyError>
where
    T: Transcript<Challenge = Fr>,
{
    match step {
        JoltVerifierStepPlan::ReceiveCommitments {
            slot: JoltProofSlot::Commitments,
        } => {
            artifacts.commitment = Some(commitment_stage::verify_commitment_phase_with_program(
                programs.commitment,
                &proof.commitments,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage1Outer,
        } => {
            artifacts.stage1_outer = Some(stage1_outer_stage::verify_stage1_outer_with_program(
                programs.stage1_outer,
                &proof.stage1_outer,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage2,
        } => {
            artifacts.stage2 = Some(stage2_stage::verify_stage2_with_program(
                programs.stage2,
                &proof.stage2,
                inputs.stage2_openings,
                inputs.stage2_ram,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage3,
        } => {
            artifacts.stage3 = Some(stage3_stage::verify_stage3_with_program(
                programs.stage3,
                &proof.stage3,
                inputs.stage3_openings,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage4,
        } => {
            artifacts.stage4 = Some(stage4_stage::verify_stage4_with_program(
                programs.stage4,
                &proof.stage4,
                inputs.stage4_openings,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage5,
        } => {
            artifacts.stage5 = Some(stage5_stage::verify_stage5_with_program(
                programs.stage5,
                &proof.stage5,
                inputs.stage5_openings,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage6,
        } => {
            artifacts.stage6 = Some(stage6_stage::verify_stage6_with_program(
                programs.stage6,
                &proof.stage6,
                inputs.stage6_openings,
                inputs.stage6_data,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifySumcheckStage {
            slot: JoltProofSlot::Stage7,
        } => {
            artifacts.stage7 = Some(stage7_stage::verify_stage7_with_program(
                programs.stage7,
                &proof.stage7,
                inputs.stage7_openings,
                transcript,
            )?);
        }
        JoltVerifierStepPlan::VerifyPcsOpening {
            slot: JoltProofSlot::Evaluation,
        } => {
            verify_jolt_evaluation_step(
                target_plan.target.evaluation,
                proof,
                inputs,
                programs,
                transcript,
                artifacts,
            )?;
        }
        step => {
            return Err(JoltVerifierProgramError::UnsupportedStep {
                step,
                reason: "step kind and proof slot are incompatible",
            }
            .into());
        }
    }
    Ok(())
}

fn verify_jolt_evaluation_step<T>(
    evaluation: JoltEvaluationPolicy,
    proof: &JoltProof,
    inputs: JoltVerifierInputs<'_>,
    programs: JoltVerifierPrograms,
    transcript: &mut T,
    artifacts: &JoltArtifactStore,
) -> Result<(), JoltVerifyError>
where
    T: Transcript<Challenge = Fr>,
{
    if evaluation == JoltEvaluationPolicy::Skip {
        return Ok(());
    }
    match (&proof.evaluation, inputs.evaluation_setup) {
        (Some(evaluation_proof), Some(setup)) => {
            verify_jolt_evaluation_proof(
                programs.stage8,
                evaluation_proof,
                artifacts.commitment()?,
                &proof.stage6,
                &proof.stage7,
                inputs.stage7_openings,
                setup,
                transcript,
            )?;
        }
        (Some(_), None) => return Err(JoltEvaluationProofError::MissingVerifierSetup.into()),
        (None, Some(_)) => return Err(JoltEvaluationProofError::MissingProof.into()),
        (None, None) if evaluation == JoltEvaluationPolicy::Required => {
            return Err(JoltEvaluationProofError::MissingProof.into());
        }
        (None, None) => {}
    }
    Ok(())
}

#[derive(Default)]
struct JoltArtifactStore {
    commitment: Option<commitment_stage::CommitmentArtifacts>,
    stage1_outer: Option<stage1_outer_stage::Stage1ExecutionArtifacts<Fr>>,
    stage2: Option<stage2_stage::Stage2ExecutionArtifacts<Fr>>,
    stage3: Option<stage3_stage::Stage3ExecutionArtifacts<Fr>>,
    stage4: Option<stage4_stage::Stage4ExecutionArtifacts<Fr>>,
    stage5: Option<stage5_stage::Stage5ExecutionArtifacts<Fr>>,
    stage6: Option<stage6_stage::Stage6ExecutionArtifacts<Fr>>,
    stage7: Option<stage7_stage::Stage7ExecutionArtifacts<Fr>>,
}

impl JoltArtifactStore {
    fn commitment(
        &self,
    ) -> Result<&commitment_stage::CommitmentArtifacts, JoltVerifierProgramError> {
        self.commitment
            .as_ref()
            .ok_or(JoltVerifierProgramError::MissingArtifact {
                slot: JoltProofSlot::Commitments,
            })
    }

    fn into_public_artifacts(self) -> Result<JoltVerificationArtifacts, JoltVerifierProgramError> {
        Ok(JoltVerificationArtifacts {
            commitment: required_artifact(self.commitment, JoltProofSlot::Commitments)?,
            stage1_outer: required_artifact(self.stage1_outer, JoltProofSlot::Stage1Outer)?,
            stage2: required_artifact(self.stage2, JoltProofSlot::Stage2)?,
            stage3: required_artifact(self.stage3, JoltProofSlot::Stage3)?,
            stage4: required_artifact(self.stage4, JoltProofSlot::Stage4)?,
            stage5: required_artifact(self.stage5, JoltProofSlot::Stage5)?,
            stage6: self.stage6.unwrap_or_default(),
            stage7: self.stage7.unwrap_or_default(),
        })
    }
}

fn required_artifact<T>(
    artifact: Option<T>,
    slot: JoltProofSlot,
) -> Result<T, JoltVerifierProgramError> {
    artifact.ok_or(JoltVerifierProgramError::MissingArtifact { slot })
}

impl<'a> JoltVerifierInputs<'a> {
    pub fn through_stage5(mut self) -> Self { self.stage6_openings = &[]; self.stage7_openings = &[]; self.evaluation_setup = None; self }
    pub fn through_stage6(mut self) -> Self { self.stage7_openings = &[]; self.evaluation_setup = None; self }
    pub fn through_stage7(mut self) -> Self { self.evaluation_setup = None; self }
    pub fn full(mut self, evaluation_setup: &'a DoryVerifierSetup) -> Self { self.evaluation_setup = Some(evaluation_setup); self }
}

pub type JoltStage6BytecodeEntry = crate::stages::stage6::Stage6BytecodeEntry;
pub type JoltStage6BytecodeReadRafData = crate::stages::stage6::Stage6BytecodeReadRafData;
pub type JoltStage6VerifierData = crate::stages::stage6::Stage6VerifierData;

impl stage8_stage::Stage8NamedEvalView<Fr> for JoltNamedEval {
    fn name(&self) -> &'static str {
        self.name
    }

    fn value(&self) -> Fr {
        self.value
    }
}

impl stage8_stage::Stage8SumcheckOutputView<Fr> for JoltSumcheckOutput {
    type Eval = JoltNamedEval;

    fn point(&self) -> &[Fr] {
        &self.point
    }

    fn evals(&self) -> &[Self::Eval] {
        &self.evals
    }
}

impl stage8_stage::Stage8OpeningInputView<Fr>
    for stage7_stage::Stage7OpeningInputValue<Fr>
{
    fn symbol(&self) -> &'static str {
        self.symbol
    }

    fn point(&self) -> &[Fr] {
        &self.point
    }
}

impl From<stage8_stage::Stage8EvaluationOpeningPointError> for JoltEvaluationProofError {
    fn from(error: stage8_stage::Stage8EvaluationOpeningPointError) -> Self {
        match error {
            stage8_stage::Stage8EvaluationOpeningPointError::MissingStage7EvaluationPoint => {
                Self::MissingStage7EvaluationPoint
            }
            stage8_stage::Stage8EvaluationOpeningPointError::InvalidPointLength {
                artifact,
                expected,
                actual,
            } => Self::InvalidPointLength {
                artifact,
                expected,
                actual,
            },
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "generated verifier entry point follows the Jolt proof artifact boundary"
)]
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
    let _state_span = tracing::info_span!("bolt.verify.evaluation_state").entered();
    let state =
        evaluation_proof_state(program, commitments, stage6, stage7, stage7_openings, transcript)?;
    drop(_state_span);
    let _dory_verify_span = tracing::info_span!("bolt.verify.dory_verify").entered();
    <DoryScheme as CommitmentScheme>::verify(
        &state.joint_commitment,
        &state.opening_point,
        state.joint_claim,
        &proof.joint_opening_proof,
        verifier_setup,
        transcript,
    )?;
    drop(_dory_verify_span);
    let _bind_span = tracing::info_span!("bolt.verify.bind_opening_inputs").entered();
    <DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &state.opening_point,
        &state.joint_claim,
    );
    drop(_bind_span);
    Ok(())
}

struct EvaluationProofState {
    opening_point: Vec<Fr>,
    joint_claim: Fr,
    joint_commitment: DoryCommitment,
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
    let (sumcheck_address_point, stage7_values) =
        stage8_stage::stage7_claim_values(program, &stage7.sumchecks)
            .ok_or(JoltEvaluationProofError::MissingStage7RaEval)?;
    let address_point = stage8_stage::reverse_point(&sumcheck_address_point);
    let (opening_point, _) =
        stage8_stage::stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<Fr>::zero_selector(&address_point);
    let claims =
        stage8_stage::evaluation_claims(program, &stage6.sumchecks, &stage7_values, lagrange_factor)
            .map_err(|error| JoltEvaluationProofError::MissingStageEval {
                stage: error.stage,
                eval: error.eval,
            })?;

    stage8_stage::append_rlc_claims(transcript, &claims);
    let gamma_powers = stage8_stage::gamma_powers(transcript, claims.len());
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

fn joint_commitment(
    commitments: &commitment_stage::CommitmentArtifacts,
    claims: &[stage8_stage::Stage8EvaluationClaim<Fr>],
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
