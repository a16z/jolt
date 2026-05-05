use super::shapes::{Stage23VerifierTypeShape, StageRuntimeVerifierTypeShape};

pub(in crate::protocols::jolt::emit::rust) fn stage_default_transcript_alias(
    stage: usize,
) -> String {
    format!("pub type DefaultStage{stage}Transcript = Blake2bTranscript<Fr>;\n")
}

pub(in crate::protocols::jolt::emit::rust) fn stage23_verifier_type_aliases(
    stage: usize,
    shape: Stage23VerifierTypeShape,
) -> String {
    let verifier_program_plan = if shape.has_opening_equalities() {
        "StageVerifierProgramPlan"
    } else {
        "StageVerifierProgramPlanNoEqualities"
    };
    let common_aliases = if shape.has_opening_equalities() {
        format!(
            r#"pub use super::common::{{
    FieldConstantPlan as Stage{stage}FieldConstantPlan, FieldExprPlan as Stage{stage}FieldExprPlan,
    OpeningBatchPlan as Stage{stage}OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage{stage}OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage{stage}OpeningClaimPlan, OpeningInputPlan as Stage{stage}OpeningInputPlan,
    PointConcatPlan as Stage{stage}PointConcatPlan, PointSlicePlan as Stage{stage}PointSlicePlan,
    ProgramStepPlan as Stage{stage}ProgramStepPlan, StageParams as Stage{stage}Params,
    SumcheckBatchPlan as Stage{stage}SumcheckBatchPlan, SumcheckEvalPlan as Stage{stage}SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage{stage}SumcheckInstanceResultPlan,
    TranscriptSqueezePlan as Stage{stage}TranscriptSqueezePlan,
    VerifierSumcheckClaimPlan as Stage{stage}SumcheckClaimPlan,
    VerifierSumcheckDriverPlan as Stage{stage}SumcheckDriverPlan,
}};
"#
        )
    } else {
        format!(
            r#"pub use super::common::{{
    FieldConstantPlan as Stage{stage}FieldConstantPlan, FieldExprPlan as Stage{stage}FieldExprPlan,
    OpeningBatchPlan as Stage{stage}OpeningBatchPlan, OpeningClaimPlan as Stage{stage}OpeningClaimPlan,
    OpeningInputPlan as Stage{stage}OpeningInputPlan, PointConcatPlan as Stage{stage}PointConcatPlan,
    PointSlicePlan as Stage{stage}PointSlicePlan, ProgramStepPlan as Stage{stage}ProgramStepPlan,
    StageParams as Stage{stage}Params, SumcheckBatchPlan as Stage{stage}SumcheckBatchPlan,
    SumcheckEvalPlan as Stage{stage}SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage{stage}SumcheckInstanceResultPlan,
    TranscriptSqueezePlan as Stage{stage}TranscriptSqueezePlan,
    VerifierSumcheckClaimPlan as Stage{stage}SumcheckClaimPlan,
    VerifierSumcheckDriverPlan as Stage{stage}SumcheckDriverPlan,
}};
"#
        )
    };

    format!(
        r#"pub type DefaultStage{stage}Transcript = Blake2bTranscript<Fr>;

pub type Stage{stage}NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage{stage}SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage{stage}ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage{stage}ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage{stage}Proof<F> = super::common::StageProof<F>;
pub type Stage{stage}OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;
pub type Stage{stage}VerifierProgramPlan = super::common::{verifier_program_plan};

{common_aliases}"#
    )
}

pub(in crate::protocols::jolt::emit::rust) fn stage_verifier_type_aliases(
    stage: usize,
    shape: StageRuntimeVerifierTypeShape,
) -> String {
    let point_zero_alias = if shape.has_point_zeros() {
        format!(
            "    PointZeroPlan as Stage{stage}PointZeroPlan, ProgramStepPlan as Stage{stage}ProgramStepPlan,\n    StageParams as Stage{stage}Params, StageProgramPlan as Stage{stage}CpuProgramPlan,"
        )
    } else {
        format!(
            "    ProgramStepPlan as Stage{stage}ProgramStepPlan, StageParams as Stage{stage}Params,\n    StageProgramPlanNoPointZeros as Stage{stage}CpuProgramPlan,"
        )
    };

    format!(
        r#"pub type Stage{stage}NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage{stage}SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage{stage}ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage{stage}ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage{stage}Proof<F> = super::common::StageProof<F>;
pub type Stage{stage}OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;

pub use super::common::{{
    FieldConstantPlan as Stage{stage}FieldConstantPlan, FieldExprPlan as Stage{stage}FieldExprPlan,
    KernelPlan as Stage{stage}KernelPlan, OpeningBatchPlan as Stage{stage}OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage{stage}OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage{stage}OpeningClaimPlan, OpeningInputPlan as Stage{stage}OpeningInputPlan,
    PointConcatPlan as Stage{stage}PointConcatPlan, PointSlicePlan as Stage{stage}PointSlicePlan,
{point_zero_alias}
    SumcheckBatchPlan as Stage{stage}SumcheckBatchPlan,
    SumcheckClaimPlan as Stage{stage}SumcheckClaimPlan, SumcheckDriverPlan as Stage{stage}SumcheckDriverPlan,
    SumcheckEvalPlan as Stage{stage}SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage{stage}SumcheckInstanceResultPlan,
    TranscriptAbsorbBytesPlan as Stage{stage}TranscriptAbsorbBytesPlan,
    TranscriptSqueezePlan as Stage{stage}TranscriptSqueezePlan,
}};
"#
    )
}

pub(in crate::protocols::jolt::emit::rust) fn stage_runtime_verifier_program_aliases(
    stage: usize,
) -> String {
    format!(
        r#"
pub type DefaultStage{stage}Transcript = Blake2bTranscript<Fr>;
pub type Stage{stage}VerifierProgramPlan = Stage{stage}CpuProgramPlan;
"#
    )
}
