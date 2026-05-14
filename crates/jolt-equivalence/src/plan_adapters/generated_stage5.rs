use bolt::protocols::jolt::Stage5CpuProgram as CompilerStage5CpuProgram;
use jolt_verifier::stages::stage5 as generated_stage5;

define_stage_adapter!(
    generated,
    leak_generated_stage5_verifier_program,
    CompilerStage5CpuProgram,
    generated_stage5,
    Stage5VerifierProgramPlan,
    Stage5Params,
    Stage5ProgramStepPlan,
    Stage5TranscriptSqueezePlan,
    Stage5TranscriptAbsorbBytesPlan,
    Stage5OpeningInputPlan,
    Stage5FieldConstantPlan,
    Stage5FieldExprPlan,
    Stage5KernelPlan,
    Stage5SumcheckClaimPlan,
    Stage5SumcheckBatchPlan,
    Stage5SumcheckDriverPlan,
    Stage5SumcheckInstanceResultPlan,
    Stage5SumcheckEvalPlan,
    Stage5PointSlicePlan,
    Stage5PointConcatPlan,
    Stage5OpeningClaimPlan,
    Stage5OpeningClaimEqualityPlan,
    Stage5OpeningBatchPlan,
    empty_output_claims = yes
);
