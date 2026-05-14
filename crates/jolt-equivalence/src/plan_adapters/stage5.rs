use bolt::protocols::jolt::Stage5CpuProgram as CompilerStage5CpuProgram;
use jolt_kernels::stage5 as kernel_stage5;

define_stage_adapter!(
    kernel,
    leak_stage5_program,
    CompilerStage5CpuProgram,
    kernel_stage5,
    Stage5CpuProgramPlan,
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
    Stage5OpeningBatchPlan
);
