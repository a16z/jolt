use bolt::protocols::jolt::Stage6CpuProgram as CompilerStage6CpuProgram;
use jolt_kernels::stage6 as kernel_stage6;

define_stage_adapter!(
    kernel,
    leak_stage6_program,
    CompilerStage6CpuProgram,
    kernel_stage6,
    Stage6CpuProgramPlan,
    Stage6Params,
    Stage6ProgramStepPlan,
    Stage6TranscriptSqueezePlan,
    Stage6TranscriptAbsorbBytesPlan,
    Stage6OpeningInputPlan,
    Stage6FieldConstantPlan,
    Stage6FieldExprPlan,
    Stage6KernelPlan,
    Stage6SumcheckClaimPlan,
    Stage6SumcheckBatchPlan,
    Stage6SumcheckDriverPlan,
    Stage6SumcheckInstanceResultPlan,
    Stage6SumcheckEvalPlan,
    Stage6PointSlicePlan,
    Stage6PointConcatPlan,
    Stage6OpeningClaimPlan,
    Stage6OpeningClaimEqualityPlan,
    Stage6OpeningBatchPlan,
    point_zero = Stage6PointZeroPlan
);
