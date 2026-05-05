use bolt::protocols::jolt::Stage7CpuProgram as CompilerStage7CpuProgram;
use jolt_kernels::stage7 as kernel_stage7;

define_stage_adapter!(
    kernel,
    leak_stage7_program,
    CompilerStage7CpuProgram,
    kernel_stage7,
    Stage7CpuProgramPlan,
    Stage7Params,
    Stage7ProgramStepPlan,
    Stage7TranscriptSqueezePlan,
    Stage7TranscriptAbsorbBytesPlan,
    Stage7OpeningInputPlan,
    Stage7FieldConstantPlan,
    Stage7FieldExprPlan,
    Stage7KernelPlan,
    Stage7SumcheckClaimPlan,
    Stage7SumcheckBatchPlan,
    Stage7SumcheckDriverPlan,
    Stage7SumcheckInstanceResultPlan,
    Stage7SumcheckEvalPlan,
    Stage7PointSlicePlan,
    Stage7PointConcatPlan,
    Stage7OpeningClaimPlan,
    Stage7OpeningClaimEqualityPlan,
    Stage7OpeningBatchPlan,
    point_zero = Stage7PointZeroPlan
);
