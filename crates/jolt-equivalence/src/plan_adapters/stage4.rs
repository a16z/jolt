use bolt::protocols::jolt::Stage4CpuProgram as CompilerStage4CpuProgram;
use jolt_kernels::stage4 as kernel_stage4;

define_stage_adapter!(
    kernel,
    leak_stage4_program,
    CompilerStage4CpuProgram,
    kernel_stage4,
    Stage4CpuProgramPlan,
    Stage4Params,
    Stage4ProgramStepPlan,
    Stage4TranscriptSqueezePlan,
    Stage4TranscriptAbsorbBytesPlan,
    Stage4OpeningInputPlan,
    Stage4FieldConstantPlan,
    Stage4FieldExprPlan,
    Stage4KernelPlan,
    Stage4SumcheckClaimPlan,
    Stage4SumcheckBatchPlan,
    Stage4SumcheckDriverPlan,
    Stage4SumcheckInstanceResultPlan,
    Stage4SumcheckEvalPlan,
    Stage4PointSlicePlan,
    Stage4PointConcatPlan,
    Stage4OpeningClaimPlan,
    Stage4OpeningClaimEqualityPlan,
    Stage4OpeningBatchPlan
);
