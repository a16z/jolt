use bolt::protocols::jolt::Stage3CpuProgram as CompilerStage3CpuProgram;
use jolt_kernels::stage3 as kernel_stage3;

define_stage_adapter_no_absorb!(
    kernel,
    leak_stage3_program,
    CompilerStage3CpuProgram,
    kernel_stage3,
    Stage3CpuProgramPlan,
    Stage3Params,
    Stage3ProgramStepPlan,
    Stage3TranscriptSqueezePlan,
    Stage3OpeningInputPlan,
    Stage3FieldConstantPlan,
    Stage3FieldExprPlan,
    Stage3SumcheckClaimPlan,
    Stage3SumcheckBatchPlan,
    Stage3SumcheckDriverPlan,
    Stage3SumcheckInstanceResultPlan,
    Stage3SumcheckEvalPlan,
    Stage3PointSlicePlan,
    Stage3PointConcatPlan,
    Stage3OpeningClaimPlan,
    Stage3OpeningBatchPlan,
    kernels = Stage3KernelPlan,
    opening_equalities = Stage3OpeningClaimEqualityPlan
);
