use bolt::protocols::jolt::Stage2CpuProgram as CompilerStage2CpuProgram;
use jolt_kernels::stage2 as kernel_stage2;

define_stage_adapter_no_absorb!(
    kernel,
    leak_stage2_program,
    CompilerStage2CpuProgram,
    kernel_stage2,
    Stage2CpuProgramPlan,
    Stage2Params,
    Stage2ProgramStepPlan,
    Stage2TranscriptSqueezePlan,
    Stage2OpeningInputPlan,
    Stage2FieldConstantPlan,
    Stage2FieldExprPlan,
    Stage2SumcheckClaimPlan,
    Stage2SumcheckBatchPlan,
    Stage2SumcheckDriverPlan,
    Stage2SumcheckInstanceResultPlan,
    Stage2SumcheckEvalPlan,
    Stage2PointSlicePlan,
    Stage2PointConcatPlan,
    Stage2OpeningClaimPlan,
    Stage2OpeningBatchPlan,
    kernels = Stage2KernelPlan
);
