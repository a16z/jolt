use bolt::protocols::jolt::Stage1CpuProgram as CompilerStage1CpuProgram;
use jolt_kernels::stage1 as kernel_stage1;

define_stage1_adapter!(
    kernel,
    leak_stage1_program,
    CompilerStage1CpuProgram,
    kernel_stage1,
    Stage1CpuProgramPlan,
    Stage1Params,
    Stage1TranscriptSqueezePlan,
    Stage1SumcheckClaimPlan,
    Stage1SumcheckBatchPlan,
    Stage1SumcheckDriverPlan,
    Stage1SumcheckInstanceResultPlan,
    Stage1SumcheckEvalPlan,
    Stage1OpeningClaimPlan,
    Stage1OpeningBatchPlan,
    kernels = Stage1KernelPlan
);
