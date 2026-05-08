use bolt::protocols::jolt::Stage1CpuProgram as CompilerStage1CpuProgram;
use jolt_verifier::stages::stage1_outer as generated_stage1;

define_stage1_adapter!(
    generated,
    leak_generated_stage1_verifier_program,
    CompilerStage1CpuProgram,
    generated_stage1,
    Stage1VerifierProgramPlan,
    Stage1Params,
    Stage1TranscriptSqueezePlan,
    Stage1SumcheckClaimPlan,
    Stage1SumcheckBatchPlan,
    Stage1SumcheckDriverPlan,
    Stage1SumcheckInstanceResultPlan,
    Stage1SumcheckEvalPlan,
    Stage1OpeningClaimPlan,
    Stage1OpeningBatchPlan
);
