use bolt::protocols::jolt::Stage2CpuProgram as CompilerStage2CpuProgram;
use jolt_verifier::stages::stage2 as generated_stage2;

define_stage_adapter_no_absorb!(
    generated,
    leak_generated_stage2_verifier_program,
    CompilerStage2CpuProgram,
    generated_stage2,
    Stage2VerifierProgramPlan,
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
    Stage2PointExprPlan,
    Stage2PointExprPlan,
    Stage2OpeningClaimPlan,
    Stage2OpeningBatchPlan,
    empty_scalar_exprs = Stage2ScalarExprPlan
);
