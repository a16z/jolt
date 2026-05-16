use bolt::protocols::jolt::Stage6CpuProgram as CompilerStage6CpuProgram;
use jolt_verifier::stages::stage6 as generated_stage6;

define_stage_adapter!(
    generated,
    leak_generated_stage6_verifier_program,
    CompilerStage6CpuProgram,
    generated_stage6,
    Stage6VerifierProgramPlan,
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
    Stage6PointExprPlan,
    Stage6PointExprPlan,
    Stage6OpeningClaimPlan,
    Stage6OpeningClaimEqualityPlan,
    Stage6OpeningBatchPlan,
    point_zero = Stage6PointExprPlan,
    scalar_expr = Stage6ScalarExprPlan,
    relation_outputs = Stage6RelationOutputPlan
);
