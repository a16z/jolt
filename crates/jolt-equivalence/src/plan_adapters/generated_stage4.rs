use bolt::protocols::jolt::Stage4CpuProgram as CompilerStage4CpuProgram;
use jolt_verifier::stages::stage4 as generated_stage4;

define_stage_adapter!(
    generated,
    leak_generated_stage4_verifier_program,
    CompilerStage4CpuProgram,
    generated_stage4,
    Stage4VerifierProgramPlan,
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
    Stage4PointExprPlan,
    Stage4PointExprPlan,
    Stage4OpeningClaimPlan,
    Stage4OpeningClaimEqualityPlan,
    Stage4OpeningBatchPlan,
    scalar_expr = Stage4ScalarExprPlan,
    empty_indexed_eval_families = empty_indexed_eval_families,
    relation_outputs = Stage4RelationOutputPlan
);
