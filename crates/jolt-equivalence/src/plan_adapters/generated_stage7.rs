use bolt::protocols::jolt::Stage7CpuProgram as CompilerStage7CpuProgram;
use jolt_verifier::stages::stage7 as generated_stage7;

define_stage_adapter!(
    generated,
    leak_generated_stage7_verifier_program,
    CompilerStage7CpuProgram,
    generated_stage7,
    Stage7VerifierProgramPlan,
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
    Stage7PointExprPlan,
    Stage7PointExprPlan,
    Stage7OpeningClaimPlan,
    Stage7OpeningClaimEqualityPlan,
    Stage7OpeningBatchPlan,
    point_zero = Stage7PointExprPlan,
    scalar_expr = Stage7ScalarExprPlan,
    empty_indexed_eval_families = empty_indexed_eval_families,
    relation_outputs = Stage7RelationOutputPlan
);
