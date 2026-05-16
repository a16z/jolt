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
    Stage7PointSlicePlan,
    Stage7PointConcatPlan,
    Stage7OpeningClaimPlan,
    Stage7OpeningClaimEqualityPlan,
    Stage7OpeningBatchPlan,
    point_zero = Stage7PointZeroPlan,
    value_expr = Stage7ValueExprPlan,
    relation_outputs = Stage7RelationOutputPlan,
    relation_output_values = Stage7StructuredPolynomialEvalPlan
);
