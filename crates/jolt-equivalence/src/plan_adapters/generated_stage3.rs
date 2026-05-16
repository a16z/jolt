use bolt::protocols::jolt::Stage3CpuProgram as CompilerStage3CpuProgram;
use jolt_verifier::stages::stage3 as generated_stage3;

define_stage_adapter_no_absorb!(
    generated,
    leak_generated_stage3_verifier_program,
    CompilerStage3CpuProgram,
    generated_stage3,
    Stage3VerifierProgramPlan,
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
    relation_outputs = Stage3RelationOutputPlan,
    relation_output_values = Stage3StructuredPolynomialEvalPlan,
    opening_equalities = Stage3OpeningClaimEqualityPlan
);
