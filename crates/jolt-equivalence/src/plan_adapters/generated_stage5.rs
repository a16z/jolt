use bolt::protocols::jolt::Stage5CpuProgram as CompilerStage5CpuProgram;
use jolt_verifier::stages::stage5 as generated_stage5;

define_stage_adapter!(
    generated,
    leak_generated_stage5_base_verifier_program,
    CompilerStage5CpuProgram,
    generated_stage5,
    Stage5CpuProgramPlan,
    Stage5Params,
    Stage5ProgramStepPlan,
    Stage5TranscriptSqueezePlan,
    Stage5TranscriptAbsorbBytesPlan,
    Stage5OpeningInputPlan,
    Stage5FieldConstantPlan,
    Stage5FieldExprPlan,
    Stage5KernelPlan,
    Stage5SumcheckClaimPlan,
    Stage5SumcheckBatchPlan,
    Stage5SumcheckDriverPlan,
    Stage5SumcheckInstanceResultPlan,
    Stage5SumcheckEvalPlan,
    Stage5PointExprPlan,
    Stage5PointExprPlan,
    Stage5OpeningClaimPlan,
    Stage5OpeningClaimEqualityPlan,
    Stage5OpeningBatchPlan,
    scalar_expr = Stage5ScalarExprPlan,
    indexed_eval_families = indexed_eval_family_rows,
    relation_outputs = Stage5RelationOutputPlan
);

pub fn leak_generated_stage5_verifier_program(
    program: &CompilerStage5CpuProgram,
) -> &'static generated_stage5::Stage5VerifierProgramPlan {
    Box::leak(Box::new(generated_stage5::Stage5VerifierProgramPlan {
        base: *leak_generated_stage5_base_verifier_program(program),
        instruction_read_raf_plan: generated_stage5::STAGE5_INSTRUCTION_READ_RAF_PLAN,
    }))
}
