mod commitment;
mod stage1;
mod stage2;
mod stage3;

pub(crate) fn push_format(source: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write as _;

    if source.write_fmt(args).is_err() {
        std::process::abort();
    }
}

pub use commitment::{
    commitment_cpu_program, emit_commitment_rust, CommitmentBatchPlan, CommitmentCpuProgram,
    CommitmentParams, EmitError, OptionalCommitmentPlan, OptionalSkipPolicy, OracleGeneration,
    OraclePlan, RustSourceFile, TranscriptStep,
};
pub use stage1::{
    emit_stage1_rust, stage1_cpu_program, Stage1CpuProgram, Stage1KernelPlan,
    Stage1OpeningBatchPlan, Stage1OpeningClaimPlan, Stage1Params, Stage1SumcheckBatchPlan,
    Stage1SumcheckClaimPlan, Stage1SumcheckDriverPlan, Stage1SumcheckEvalPlan,
};
pub use stage2::{emit_stage2_rust, stage2_cpu_program, Stage2CpuProgram};
pub use stage3::{emit_stage3_rust, stage3_cpu_program, Stage3CpuProgram};
