mod constants;
mod parse;
mod source;
mod verify;

use super::stage_common::{stage_fallible_role_module_source, stage_role_filename};
use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1CpuProgram {
    pub role: Role,
    pub params: Stage1Params,
    pub transcript_squeezes: Vec<Stage1TranscriptSqueezePlan>,
    pub kernels: Vec<Stage1KernelPlan>,
    pub claims: Vec<Stage1SumcheckClaimPlan>,
    pub batches: Vec<Stage1SumcheckBatchPlan>,
    pub drivers: Vec<Stage1SumcheckDriverPlan>,
    pub instance_results: Vec<Stage1SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage1SumcheckEvalPlan>,
    pub opening_claims: Vec<Stage1OpeningClaimPlan>,
    pub opening_batches: Vec<Stage1OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckClaimPlan {
    pub symbol: String,
    pub stage: String,
    pub domain: String,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: String,
    pub kernel: Option<String>,
    pub relation: Option<String>,
    pub claim_value: String,
    pub input_openings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
    pub claim_label: String,
    pub round_label: String,
    pub round_schedule: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckDriverPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub kernel: Option<String>,
    pub relation: Option<String>,
    pub batch: String,
    pub policy: String,
    pub round_schedule: Vec<usize>,
    pub claim_label: String,
    pub round_label: String,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckInstanceResultPlan {
    pub symbol: String,
    pub source: String,
    pub claim: String,
    pub relation: String,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: String,
    pub degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage1_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage1CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage1CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage1_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage1_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage1CpuProgram {
    fn emit_source(&self) -> Result<String, EmitError> {
        stage_fallible_role_module_source(
            &self.role,
            || self.emit_prover_constants(),
            || self.emit_verifier_constants(),
            || {
                (
                    Self::emit_prover_imports().to_owned(),
                    Self::emit_prover_types().to_owned(),
                    Self::emit_prover_entrypoint(),
                )
            },
            || {
                (
                    Self::emit_verifier_imports(),
                    Self::emit_verifier_types().to_owned(),
                    Self::emit_verifier_entrypoint(),
                )
            },
        )
    }

    fn filename(&self) -> &'static str {
        stage_role_filename(
            &self.role,
            "prove_stage1_outer.rs",
            "verify_stage1_outer.rs",
        )
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage1::{execute_stage1_program, Stage1CpuProgramPlan, Stage1ExecutionArtifacts, Stage1ExecutionMode, Stage1KernelError, Stage1KernelExecutor, Stage1KernelPlan, Stage1OpeningBatchPlan, Stage1OpeningClaimPlan, Stage1Params, Stage1SumcheckBatchPlan, Stage1SumcheckClaimPlan, Stage1SumcheckDriverPlan, Stage1SumcheckEvalPlan, Stage1SumcheckInstanceResultPlan, Stage1TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage1Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::append_labeled_scalar;\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::{CompressedLabeledRoundPoly, LabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckVerifier};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultStage1Transcript = Blake2bTranscript<Fr>;

pub type Stage1Params = super::common::StageParams;
pub type Stage1NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage1SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage1ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage1ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage1Proof<F> = super::common::StageProof<F>;
pub type Stage1VerifierProgramPlan = super::common::VerifierProgramPlanMinimal;

pub use super::common::{
    OpeningBatchPlan as Stage1OpeningBatchPlan, OpeningClaimPlan as Stage1OpeningClaimPlan,
    SumcheckBatchPlan as Stage1SumcheckBatchPlan, SumcheckEvalPlan as Stage1SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage1SumcheckInstanceResultPlan,
    TranscriptSqueezePlan as Stage1TranscriptSqueezePlan,
    SumcheckClaimPlan as Stage1SumcheckClaimPlan,
    SumcheckDriverPlan as Stage1SumcheckDriverPlan,
};

#[derive(Debug)]
pub enum VerifyStage1Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { driver: &'static str, claim: &'static str },
    MissingDependency { driver: &'static str, dependency: &'static str },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}
"
    }
}
