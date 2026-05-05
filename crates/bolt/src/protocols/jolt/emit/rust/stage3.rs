mod constants;
mod parse;
mod source;
mod verify;

use super::stage_common::{
    stage23_verifier_type_aliases, stage_default_transcript_alias,
    stage_fallible_role_module_source, stage_prover_imports, stage_role_filename,
    stage_verifier_error_enum, Stage23VerifierTypeShape, StageProverImportShape,
    StageVerifierErrorShape,
};
use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3CpuProgram {
    pub role: Role,
    pub params: Stage3Params,
    pub steps: Vec<Stage3ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage3TranscriptSqueezePlan>,
    pub opening_inputs: Vec<Stage3OpeningInputPlan>,
    pub field_constants: Vec<Stage3FieldConstantPlan>,
    pub field_exprs: Vec<Stage3FieldExprPlan>,
    pub kernels: Vec<Stage3KernelPlan>,
    pub claims: Vec<Stage3SumcheckClaimPlan>,
    pub batches: Vec<Stage3SumcheckBatchPlan>,
    pub drivers: Vec<Stage3SumcheckDriverPlan>,
    pub instance_results: Vec<Stage3SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage3SumcheckEvalPlan>,
    pub point_slices: Vec<Stage3PointSlicePlan>,
    pub point_concats: Vec<Stage3PointConcatPlan>,
    pub opening_claims: Vec<Stage3OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage3OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage3OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckClaimPlan {
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
pub struct Stage3SumcheckBatchPlan {
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
pub struct Stage3SumcheckDriverPlan {
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
pub struct Stage3SumcheckInstanceResultPlan {
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
pub struct Stage3SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage3_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage3CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage3CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage3_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage3_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage3CpuProgram {
    fn emit_source(&self) -> Result<String, EmitError> {
        stage_fallible_role_module_source(
            &self.role,
            || self.emit_prover_constants(),
            || self.emit_verifier_constants(),
            || {
                (
                    Self::emit_prover_imports(),
                    Self::emit_prover_types(),
                    Self::emit_prover_entrypoint(),
                )
            },
            || {
                (
                    Self::emit_verifier_imports(),
                    Self::emit_verifier_types(),
                    Self::emit_verifier_entrypoint(),
                )
            },
        )
    }

    fn filename(&self) -> &'static str {
        stage_role_filename(&self.role, "prove_stage3.rs", "verify_stage3.rs")
    }

    fn emit_prover_imports() -> String {
        stage_prover_imports(3, StageProverImportShape::STAGE3)
    }

    fn emit_prover_types() -> String {
        stage_default_transcript_alias(3)
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::{batch_claims, eval_by_name, find_batch, find_plan, reverse_slice};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_poly::{EqPlusOnePolynomial, EqPolynomial};\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_verifier_types() -> String {
        let mut source = stage23_verifier_type_aliases(3, Stage23VerifierTypeShape::STAGE3);
        source.push_str(&stage_verifier_error_enum(
            3,
            StageVerifierErrorShape::STANDARD,
        ));
        source
    }
}
