#![expect(
    clippy::needless_raw_string_hashes,
    reason = "generated Rust templates are kept as raw string blocks for copyable output"
)]

mod constants;
mod parse;
mod source;
mod verify;

use super::stage_common::{
    stage_default_transcript_alias, stage_prover_imports, stage_role_filename,
    stage_role_module_source, stage_runtime_verifier_program_aliases, stage_verifier_error_enum,
    stage_verifier_type_aliases, StageProverImportShape, StageRuntimeVerifierTypeShape,
    StageVerifierErrorShape,
};
use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5CpuProgram {
    pub role: Role,
    pub params: Stage5Params,
    pub steps: Vec<Stage5ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage5TranscriptSqueezePlan>,
    pub transcript_absorb_bytes: Vec<Stage5TranscriptAbsorbBytesPlan>,
    pub opening_inputs: Vec<Stage5OpeningInputPlan>,
    pub field_constants: Vec<Stage5FieldConstantPlan>,
    pub field_exprs: Vec<Stage5FieldExprPlan>,
    pub kernels: Vec<Stage5KernelPlan>,
    pub claims: Vec<Stage5SumcheckClaimPlan>,
    pub batches: Vec<Stage5SumcheckBatchPlan>,
    pub drivers: Vec<Stage5SumcheckDriverPlan>,
    pub instance_results: Vec<Stage5SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage5SumcheckEvalPlan>,
    pub point_slices: Vec<Stage5PointSlicePlan>,
    pub point_concats: Vec<Stage5PointConcatPlan>,
    pub opening_claims: Vec<Stage5OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage5OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage5OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5TranscriptAbsorbBytesPlan {
    pub symbol: String,
    pub label: String,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckClaimPlan {
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
pub struct Stage5SumcheckBatchPlan {
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
pub struct Stage5SumcheckDriverPlan {
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
pub struct Stage5SumcheckInstanceResultPlan {
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
pub struct Stage5SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage5_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage5CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage5CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage5_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage5_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

impl Stage5CpuProgram {
    fn filename(&self) -> &'static str {
        stage_role_filename(&self.role, "prove_stage5.rs", "verify_stage5.rs")
    }

    fn emit_source(&self) -> String {
        let constants = self.emit_constants();
        stage_role_module_source(
            &self.role,
            &constants,
            self.emit_entrypoint(),
            || (Self::emit_prover_imports(), Self::emit_prover_types()),
            || (Self::emit_verifier_imports(), Self::emit_verifier_types()),
        )
    }

    fn emit_prover_imports() -> String {
        stage_prover_imports(5, StageProverImportShape::STAGE4_OR_5)
    }

    fn emit_prover_types() -> String {
        stage_default_transcript_alias(5)
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::{batch_claims, eval_by_name, find_batch, find_plan, identity_polynomial_eval, indexed_evals_by_prefix, indexed_evals_by_prefix_any, lt_polynomial_eval, normalize_instruction_read_raf_point, operand_polynomial_eval, reverse_slice, suffix_point};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_lookup_tables::LookupTableKind;\n\
         use jolt_poly::EqPolynomial;\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    fn emit_verifier_types() -> String {
        let mut source = stage_verifier_type_aliases(5, StageRuntimeVerifierTypeShape::STAGE4_OR_5);
        source.push_str(&stage_runtime_verifier_program_aliases(5));
        source.push_str(&stage_verifier_error_enum(
            5,
            StageVerifierErrorShape::STANDARD,
        ));
        source
    }

    fn role_label(&self) -> &'static str {
        match self.role {
            Role::Prover => "prover",
            Role::Verifier => "verifier",
        }
    }

    fn program_plan_type(&self) -> &'static str {
        match self.role {
            Role::Prover => "Stage5CpuProgramPlan",
            Role::Verifier => "Stage5VerifierProgramPlan",
        }
    }
}
