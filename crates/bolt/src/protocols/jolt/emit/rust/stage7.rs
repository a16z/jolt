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
pub struct Stage7CpuProgram {
    pub role: Role,
    pub params: Stage7Params,
    pub steps: Vec<Stage7ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage7TranscriptSqueezePlan>,
    pub transcript_absorb_bytes: Vec<Stage7TranscriptAbsorbBytesPlan>,
    pub opening_inputs: Vec<Stage7OpeningInputPlan>,
    pub field_constants: Vec<Stage7FieldConstantPlan>,
    pub field_exprs: Vec<Stage7FieldExprPlan>,
    pub kernels: Vec<Stage7KernelPlan>,
    pub claims: Vec<Stage7SumcheckClaimPlan>,
    pub batches: Vec<Stage7SumcheckBatchPlan>,
    pub drivers: Vec<Stage7SumcheckDriverPlan>,
    pub instance_results: Vec<Stage7SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage7SumcheckEvalPlan>,
    pub point_zeros: Vec<Stage7PointZeroPlan>,
    pub point_slices: Vec<Stage7PointSlicePlan>,
    pub point_concats: Vec<Stage7PointConcatPlan>,
    pub opening_claims: Vec<Stage7OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage7OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage7OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7TranscriptAbsorbBytesPlan {
    pub symbol: String,
    pub label: String,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7SumcheckClaimPlan {
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
pub struct Stage7SumcheckBatchPlan {
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
pub struct Stage7SumcheckDriverPlan {
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
pub struct Stage7SumcheckInstanceResultPlan {
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
pub struct Stage7SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7PointZeroPlan {
    pub symbol: String,
    pub field: String,
    pub arity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage7_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage7CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage7CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage7_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage7_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

impl Stage7CpuProgram {
    fn filename(&self) -> &'static str {
        stage_role_filename(&self.role, "prove_stage7.rs", "verify_stage7.rs")
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
        stage_prover_imports(7, StageProverImportShape::STAGE6_OR_7)
    }

    fn emit_prover_types() -> String {
        stage_default_transcript_alias(7)
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::{batch_claims, eval_by_name, find_batch, find_plan, normalize_bytecode_read_raf_point, normalize_instruction_read_raf_point, reverse_slice};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_poly::EqPolynomial;\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    fn emit_verifier_types() -> String {
        let mut source = stage_verifier_type_aliases(7, StageRuntimeVerifierTypeShape::STAGE6_OR_7);
        source.push_str(&stage_runtime_verifier_program_aliases(7));
        source.push_str(&stage_verifier_error_enum(
            7,
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
            Role::Prover => "Stage7CpuProgramPlan",
            Role::Verifier => "Stage7VerifierProgramPlan",
        }
    }
}
