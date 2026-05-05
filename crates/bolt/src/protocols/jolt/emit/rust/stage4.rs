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
pub struct Stage4CpuProgram {
    pub role: Role,
    pub params: Stage4Params,
    pub steps: Vec<Stage4ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage4TranscriptSqueezePlan>,
    pub transcript_absorb_bytes: Vec<Stage4TranscriptAbsorbBytesPlan>,
    pub opening_inputs: Vec<Stage4OpeningInputPlan>,
    pub field_constants: Vec<Stage4FieldConstantPlan>,
    pub field_exprs: Vec<Stage4FieldExprPlan>,
    pub kernels: Vec<Stage4KernelPlan>,
    pub claims: Vec<Stage4SumcheckClaimPlan>,
    pub batches: Vec<Stage4SumcheckBatchPlan>,
    pub drivers: Vec<Stage4SumcheckDriverPlan>,
    pub instance_results: Vec<Stage4SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage4SumcheckEvalPlan>,
    pub point_slices: Vec<Stage4PointSlicePlan>,
    pub point_concats: Vec<Stage4PointConcatPlan>,
    pub opening_claims: Vec<Stage4OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage4OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage4OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4TranscriptAbsorbBytesPlan {
    pub symbol: String,
    pub label: String,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckClaimPlan {
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
pub struct Stage4SumcheckBatchPlan {
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
pub struct Stage4SumcheckDriverPlan {
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
pub struct Stage4SumcheckInstanceResultPlan {
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
pub struct Stage4SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage4_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage4CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage4CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage4_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage4_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

impl Stage4CpuProgram {
    fn filename(&self) -> &'static str {
        stage_role_filename(&self.role, "prove_stage4.rs", "verify_stage4.rs")
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
        stage_prover_imports(4, StageProverImportShape::STAGE4_OR_5)
    }

    fn emit_prover_types() -> String {
        stage_default_transcript_alias(4)
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::{batch_claims, eval_by_name, find_batch, find_plan, lt_polynomial_eval, reverse_slice};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_poly::EqPolynomial;\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    fn emit_verifier_types() -> String {
        let mut source = stage_verifier_type_aliases(4, StageRuntimeVerifierTypeShape::STAGE4_OR_5);
        source.push_str(&stage_runtime_verifier_program_aliases(4));
        source.push_str(&stage_verifier_error_enum(
            4,
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
            Role::Prover => "Stage4CpuProgramPlan",
            Role::Verifier => "Stage4VerifierProgramPlan",
        }
    }
}
