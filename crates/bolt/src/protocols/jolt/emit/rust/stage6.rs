mod constants;
mod parse;
mod source;
mod types;
mod verify;

use super::stage_common::{
    stage_default_transcript_alias, stage_prover_imports, stage_role_filename,
    stage_role_module_source, StageProverImportShape,
};
use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6CpuProgram {
    pub role: Role,
    pub params: Stage6Params,
    pub steps: Vec<Stage6ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage6TranscriptSqueezePlan>,
    pub transcript_absorb_bytes: Vec<Stage6TranscriptAbsorbBytesPlan>,
    pub opening_inputs: Vec<Stage6OpeningInputPlan>,
    pub field_constants: Vec<Stage6FieldConstantPlan>,
    pub field_exprs: Vec<Stage6FieldExprPlan>,
    pub kernels: Vec<Stage6KernelPlan>,
    pub claims: Vec<Stage6SumcheckClaimPlan>,
    pub batches: Vec<Stage6SumcheckBatchPlan>,
    pub drivers: Vec<Stage6SumcheckDriverPlan>,
    pub instance_results: Vec<Stage6SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage6SumcheckEvalPlan>,
    pub point_zeros: Vec<Stage6PointZeroPlan>,
    pub point_slices: Vec<Stage6PointSlicePlan>,
    pub point_concats: Vec<Stage6PointConcatPlan>,
    pub opening_claims: Vec<Stage6OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage6OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage6OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptAbsorbBytesPlan {
    pub symbol: String,
    pub label: String,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckClaimPlan {
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
pub struct Stage6SumcheckBatchPlan {
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
pub struct Stage6SumcheckDriverPlan {
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
pub struct Stage6SumcheckInstanceResultPlan {
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
pub struct Stage6SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PointZeroPlan {
    pub symbol: String,
    pub field: String,
    pub arity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage6_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage6CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage6CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage6_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage6_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

impl Stage6CpuProgram {
    fn filename(&self) -> &'static str {
        stage_role_filename(&self.role, "prove_stage6.rs", "verify_stage6.rs")
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
        stage_prover_imports(6, StageProverImportShape::STAGE6_OR_7)
    }

    fn emit_prover_types() -> String {
        stage_default_transcript_alias(6)
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::{batch_claims, expected_stage67_booleanity, expected_stage67_bytecode_read_raf, expected_stage67_hamming_booleanity, expected_stage67_inc_claim_reduction, expected_stage67_instruction_ra_virtual, expected_stage67_ram_ra_virtual, find_batch, find_plan, normalize_bytecode_read_raf_point, normalize_instruction_read_raf_point, stage67_trace_rounds, Stage67BytecodeEntry, Stage67BytecodeSymbols, Stage67RelationSymbols};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    fn role_label(&self) -> &'static str {
        match self.role {
            Role::Prover => "prover",
            Role::Verifier => "verifier",
        }
    }

    fn program_plan_type(&self) -> &'static str {
        match self.role {
            Role::Prover => "Stage6CpuProgramPlan",
            Role::Verifier => "Stage6VerifierProgramPlan",
        }
    }
}
