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
pub struct Stage2CpuProgram {
    pub role: Role,
    pub params: Stage2Params,
    pub steps: Vec<Stage2ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage2TranscriptSqueezePlan>,
    pub opening_inputs: Vec<Stage2OpeningInputPlan>,
    pub field_constants: Vec<Stage2FieldConstantPlan>,
    pub field_exprs: Vec<Stage2FieldExprPlan>,
    pub kernels: Vec<Stage2KernelPlan>,
    pub claims: Vec<Stage2SumcheckClaimPlan>,
    pub batches: Vec<Stage2SumcheckBatchPlan>,
    pub drivers: Vec<Stage2SumcheckDriverPlan>,
    pub instance_results: Vec<Stage2SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage2SumcheckEvalPlan>,
    pub point_slices: Vec<Stage2PointSlicePlan>,
    pub point_concats: Vec<Stage2PointConcatPlan>,
    pub opening_claims: Vec<Stage2OpeningClaimPlan>,
    pub opening_batches: Vec<Stage2OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2SumcheckClaimPlan {
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
pub struct Stage2SumcheckBatchPlan {
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
pub struct Stage2SumcheckDriverPlan {
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
pub struct Stage2SumcheckInstanceResultPlan {
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
pub struct Stage2SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage2_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage2CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage2CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage2_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage2_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage2CpuProgram {
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
        stage_role_filename(&self.role, "prove_stage2.rs", "verify_stage2.rs")
    }

    fn emit_prover_imports() -> String {
        stage_prover_imports(2, StageProverImportShape::STAGE2)
    }

    fn emit_prover_types() -> String {
        stage_default_transcript_alias(2)
    }

    fn emit_verifier_imports() -> &'static str {
        "use super::common::{append_labeled_scalar, batch_claims, eval_by_name, find_batch, find_plan, pow_field, require_operand_count, reverse_slice, single_operand};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_poly::lagrange::{lagrange_evals, lagrange_kernel_eval};\n\
         use jolt_poly::{EqPolynomial, UnivariatePoly};\n\
         use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckVerifier};\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    fn emit_verifier_types() -> String {
        let mut source = stage23_verifier_type_aliases(2, Stage23VerifierTypeShape::STAGE2);
        source.push_str(
            r#"
#[derive(Clone, Copy, Debug)]
pub struct Stage2RamAccess {
    pub remapped_address: Option<usize>,
    pub read_value: u64,
    pub write_value: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamOutputLayout {
    pub io_start: usize,
    pub io_end: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamData<'a> {
    pub log_k: usize,
    pub start_address: u64,
    pub initial_ram: &'a [u64],
    pub final_ram: &'a [u64],
    pub accesses: &'a [Stage2RamAccess],
    pub output_layout: Option<Stage2RamOutputLayout>,
}

#[derive(Clone, Debug, Default)]
struct Stage2ValueStore<F: Field>(super::common::ValueStore<F>);
"#,
        );
        source.push_str(&stage_verifier_error_enum(2, StageVerifierErrorShape::RAM));
        source
    }
}
