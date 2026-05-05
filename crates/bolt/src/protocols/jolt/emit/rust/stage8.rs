mod constants;
mod parse;
mod source;
mod verify;

use super::stage_common::stage_role_filename;
use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

const EVALUATION_POINT_SOURCE_SYMBOL: &str = "stage8.evaluation.point_source";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8CpuProgram {
    pub role: Role,
    pub params: Stage8Params,
    pub function: String,
    pub opening_inputs: Vec<Stage8OpeningInputPlan>,
    pub opening_claims: Vec<Stage8OpeningClaimPlan>,
    pub opening_batches: Vec<Stage8OpeningBatchPlan>,
    pub pcs_proofs: Vec<Stage8PcsProofPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub family: String,
    pub domain: String,
    pub point_arity: usize,
    pub point_source: String,
    pub eval_source: String,
    pub source_stage: String,
    pub source_claim: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8OpeningBatchPlan {
    pub symbol: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8PcsProofPlan {
    pub symbol: String,
    pub mode: String,
    pub pcs: String,
    pub proof_slot: String,
    pub transcript_label: String,
    pub batch: String,
}

pub fn stage8_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage8CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage8CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage8_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage8_cpu_program(module)?;
    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage8CpuProgram {
    fn filename(&self) -> &'static str {
        stage_role_filename(&self.role, "prove_stage8.rs", "verify_stage8.rs")
    }
}
