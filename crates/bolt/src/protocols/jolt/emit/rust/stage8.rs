use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::OperationRef;

use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
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
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let role = module
            .role()
            .ok_or_else(|| EmitError::new("stage8 CPU module missing role"))?;
        let mut params = None;
        let mut function = None;
        let mut opening_inputs = Vec::new();
        let mut opening_claims = Vec::new();
        let mut opening_batches = Vec::new();
        let mut pcs_proofs = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(Stage8Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.function" => {
                    function = Some(string_attr(op, "sym_name")?);
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage8OpeningInputPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source_stage: symbol_attr(op, "source_stage")?,
                        source_claim: symbol_attr(op, "source_claim")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                    });
                }
                "cpu.pcs_opening_claim" => {
                    opening_claims.push(Stage8OpeningClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        oracle: symbol_attr(op, "oracle")?,
                        family: symbol_attr(op, "family")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        point_source: operand_symbol(op, 0)?,
                        eval_source: operand_symbol(op, 1)?,
                        source_stage: String::new(),
                        source_claim: String::new(),
                    });
                }
                "cpu.pcs_opening_batch" => {
                    opening_batches.push(Stage8OpeningBatchPlan {
                        symbol: string_attr(op, "sym_name")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        policy: string_attr(op, "policy")?,
                        count: int_attr(op, "count")?,
                        ordered_claims: symbol_array_attr(op, "ordered_claims")?,
                        claim_operands: operand_symbols(op, 0)?,
                    });
                }
                "cpu.pcs_batch_open" | "cpu.pcs_batch_verify" => {
                    let mode = match operation_name(op).as_str() {
                        "cpu.pcs_batch_open" => "open",
                        "cpu.pcs_batch_verify" => "verify",
                        _ => unreachable!(),
                    };
                    pcs_proofs.push(Stage8PcsProofPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: mode.to_owned(),
                        pcs: symbol_attr(op, "pcs")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        transcript_label: string_attr(op, "transcript_label")?,
                        batch: operand_symbol(op, 1)?,
                    });
                }
                _ => {}
            }
        }

        let input_by_symbol = opening_inputs
            .iter()
            .map(|input| (input.symbol.as_str(), input))
            .collect::<BTreeMap<_, _>>();
        for claim in &mut opening_claims {
            let input = input_by_symbol
                .get(claim.point_source.as_str())
                .ok_or_else(|| {
                    EmitError::new(format!(
                        "stage8 opening claim `{}` references missing point source `{}`",
                        claim.symbol, claim.point_source
                    ))
                })?;
            claim.source_stage = input.source_stage.clone();
            claim.source_claim = input.source_claim.clone();
        }

        Ok(Self {
            role,
            params: params.ok_or_else(|| EmitError::new("stage8 program missing cpu.params"))?,
            function: function
                .ok_or_else(|| EmitError::new("stage8 program missing cpu.function"))?,
            opening_inputs,
            opening_claims,
            opening_batches,
            pcs_proofs,
        })
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        if self.function != "jolt.stage8" {
            return Err(EmitError::new(format!(
                "stage8 emitter expected function `jolt.stage8`, got `{}`",
                self.function
            )));
        }
        if self.opening_batches.len() != 1 {
            return Err(EmitError::new(format!(
                "stage8 emitter expects one PCS opening batch, got {}",
                self.opening_batches.len()
            )));
        }
        if self.pcs_proofs.len() != 1 {
            return Err(EmitError::new(format!(
                "stage8 emitter expects one PCS proof op, got {}",
                self.pcs_proofs.len()
            )));
        }
        let expected_mode = match self.role {
            Role::Prover => "open",
            Role::Verifier => "verify",
        };
        if self.pcs_proofs[0].mode != expected_mode {
            return Err(EmitError::new(format!(
                "stage8 {} artifact expected PCS mode `{expected_mode}`, got `{}`",
                self.role, self.pcs_proofs[0].mode
            )));
        }
        let batch = &self.opening_batches[0];
        if batch.count != self.opening_claims.len() {
            return Err(EmitError::new(format!(
                "stage8 opening batch count {} does not match {} opening claims",
                batch.count,
                self.opening_claims.len()
            )));
        }
        if batch.ordered_claims != batch.claim_operands {
            return Err(EmitError::new(
                "stage8 opening batch ordered claims do not match SSA operands",
            ));
        }
        if !self
            .opening_inputs
            .iter()
            .any(|input| input.symbol == EVALUATION_POINT_SOURCE_SYMBOL)
        {
            return Err(EmitError::new(format!(
                "stage8 program missing `{EVALUATION_POINT_SOURCE_SYMBOL}` opening-point source"
            )));
        }
        let input_symbols = self
            .opening_inputs
            .iter()
            .map(|input| input.symbol.as_str())
            .collect::<BTreeSet<_>>();
        for claim in &self.opening_claims {
            if !input_symbols.contains(claim.point_source.as_str()) {
                return Err(EmitError::new(format!(
                    "stage8 claim `{}` point source `{}` is not an opening input",
                    claim.symbol, claim.point_source
                )));
            }
            if claim.point_source != claim.eval_source {
                return Err(EmitError::new(format!(
                    "stage8 claim `{}` must take point and eval from the same opening input",
                    claim.symbol
                )));
            }
        }
        Ok(())
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_stage8.rs",
            Role::Verifier => "verify_stage8.rs",
        }
    }

    fn emit_source(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str("#![allow(clippy::too_many_lines)]\n\n");
        source.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        source.push_str(
            "pub struct Stage8Params {\n    pub field: &'static str,\n    pub pcs: &'static str,\n    pub transcript: &'static str,\n}\n\n",
        );
        source.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        source.push_str(
            "pub struct Stage8OpeningInputPlan {\n    pub symbol: &'static str,\n    pub source_stage: &'static str,\n    pub source_claim: &'static str,\n    pub oracle: &'static str,\n    pub domain: &'static str,\n    pub point_arity: usize,\n    pub claim_kind: &'static str,\n}\n\n",
        );
        source.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        source.push_str(
            "pub struct Stage8OpeningClaimPlan {\n    pub symbol: &'static str,\n    pub oracle: &'static str,\n    pub family: &'static str,\n    pub domain: &'static str,\n    pub point_arity: usize,\n    pub point_source: &'static str,\n    pub eval_source: &'static str,\n    pub source_stage: &'static str,\n    pub source_claim: &'static str,\n}\n\n",
        );
        source.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        source.push_str(
            "pub struct Stage8OpeningBatchPlan {\n    pub symbol: &'static str,\n    pub proof_slot: &'static str,\n    pub policy: &'static str,\n    pub count: usize,\n    pub ordered_claims: &'static [&'static str],\n}\n\n",
        );
        source.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        source.push_str(
            "pub struct Stage8PcsProofPlan {\n    pub symbol: &'static str,\n    pub mode: &'static str,\n    pub pcs: &'static str,\n    pub proof_slot: &'static str,\n    pub transcript_label: &'static str,\n    pub batch: &'static str,\n}\n\n",
        );
        source.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        source.push_str(
            "pub struct Stage8EvaluationProgramPlan {\n    pub role: &'static str,\n    pub function: &'static str,\n    pub params: Stage8Params,\n    pub evaluation_point_source: Stage8OpeningInputPlan,\n    pub opening_inputs: &'static [Stage8OpeningInputPlan],\n    pub opening_claims: &'static [Stage8OpeningClaimPlan],\n    pub opening_batch: Stage8OpeningBatchPlan,\n    pub pcs_proof: Stage8PcsProofPlan,\n}\n\n",
        );

        push_format(
            &mut source,
            format_args!(
                "pub const STAGE8_PARAMS: Stage8Params = Stage8Params {{ field: {}, pcs: {}, transcript: {} }};\n\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript),
            ),
        );
        let point_source = self
            .opening_inputs
            .iter()
            .find(|input| input.symbol == EVALUATION_POINT_SOURCE_SYMBOL)
            .ok_or_else(|| {
                EmitError::new(format!(
                    "evaluation program missing `{EVALUATION_POINT_SOURCE_SYMBOL}` opening-point source"
                ))
            })?;
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE8_EVALUATION_POINT_SOURCE: Stage8OpeningInputPlan = {};\n\n",
                opening_input_literal(point_source),
            ),
        );
        source.push_str("pub const STAGE8_OPENING_INPUTS: &[Stage8OpeningInputPlan] = &[\n");
        for input in &self.opening_inputs {
            push_format(
                &mut source,
                format_args!("    {},\n", opening_input_literal(input)),
            );
        }
        source.push_str("];\n\n");
        source.push_str("pub const STAGE8_OPENING_CLAIMS: &[Stage8OpeningClaimPlan] = &[\n");
        for claim in &self.opening_claims {
            push_format(
                &mut source,
                format_args!(
                    "    Stage8OpeningClaimPlan {{ symbol: {}, oracle: {}, family: {}, domain: {}, point_arity: {}, point_source: {}, eval_source: {}, source_stage: {}, source_claim: {} }},\n",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.family),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source),
                    rust_str(&claim.source_stage),
                    rust_str(&claim.source_claim),
                ),
            );
        }
        source.push_str("];\n\n");
        let batch = &self.opening_batches[0];
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE8_OPENING_BATCH_ORDERED_CLAIMS: &[&str] = &{};\n\n",
                rust_str_array(&batch.ordered_claims),
            ),
        );
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE8_OPENING_BATCH: Stage8OpeningBatchPlan = Stage8OpeningBatchPlan {{ symbol: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE8_OPENING_BATCH_ORDERED_CLAIMS }};\n\n",
                rust_str(&batch.symbol),
                rust_str(&batch.proof_slot),
                rust_str(&batch.policy),
                batch.count,
            ),
        );
        let proof = &self.pcs_proofs[0];
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE8_PCS_PROOF: Stage8PcsProofPlan = Stage8PcsProofPlan {{ symbol: {}, mode: {}, pcs: {}, proof_slot: {}, transcript_label: {}, batch: {} }};\n\n",
                rust_str(&proof.symbol),
                rust_str(&proof.mode),
                rust_str(&proof.pcs),
                rust_str(&proof.proof_slot),
                rust_str(&proof.transcript_label),
                rust_str(&proof.batch),
            ),
        );
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE8_PROGRAM: Stage8EvaluationProgramPlan = Stage8EvaluationProgramPlan {{\n    role: {},\n    function: {},\n    params: STAGE8_PARAMS,\n    evaluation_point_source: STAGE8_EVALUATION_POINT_SOURCE,\n    opening_inputs: STAGE8_OPENING_INPUTS,\n    opening_claims: STAGE8_OPENING_CLAIMS,\n    opening_batch: STAGE8_OPENING_BATCH,\n    pcs_proof: STAGE8_PCS_PROOF,\n}};\n",
                rust_str(self.role.as_str()),
                rust_str(&self.function),
            ),
        );
        Ok(source)
    }
}

fn opening_input_literal(input: &Stage8OpeningInputPlan) -> String {
    format!(
        "Stage8OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }}",
        rust_str(&input.symbol),
        rust_str(&input.source_stage),
        rust_str(&input.source_claim),
        rust_str(&input.oracle),
        rust_str(&input.domain),
        input.point_arity,
        rust_str(&input.claim_kind),
    )
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn rust_str_array(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| rust_str(value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn string_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

fn symbol_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol reference"))
}

fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, EmitError> {
    let value = operation
        .attribute(attr)
        .ok()
        .and_then(|attr| attr.to_string().strip_suffix(" : i64").map(str::to_owned))
        .ok_or_else(|| attr_error(operation, attr, "integer"))?;
    value
        .parse()
        .map_err(|_| attr_error(operation, attr, "integer"))
}

fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let value = operation
        .attribute(attr)
        .ok()
        .map(|attr| attr.to_string())
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&value).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

fn parse_symbol_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().strip_prefix('@').map(str::to_owned))
        .collect()
}

fn operand_symbols(
    operation: OperationRef<'_, '_>,
    start_index: usize,
) -> Result<Vec<String>, EmitError> {
    (start_index..operation.operand_count())
        .map(|index| operand_symbol(operation, index))
        .collect()
}

fn operand_symbol(operation: OperationRef<'_, '_>, index: usize) -> Result<String, EmitError> {
    let operand = operation.operand(index).map_err(|_| {
        EmitError::new(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        EmitError::new(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    string_attr(owner.owner(), "sym_name")
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> EmitError {
    EmitError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}

fn operation_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}
