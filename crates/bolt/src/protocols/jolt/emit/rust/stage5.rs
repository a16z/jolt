#![expect(
    clippy::needless_raw_string_hashes,
    reason = "generated Rust templates are kept as raw string blocks for copyable output"
)]

use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use super::output_claims::{
    FieldExprDependencies, StructuredPolynomialEvalPlan as Stage5StructuredPolynomialEvalPlan,
    StructuredPolynomialPointPlan as Stage5StructuredPolynomialPointPlan,
    SumcheckOutputClaimAst as Stage5SumcheckOutputClaimAst,
    SumcheckOutputClaimPlan as Stage5SumcheckOutputClaimPlan,
};
use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
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
    pub output_values: Vec<Stage5StructuredPolynomialEvalPlan>,
    pub output_claims: Vec<Stage5SumcheckOutputClaimPlan>,
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

impl FieldExprDependencies for Stage5FieldExprPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn operands(&self) -> &[String] {
        &self.operands
    }
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
        source: program.emit_source()?,
    })
}

impl Stage5CpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut steps = Vec::new();
        let mut transcript_squeezes = Vec::new();
        let mut transcript_absorb_bytes = Vec::new();
        let mut opening_inputs = Vec::new();
        let mut field_constants = Vec::new();
        let mut field_exprs = Vec::new();
        let mut kernels = Vec::new();
        let mut claims = Vec::new();
        let mut batches = Vec::new();
        let mut drivers = Vec::new();
        let mut instance_results = Vec::new();
        let mut evals = Vec::new();
        let mut output_values = Vec::new();
        let mut output_claim_asts = Vec::new();
        let mut point_slices = Vec::new();
        let mut point_concats = Vec::new();
        let mut opening_claims = Vec::new();
        let mut opening_equalities = Vec::new();
        let mut opening_batches = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(Stage5Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage5KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage5ProgramStepPlan {
                        kind: "transcript_squeeze".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_squeezes.push(Stage5TranscriptSqueezePlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.transcript_absorb_bytes" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage5ProgramStepPlan {
                        kind: "transcript_absorb_bytes".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_absorb_bytes.push(Stage5TranscriptAbsorbBytesPlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        payload: string_attr(op, "payload")?,
                    });
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage5OpeningInputPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source_stage: symbol_attr(op, "source_stage")?,
                        source_claim: symbol_attr(op, "source_claim")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                    });
                }
                "cpu.field_const" => {
                    field_constants.push(Stage5FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.field_zero" => {
                    field_constants.push(Stage5FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 0,
                    });
                }
                "cpu.field_one" => {
                    field_constants.push(Stage5FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 1,
                    });
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage5FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: operation_name(op).replace("cpu.field_", "field."),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.field_pow" => {
                    let exponent = int_attr(op, "exponent")?;
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage5FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: format!("field.pow:{exponent}"),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.sumcheck_claim" => {
                    claims.push(Stage5SumcheckClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        domain: symbol_attr(op, "domain")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                        claim: symbol_attr(op, "claim")?,
                        kernel: Some(symbol_attr(op, "kernel")?),
                        relation: None,
                        claim_value: operand_symbol(op, 0)?,
                        input_openings: operand_symbols(op, 1)?,
                    });
                }
                "cpu.sumcheck_verify_claim" => {
                    claims.push(Stage5SumcheckClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        domain: symbol_attr(op, "domain")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                        claim: symbol_attr(op, "claim")?,
                        kernel: None,
                        relation: Some(symbol_attr(op, "relation")?),
                        claim_value: operand_symbol(op, 0)?,
                        input_openings: operand_symbols(op, 1)?,
                    });
                }
                "cpu.sumcheck_batch" => {
                    batches.push(Stage5SumcheckBatchPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        policy: string_attr(op, "policy")?,
                        count: int_attr(op, "count")?,
                        ordered_claims: symbol_array_attr(op, "ordered_claims")?,
                        claim_operands: operand_symbols(op, 0)?,
                        claim_label: string_attr(op, "claim_label")?,
                        round_label: string_attr(op, "round_label")?,
                        round_schedule: int_array_attr(op, "round_schedule")?,
                    });
                }
                "cpu.sumcheck_driver" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage5ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage5SumcheckDriverPlan {
                        symbol,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        kernel: Some(symbol_attr(op, "kernel")?),
                        relation: None,
                        batch: operand_symbol(op, 1)?,
                        policy: string_attr(op, "policy")?,
                        round_schedule: int_array_attr(op, "round_schedule")?,
                        claim_label: string_attr(op, "claim_label")?,
                        round_label: string_attr(op, "round_label")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                    });
                }
                "cpu.sumcheck_verify" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage5ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage5SumcheckDriverPlan {
                        symbol,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        kernel: None,
                        relation: Some(symbol_attr(op, "relation")?),
                        batch: operand_symbol(op, 1)?,
                        policy: string_attr(op, "policy")?,
                        round_schedule: int_array_attr(op, "round_schedule")?,
                        claim_label: string_attr(op, "claim_label")?,
                        round_label: string_attr(op, "round_label")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                    });
                }
                "cpu.sumcheck_instance_result" => {
                    instance_results.push(Stage5SumcheckInstanceResultPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        claim: symbol_attr(op, "claim")?,
                        relation: symbol_attr(op, "relation")?,
                        index: int_attr(op, "index")?,
                        point_arity: int_attr(op, "point_arity")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        round_offset: int_attr(op, "round_offset")?,
                        point_order: string_attr(op, "point_order")?,
                        degree: int_attr(op, "degree")?,
                    });
                }
                "cpu.sumcheck_eval" => {
                    evals.push(Stage5SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.structured_polynomial_eval" => {
                    output_values.push(Stage5StructuredPolynomialEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        polynomial: string_attr(op, "polynomial")?,
                        x_point: Stage5StructuredPolynomialPointPlan {
                            source: operand_symbol(op, 0)?,
                            segment: string_attr(op, "x_point_segment")?,
                            length: string_attr(op, "x_point_length")?,
                            order: string_attr(op, "x_point_order")?,
                        },
                        y_point: Stage5StructuredPolynomialPointPlan {
                            source: operand_symbol(op, 1)?,
                            segment: string_attr(op, "y_point_segment")?,
                            length: string_attr(op, "y_point_length")?,
                            order: string_attr(op, "y_point_order")?,
                        },
                    });
                }
                "cpu.sumcheck_output_claim" => {
                    output_claim_asts.push(Stage5SumcheckOutputClaimAst {
                        relation: symbol_attr(op, "relation")?,
                        claim_value: operand_symbol(op, 0)?,
                        polynomial_evals: symbol_array_attr(op, "polynomial_evals")?,
                        polynomial_eval_operands: operand_symbols(op, 1)?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(Stage5PointSlicePlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        offset: int_attr(op, "offset")?,
                        length: int_attr(op, "length")?,
                        input: operand_symbol(op, 0)?,
                    });
                }
                "cpu.point_concat" => {
                    point_concats.push(Stage5PointConcatPlan {
                        symbol: string_attr(op, "sym_name")?,
                        layout: string_attr(op, "layout")?,
                        arity: int_attr(op, "arity")?,
                        inputs: operand_symbols(op, 0)?,
                    });
                }
                "cpu.opening_claim" => {
                    opening_claims.push(Stage5OpeningClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                        point_source: operand_symbol(op, 0)?,
                        eval_source: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_claim_equal" => {
                    opening_equalities.push(Stage5OpeningClaimEqualityPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: string_attr(op, "mode")?,
                        lhs: operand_symbol(op, 0)?,
                        rhs: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_batch" => {
                    opening_batches.push(Stage5OpeningBatchPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        policy: string_attr(op, "policy")?,
                        count: int_attr(op, "count")?,
                        ordered_claims: symbol_array_attr(op, "ordered_claims")?,
                        claim_operands: operand_symbols(op, 0)?,
                    });
                }
                _ => {}
            }
        }

        let role = module
            .role()
            .ok_or_else(|| EmitError::new("missing cpu party role"))?;
        if role == Role::Prover {
            super::output_claims::prune_output_only_field_exprs(
                &mut field_exprs,
                claims.iter().map(|claim| claim.claim_value.as_str()),
                output_claim_asts
                    .iter()
                    .map(|claim| claim.claim_value.as_str()),
            );
        }
        let output_claims = if role == Role::Verifier {
            super::output_claims::resolve_output_claims(
                "stage5",
                &output_values,
                &[],
                &[],
                &[],
                &field_exprs,
                output_claim_asts,
            )?
        } else {
            Vec::new()
        };

        Ok(Self {
            params: params.ok_or_else(|| EmitError::new("missing cpu.params"))?,
            role,
            steps,
            transcript_squeezes,
            transcript_absorb_bytes,
            opening_inputs,
            field_constants,
            field_exprs,
            kernels,
            claims,
            batches,
            drivers,
            instance_results,
            evals,
            output_values,
            output_claims,
            point_slices,
            point_concats,
            opening_claims,
            opening_equalities,
            opening_batches,
        })
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol("field", &self.params.field, "bn254_fr")?;
        require_supported_symbol("pcs", &self.params.pcs, "dory")?;
        require_supported_symbol("transcript", &self.params.transcript, "blake2b_transcript")?;
        self.verify_transcript_steps()?;
        self.verify_field_flow()?;
        self.verify_claim_batches()?;
        match self.role {
            Role::Prover => {
                self.verify_kernel_definitions()?;
                self.verify_prover_driver_bindings()?;
            }
            Role::Verifier => self.verify_verifier_driver_bindings()?,
        }
        if self.role == Role::Verifier {
            self.verify_output_claims()?;
        }
        self.verify_opening_flow()
    }

    fn verify_transcript_steps(&self) -> Result<(), EmitError> {
        for squeeze in &self.transcript_squeezes {
            if !matches!(
                squeeze.kind.as_str(),
                "challenge_scalar" | "challenge_vector"
            ) {
                return Err(EmitError::new(format!(
                    "stage5 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage5 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        for absorb in &self.transcript_absorb_bytes {
            if absorb.label.is_empty() {
                return Err(EmitError::new(format!(
                    "stage5 transcript byte absorb @{} has empty label",
                    absorb.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_field_flow(&self) -> Result<(), EmitError> {
        for constant in &self.field_constants {
            require_supported_symbol("field constant field", &constant.field, "bn254_fr")?;
        }
        let field_values = self.field_value_symbols();
        for expr in &self.field_exprs {
            verify_count(
                "field expr operands",
                &expr.symbol,
                expr.operand_names.len(),
                expr.operands.len(),
            )?;
            for operand in &expr.operands {
                if !field_values.contains(operand) {
                    return Err(EmitError::new(format!(
                        "field expr @{} references missing field value @{operand}",
                        expr.symbol
                    )));
                }
            }
        }
        for claim in &self.claims {
            if !field_values.contains(&claim.claim_value) {
                return Err(EmitError::new(format!(
                    "sumcheck claim @{} uses missing claim value @{}",
                    claim.symbol, claim.claim_value
                )));
            }
        }
        Ok(())
    }

    fn field_value_symbols(&self) -> BTreeSet<String> {
        let mut values = symbols(self.opening_inputs.iter().map(|input| &input.symbol));
        values.extend(symbols(
            self.field_constants.iter().map(|constant| &constant.symbol),
        ));
        values.extend(symbols(
            self.transcript_squeezes
                .iter()
                .filter(|squeeze| matches!(squeeze.kind.as_str(), "challenge_scalar" | "scalar"))
                .map(|squeeze| &squeeze.symbol),
        ));
        values.extend(symbols(
            self.output_values.iter().map(|value| &value.symbol),
        ));
        values.extend(symbols(self.field_exprs.iter().map(|expr| &expr.symbol)));
        values.extend(symbols(self.evals.iter().map(|eval| &eval.symbol)));
        values
    }

    fn point_value_symbols(&self) -> BTreeSet<String> {
        let mut values = symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.symbol),
        );
        values.extend(symbols(
            self.opening_inputs.iter().map(|input| &input.symbol),
        ));
        values.extend(symbols(self.point_slices.iter().map(|slice| &slice.symbol)));
        values.extend(symbols(
            self.point_concats.iter().map(|concat| &concat.symbol),
        ));
        values
    }

    fn verify_kernel_definitions(&self) -> Result<(), EmitError> {
        for kernel in &self.kernels {
            if kernel.backend != "cpu" {
                return Err(EmitError::new(format!(
                    "stage5 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage5 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage5.instruction_read_raf" => "jolt_stage5_instruction_read_raf",
                "jolt.stage5.ram_ra_claim_reduction" => "jolt_stage5_ram_ra_claim_reduction",
                "jolt.stage5.registers_val_evaluation" => "jolt_stage5_registers_val_evaluation",
                "jolt.stage5.batched" => "jolt_stage5_batched",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage5 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage5 kernel @{} ABI `{}` does not match relation @{}",
                    kernel.symbol, kernel.abi, kernel.relation
                )));
            }
        }
        Ok(())
    }

    fn verify_claim_batches(&self) -> Result<(), EmitError> {
        let claims = symbols(self.claims.iter().map(|claim| &claim.symbol));
        for batch in &self.batches {
            verify_count(
                "sumcheck batch",
                &batch.symbol,
                batch.count,
                batch.ordered_claims.len(),
            )?;
            verify_count(
                "sumcheck batch operands",
                &batch.symbol,
                batch.count,
                batch.claim_operands.len(),
            )?;
            if batch.ordered_claims != batch.claim_operands {
                return Err(EmitError::new(format!(
                    "sumcheck batch @{} operand order does not match ordered_claims",
                    batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                if !claims.contains(claim) {
                    return Err(EmitError::new(format!(
                        "sumcheck batch @{} references missing claim @{claim}",
                        batch.symbol
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_prover_driver_bindings(&self) -> Result<(), EmitError> {
        let kernels = symbols(self.kernels.iter().map(|kernel| &kernel.symbol));
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        for claim in &self.claims {
            let Some(kernel) = claim.kernel.as_deref() else {
                return Err(EmitError::new(format!(
                    "prover sumcheck claim @{} is missing kernel",
                    claim.symbol
                )));
            };
            if !kernels.contains(kernel) {
                return Err(EmitError::new(format!(
                    "sumcheck claim @{} references missing kernel @{kernel}",
                    claim.symbol
                )));
            }
        }
        for driver in &self.drivers {
            let Some(kernel) = driver.kernel.as_deref() else {
                return Err(EmitError::new(format!(
                    "prover sumcheck driver @{} is missing kernel",
                    driver.symbol
                )));
            };
            if !kernels.contains(kernel) {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} references missing kernel @{kernel}",
                    driver.symbol
                )));
            }
            let batch = batches.get(driver.batch.as_str()).ok_or_else(|| {
                EmitError::new(format!(
                    "sumcheck driver @{} references missing batch @{}",
                    driver.symbol, driver.batch
                ))
            })?;
            verify_count(
                "sumcheck driver round_schedule",
                &driver.symbol,
                driver.num_rounds,
                driver.round_schedule.iter().sum(),
            )?;
            if driver.round_schedule != batch.round_schedule {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} round_schedule differs from batch @{}",
                    driver.symbol, batch.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_verifier_driver_bindings(&self) -> Result<(), EmitError> {
        if !self.kernels.is_empty() {
            return Err(EmitError::new(
                "verifier stage5 program must not contain kernels",
            ));
        }
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        for claim in &self.claims {
            if claim.kernel.is_some() || claim.relation.is_none() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck claim @{} must carry relation and no kernel",
                    claim.symbol
                )));
            }
        }
        for driver in &self.drivers {
            if driver.kernel.is_some() || driver.relation.is_none() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck driver @{} must carry relation and no kernel",
                    driver.symbol
                )));
            }
            let batch = batches.get(driver.batch.as_str()).ok_or_else(|| {
                EmitError::new(format!(
                    "sumcheck driver @{} references missing batch @{}",
                    driver.symbol, driver.batch
                ))
            })?;
            verify_count(
                "sumcheck driver round_schedule",
                &driver.symbol,
                driver.num_rounds,
                driver.round_schedule.iter().sum(),
            )?;
            if driver.round_schedule != batch.round_schedule {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} round_schedule differs from batch @{}",
                    driver.symbol, batch.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_output_claims(&self) -> Result<(), EmitError> {
        let relations = symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.relation),
        );
        let field_values = self.field_value_symbols();
        let point_values = self.point_value_symbols();
        super::output_claims::verify_output_claims(
            "stage5",
            super::output_claims::OutputClaimVerification {
                output_values: &self.output_values,
                output_families: &[],
                output_product_families: &[],
                output_function_families: &[],
                output_claims: &self.output_claims,
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        )
    }

    fn verify_opening_flow(&self) -> Result<(), EmitError> {
        let mut point_sources = symbols(self.drivers.iter().map(|driver| &driver.symbol));
        point_sources.extend(symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.symbol),
        ));
        point_sources.extend(symbols(
            self.opening_inputs.iter().map(|input| &input.symbol),
        ));
        point_sources.extend(symbols(self.point_slices.iter().map(|slice| &slice.symbol)));
        point_sources.extend(symbols(
            self.point_concats.iter().map(|concat| &concat.symbol),
        ));
        for slice in &self.point_slices {
            if !point_sources.contains(&slice.input) {
                return Err(EmitError::new(format!(
                    "point slice @{} uses missing point source @{}",
                    slice.symbol, slice.input
                )));
            }
        }
        for concat in &self.point_concats {
            for input in &concat.inputs {
                if !point_sources.contains(input) {
                    return Err(EmitError::new(format!(
                        "point concat @{} uses missing point source @{input}",
                        concat.symbol
                    )));
                }
            }
        }
        let eval_sources = self.field_value_symbols();
        let mut opening_sources = symbols(self.opening_inputs.iter().map(|input| &input.symbol));
        opening_sources.extend(symbols(
            self.opening_claims.iter().map(|claim| &claim.symbol),
        ));
        for equality in &self.opening_equalities {
            if !opening_sources.contains(&equality.lhs) {
                return Err(EmitError::new(format!(
                    "opening equality @{} uses missing lhs opening @{}",
                    equality.symbol, equality.lhs
                )));
            }
            if !opening_sources.contains(&equality.rhs) {
                return Err(EmitError::new(format!(
                    "opening equality @{} uses missing rhs opening @{}",
                    equality.symbol, equality.rhs
                )));
            }
        }
        for claim in &self.claims {
            for input in &claim.input_openings {
                if !opening_sources.contains(input) {
                    return Err(EmitError::new(format!(
                        "sumcheck claim @{} uses missing opening @{input}",
                        claim.symbol
                    )));
                }
            }
        }
        let drivers = symbols(self.drivers.iter().map(|driver| &driver.symbol));
        for instance in &self.instance_results {
            if !drivers.contains(&instance.source) {
                return Err(EmitError::new(format!(
                    "sumcheck instance result @{} references missing driver @{}",
                    instance.symbol, instance.source
                )));
            }
        }
        for eval in &self.evals {
            if !drivers.contains(&eval.source) {
                return Err(EmitError::new(format!(
                    "sumcheck eval @{} references missing driver @{}",
                    eval.symbol, eval.source
                )));
            }
        }
        for claim in &self.opening_claims {
            if !point_sources.contains(&claim.point_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing point source @{}",
                    claim.symbol, claim.point_source
                )));
            }
            if !eval_sources.contains(&claim.eval_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing eval source @{}",
                    claim.symbol, claim.eval_source
                )));
            }
        }
        let openings = symbols(self.opening_claims.iter().map(|claim| &claim.symbol));
        for batch in &self.opening_batches {
            verify_count(
                "opening batch",
                &batch.symbol,
                batch.count,
                batch.ordered_claims.len(),
            )?;
            verify_count(
                "opening batch operands",
                &batch.symbol,
                batch.count,
                batch.claim_operands.len(),
            )?;
            if batch.ordered_claims != batch.claim_operands {
                return Err(EmitError::new(format!(
                    "opening batch @{} operand order does not match ordered_claims",
                    batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                if !openings.contains(claim) {
                    return Err(EmitError::new(format!(
                        "opening batch @{} references missing opening @{claim}",
                        batch.symbol
                    )));
                }
            }
        }
        Ok(())
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_stage5.rs",
            Role::Verifier => "verify_stage5.rs",
        }
    }

    fn emit_source(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        match self.role {
            Role::Prover => {
                source.push_str(Self::emit_prover_imports());
                source.push_str("\n\n");
                source.push_str(Self::emit_prover_types());
            }
            Role::Verifier => {
                source.push_str(Self::emit_verifier_imports());
                source.push_str("\n\n");
                source.push_str(&Self::emit_verifier_types());
            }
        }
        source.push('\n');
        source.push_str(&self.emit_constants()?);
        source.push('\n');
        source.push_str(self.emit_entrypoint());
        Ok(source)
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage5::{execute_stage5_program, Stage5CpuProgramPlan, Stage5ExecutionArtifacts, Stage5ExecutionMode, Stage5FieldConstantPlan, Stage5FieldExprPlan, Stage5KernelError, Stage5KernelExecutor, Stage5KernelPlan, Stage5OpeningBatchPlan, Stage5OpeningClaimEqualityPlan, Stage5OpeningClaimPlan, Stage5OpeningInputPlan, Stage5Params, Stage5PointConcatPlan, Stage5PointSlicePlan, Stage5ProgramStepPlan, Stage5SumcheckBatchPlan, Stage5SumcheckClaimPlan, Stage5SumcheckDriverPlan, Stage5SumcheckEvalPlan, Stage5SumcheckInstanceResultPlan, Stage5TranscriptAbsorbBytesPlan, Stage5TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage5Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use bolt_verifier_runtime::{batch_claims, eval_by_name, eval_family_values, find_batch, find_plan, reverse_slice, NamedEvalFamilyPlan};\n\
         use super::jolt_relations::{identity_polynomial_eval, normalize_instruction_read_raf_point, operand_polynomial_eval};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_lookup_tables::LookupTableKind;\n\
         use jolt_poly::EqPolynomial;\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    #[expect(dead_code)]
    fn emit_types() -> &'static str {
        r#"#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckClaimPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<&'static str>,
    pub claim_value: &'static str,
    pub input_openings: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub round_schedule: &'static [usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckDriverPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<&'static str>,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckInstanceResultPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub claim: &'static str,
    pub relation: &'static str,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: &'static str,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage5Params,
    pub steps: &'static [Stage5ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage5TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage5TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage5OpeningInputPlan],
    pub field_constants: &'static [Stage5FieldConstantPlan],
    pub field_exprs: &'static [Stage5FieldExprPlan],
    pub kernels: &'static [Stage5KernelPlan],
    pub claims: &'static [Stage5SumcheckClaimPlan],
    pub batches: &'static [Stage5SumcheckBatchPlan],
    pub drivers: &'static [Stage5SumcheckDriverPlan],
    pub instance_results: &'static [Stage5SumcheckInstanceResultPlan],
    pub evals: &'static [Stage5SumcheckEvalPlan],
    pub point_slices: &'static [Stage5PointSlicePlan],
    pub point_concats: &'static [Stage5PointConcatPlan],
    pub opening_claims: &'static [Stage5OpeningClaimPlan],
    pub opening_equalities: &'static [Stage5OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage5OpeningBatchPlan],
}
"#
    }

    fn emit_verifier_type_aliases() -> &'static str {
        r#"pub type Stage5NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage5SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage5ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage5ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage5Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage5OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage5CpuProgramPlan = bolt_verifier_runtime::StageProgramPlanNoPointZeros<Stage5RelationKind>;
pub type Stage5SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage5RelationKind>;
pub type Stage5SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage5RelationKind>;
pub type Stage5SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage5RelationKind>;
pub type Stage5SumcheckOutputClaimPlan = bolt_verifier_runtime::SumcheckOutputClaimPlan<Stage5RelationKind>;
pub type Stage5StructuredPolynomialEvalPlan = bolt_verifier_runtime::StructuredPolynomialEvalPlan;

pub use super::jolt_relations::JoltRelationKind as Stage5RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage5ClaimKind, FieldConstantPlan as Stage5FieldConstantPlan,
    FieldExprKind as Stage5FieldExprKind,
    FieldExprPlan as Stage5FieldExprPlan,
    KernelPlan as Stage5KernelPlan, OpeningBatchPlan as Stage5OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage5OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage5OpeningClaimPlan, OpeningInputPlan as Stage5OpeningInputPlan,
    OpeningEqualityMode as Stage5OpeningEqualityMode,
    PointConcatPlan as Stage5PointConcatPlan, PointSlicePlan as Stage5PointSlicePlan,
    ProgramStepKind as Stage5ProgramStepKind,
    ProgramStepPlan as Stage5ProgramStepPlan, StageParams as Stage5Params,
    SumcheckBatchPlan as Stage5SumcheckBatchPlan,
    SumcheckEvalPlan as Stage5SumcheckEvalPlan,
    StructuredPolynomialPointLength as Stage5StructuredPolynomialPointLength,
    StructuredPolynomialPointOrder as Stage5StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan as Stage5StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment as Stage5StructuredPolynomialPointSegment,
    StructuredPolynomialKind as Stage5StructuredPolynomialKind,
    TranscriptAbsorbBytesPlan as Stage5TranscriptAbsorbBytesPlan,
    TranscriptSqueezeKind as Stage5TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage5TranscriptSqueezePlan,
};
"#
    }

    fn emit_verifier_types() -> String {
        let mut source = Self::emit_verifier_type_aliases().to_owned();
        source.push_str(
            r#"
pub type DefaultStage5Transcript = Blake2bTranscript<Fr>;
pub type Stage5VerifierProgramPlan = Stage5CpuProgramPlan;

#[derive(Debug)]
pub enum VerifyStage5Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage5RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage5Error);
"#,
        );
        source
    }

    fn emit_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants()?;
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants()?);
        if self.role == Role::Verifier {
            source.push_str(&self.emit_verifier_output_claim_constants()?);
        }
        let output_claims_field = if self.role == Role::Verifier {
            "    output_claims: STAGE5_SUMCHECK_OUTPUT_CLAIMS,\n"
        } else {
            ""
        };
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_PROGRAM: {} = Stage5CpuProgramPlan {{\n\
                 \x20   role: {},\n\
                 \x20   params: STAGE5_PARAMS,\n\
                 \x20   steps: STAGE5_PROGRAM_STEPS,\n\
                 \x20   transcript_squeezes: STAGE5_TRANSCRIPT_SQUEEZES,\n\
                 \x20   transcript_absorb_bytes: STAGE5_TRANSCRIPT_ABSORB_BYTES,\n\
                 \x20   opening_inputs: STAGE5_OPENING_INPUTS,\n\
                 \x20   field_constants: STAGE5_FIELD_CONSTANTS,\n\
                 \x20   field_exprs: STAGE5_FIELD_EXPRS,\n\
                 \x20   kernels: STAGE5_KERNELS,\n\
                 \x20   claims: STAGE5_SUMCHECK_CLAIMS,\n\
                 \x20   batches: STAGE5_SUMCHECK_BATCHES,\n\
                 \x20   drivers: STAGE5_SUMCHECK_DRIVERS,\n\
                 \x20   instance_results: STAGE5_SUMCHECK_INSTANCE_RESULTS,\n\
                 \x20   evals: STAGE5_SUMCHECK_EVALS,\n\
                 {output_claims_field}\
                 \x20   point_slices: STAGE5_POINT_SLICES,\n\
                 \x20   point_concats: STAGE5_POINT_CONCATS,\n\
                 \x20   opening_claims: STAGE5_OPENING_CLAIMS,\n\
                 \x20   opening_equalities: STAGE5_OPENING_EQUALITIES,\n\
                 \x20   opening_batches: STAGE5_OPENING_BATCHES,\n\
                 }};\n",
                self.program_plan_type(),
                rust_str(self.role_label())
            ),
        );
        Ok(source)
    }

    fn emit_shared_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_PARAMS: Stage5Params = Stage5Params {{ field: {}, pcs: {}, transcript: {} }};\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript)
            ),
        );
        source.push_str(&self.emit_program_step_constants()?);
        source.push_str(&self.emit_transcript_squeeze_constants()?);
        source.push_str(&self.emit_transcript_absorb_bytes_constants());
        source.push_str(&self.emit_opening_input_constants()?);
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_field_expr_constants()?);
        Ok(source)
    }

    fn emit_program_step_constants(&self) -> Result<String, EmitError> {
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(format!(
                    "    Stage5ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    super::plan_tokens::role_program_step_kind_expr(
                        "Stage5", &self.role, &step.kind
                    )?,
                    rust_str(&step.symbol),
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE5_PROGRAM_STEPS: &[Stage5ProgramStepPlan] = &[\n{steps}\n];\n\n"
        ))
    }

    fn emit_transcript_squeeze_constants(&self) -> Result<String, EmitError> {
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                Ok(format!(
                    "    Stage5TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    super::plan_tokens::role_transcript_squeeze_kind_expr("Stage5", &self.role, &squeeze.kind)?,
                    squeeze.count,
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE5_TRANSCRIPT_SQUEEZES: &[Stage5TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        ))
    }

    fn emit_transcript_absorb_bytes_constants(&self) -> String {
        let absorbs = self
            .transcript_absorb_bytes
            .iter()
            .map(|absorb| {
                format!(
                    "    Stage5TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                    rust_str(&absorb.symbol),
                    rust_str(&absorb.label),
                    rust_str(&absorb.payload),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE5_TRANSCRIPT_ABSORB_BYTES: &[Stage5TranscriptAbsorbBytesPlan] = &[\n{absorbs}\n];\n\n"
        )
    }

    fn emit_opening_input_constants(&self) -> Result<String, EmitError> {
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                Ok(format!(
                    "    Stage5OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage5", &self.role, &input.claim_kind)?
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE5_OPENING_INPUTS: &[Stage5OpeningInputPlan] = &[\n{inputs}\n];\n\n"
        ))
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage5FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE5_FIELD_CONSTANTS: &[Stage5FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let exprs = self
                .field_exprs
                .iter()
                .map(|expr| {
                    Ok(format!(
                        "    Stage5FieldExprPlan {{ symbol: {}, kind: {}, operands: {} }},",
                        rust_str(&expr.symbol),
                        super::plan_tokens::role_field_expr_kind_expr(
                            "Stage5",
                            &self.role,
                            &expr.formula
                        )?,
                        super::plan_tokens::rust_str_slice_expr(&expr.operands)
                    ))
                })
                .collect::<Result<Vec<_>, EmitError>>()?
                .join("\n");
            return Ok(format!(
                "pub const STAGE5_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[\n{exprs}\n];\n"
            ));
        }

        let mut source = String::new();
        let mut arrays = Vec::new();
        let mut array_refs = Vec::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            let operands = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE5_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE5_FIELD_EXPR_OPERANDS",
                &expr.operand_names,
            );
            array_refs.push((index, operand_names, operands));
        }
        let exprs = self
            .field_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                let (_, operand_names, operands) = &array_refs[index];
                format!(
                    "    Stage5FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
                    rust_str(&expr.symbol),
                    rust_str(&expr.kind),
                    rust_str(&expr.formula)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[\n{exprs}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_kernel_constants(&self) -> String {
        let kernels = self
            .kernels
            .iter()
            .map(|kernel| {
                format!(
                    "    Stage5KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE5_KERNELS: &[Stage5KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let claims = self
                .claims
                .iter()
                .map(|claim| {
                    Ok(format!(
                        "    Stage5SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {} }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_option_str(claim.kernel.as_deref()),
                        super::plan_tokens::role_optional_relation_kind_expr(
                            "Stage5",
                            &self.role,
                            claim.relation.as_deref()
                        )?,
                        rust_str(&claim.claim_value)
                    ))
                })
                .collect::<Result<Vec<_>, EmitError>>()?
                .join("\n");
            return Ok(format!(
                "pub const STAGE5_SUMCHECK_CLAIMS: &[Stage5SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ));
        }

        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE5_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                Ok(format!(
                    "    Stage5SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: STAGE5_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_option_str(claim.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage5",
                        &self.role,
                        claim.relation.as_deref()
                    )?,
                    rust_str(&claim.claim_value)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_SUMCHECK_CLAIMS: &[Stage5SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE5_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            let batches = self
                .batches
                .iter()
                .enumerate()
                .map(|(index, batch)| {
                    format!(
                        "    Stage5SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE5_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                        rust_str(&batch.symbol),
                        rust_str(&batch.stage),
                        rust_str(&batch.proof_slot),
                        rust_str(&batch.policy),
                        batch.count,
                        super::plan_tokens::rust_str_slice_expr(&batch.claim_operands),
                        rust_str(&batch.claim_label),
                        rust_str(&batch.round_label)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            push_format(
                &mut source,
                format_args!(
                    "pub const STAGE5_SUMCHECK_BATCHES: &[Stage5SumcheckBatchPlan] = &[\n{batches}\n];\n"
                ),
            );
            return source;
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE5_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE5_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE5_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage5SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE5_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE5_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE5_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                    rust_str(&batch.symbol),
                    rust_str(&batch.stage),
                    rust_str(&batch.proof_slot),
                    rust_str(&batch.policy),
                    batch.count,
                    rust_str(&batch.claim_label),
                    rust_str(&batch.round_label)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_SUMCHECK_BATCHES: &[Stage5SumcheckBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE5_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                Ok(format!(
                    "    Stage5SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE5_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                    rust_str(&driver.symbol),
                    rust_str(&driver.stage),
                    rust_str(&driver.proof_slot),
                    rust_option_str(driver.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage5",
                        &self.role,
                        driver.relation.as_deref()
                    )?,
                    rust_str(&driver.batch),
                    rust_str(&driver.policy),
                    rust_str(&driver.claim_label),
                    rust_str(&driver.round_label),
                    driver.num_rounds,
                    driver.degree
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_SUMCHECK_DRIVERS: &[Stage5SumcheckDriverPlan] = &[\n{drivers}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_tail_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants()?);
        source.push_str(&self.emit_sumcheck_eval_constants());
        if self.role == Role::Verifier {
            source.push_str(&self.emit_named_eval_family_constants()?);
        }
        source.push_str(&self.emit_point_slice_constants());
        source.push_str(&self.emit_point_concat_constants());
        source.push_str(&self.emit_opening_claim_constants()?);
        source.push_str(&self.emit_opening_claim_equality_constants()?);
        source.push_str(&self.emit_opening_batch_constants());
        Ok(source)
    }

    fn emit_sumcheck_instance_result_constants(&self) -> Result<String, EmitError> {
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                Ok(format!(
                    "    Stage5SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    super::plan_tokens::role_relation_kind_expr(
                        "Stage5",
                        &self.role,
                        &instance.relation
                    )?,
                    instance.index,
                    instance.point_arity,
                    instance.num_rounds,
                    instance.round_offset,
                    rust_str(&instance.point_order),
                    instance.degree
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE5_SUMCHECK_INSTANCE_RESULTS: &[Stage5SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        ))
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let evals = self
            .evals
            .iter()
            .map(|eval| {
                format!(
                    "    Stage5SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE5_SUMCHECK_EVALS: &[Stage5SumcheckEvalPlan] = &[\n{evals}\n];\n\n")
    }

    fn emit_named_eval_family_constants(&self) -> Result<String, EmitError> {
        const FAMILIES: &[(&str, &str, &str, &str)] = &[
            (
                "STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVAL_NAMES",
                "STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS",
                "stage5.instruction_read_raf.eval.LookupTableFlag",
                "LookupTableFlag_",
            ),
            (
                "STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVAL_NAMES",
                "STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS",
                "stage5.instruction_read_raf.eval.InstructionRa",
                "InstructionRa_",
            ),
        ];

        let mut source = String::new();
        for (names_const, family_const, family_symbol, oracle_prefix) in FAMILIES {
            let names = self.indexed_eval_family_names(family_symbol, oracle_prefix)?;
            let names_source = names
                .iter()
                .map(|name| rust_str(name))
                .collect::<Vec<_>>()
                .join(", ");
            push_format(
                &mut source,
                format_args!(
                    "#[rustfmt::skip]\npub const {names_const}: &[&str] = &[{names_source}];\n"
                ),
            );
            push_format(
                &mut source,
                format_args!(
                    "pub const {family_const}: NamedEvalFamilyPlan = NamedEvalFamilyPlan {{ symbol: {}, evals: {names_const} }};\n\n",
                    rust_str(family_symbol),
                ),
            );
        }
        Ok(source)
    }

    fn indexed_eval_family_names(
        &self,
        family_symbol: &str,
        oracle_prefix: &str,
    ) -> Result<Vec<&str>, EmitError> {
        let mut indexed_names = Vec::new();
        for eval in &self.evals {
            let Some(suffix) = eval.oracle.strip_prefix(oracle_prefix) else {
                continue;
            };
            let index = suffix.parse::<usize>().map_err(|_| {
                EmitError::new(format!(
                    "invalid indexed eval oracle `{}` for family `{family_symbol}`",
                    eval.oracle
                ))
            })?;
            indexed_names.push((index, eval.name.as_str()));
        }
        if indexed_names.is_empty() {
            return Err(EmitError::new(format!(
                "missing eval family `{family_symbol}`"
            )));
        }
        indexed_names.sort_by_key(|(index, _)| *index);
        for (expected, (actual, _)) in indexed_names.iter().enumerate() {
            if expected != *actual {
                return Err(EmitError::new(format!(
                    "non-contiguous eval family `{family_symbol}` at index {actual}"
                )));
            }
        }
        Ok(indexed_names
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>())
    }

    fn emit_verifier_output_claim_constants(&self) -> Result<String, EmitError> {
        super::output_claims::emit_verifier_output_claim_constants(
            "Stage5",
            &self.role,
            &self.output_claims,
        )
    }

    fn emit_point_slice_constants(&self) -> String {
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage5PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
                    rust_str(&slice.symbol),
                    rust_str(&slice.source),
                    slice.offset,
                    slice.length,
                    rust_str(&slice.input)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE5_POINT_SLICES: &[Stage5PointSlicePlan] = &[\n{slices}\n];\n\n")
    }

    fn emit_point_concat_constants(&self) -> String {
        if self.role == Role::Verifier {
            let concats = self
                .point_concats
                .iter()
                .map(|concat| {
                    format!(
                        "    Stage5PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: {} }},",
                        rust_str(&concat.symbol),
                        rust_str(&concat.layout),
                        concat.arity,
                        super::plan_tokens::rust_str_slice_expr(&concat.inputs)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            return format!(
                "pub const STAGE5_POINT_CONCATS: &[Stage5PointConcatPlan] = &[\n{concats}\n];\n"
            );
        }

        let mut source = String::new();
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE5_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage5PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE5_POINT_CONCAT_{index}_INPUTS }},",
                    rust_str(&concat.symbol),
                    rust_str(&concat.layout),
                    concat.arity
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_POINT_CONCATS: &[Stage5PointConcatPlan] = &[\n{concats}\n];\n"
            ),
        );
        source
    }

    fn emit_opening_claim_constants(&self) -> Result<String, EmitError> {
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(format!(
                    "    Stage5OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage5", &self.role, &claim.claim_kind)?,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE5_OPENING_CLAIMS: &[Stage5OpeningClaimPlan] = &[\n{claims}\n];\n\n"
        ))
    }

    fn emit_opening_claim_equality_constants(&self) -> Result<String, EmitError> {
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                Ok(format!(
                    "    Stage5OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    super::plan_tokens::role_opening_equality_mode_expr("Stage5", &self.role, &equality.mode)?,
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE5_OPENING_EQUALITIES: &[Stage5OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        ))
    }

    fn emit_opening_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let batches = self
                .opening_batches
                .iter()
                .map(|batch| {
                    format!(
                        "    Stage5OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
                        rust_str(&batch.symbol),
                        rust_str(&batch.stage),
                        rust_str(&batch.proof_slot),
                        rust_str(&batch.policy),
                        batch.count,
                        super::plan_tokens::rust_str_slice_expr(&batch.ordered_claims),
                        super::plan_tokens::rust_str_slice_expr(&batch.claim_operands)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            return format!(
                "pub const STAGE5_OPENING_BATCHES: &[Stage5OpeningBatchPlan] = &[\n{batches}\n];\n"
            );
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE5_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE5_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage5OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE5_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE5_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
                    rust_str(&batch.symbol),
                    rust_str(&batch.stage),
                    rust_str(&batch.proof_slot),
                    rust_str(&batch.policy),
                    batch.count
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE5_OPENING_BATCHES: &[Stage5OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage5_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage5ExecutionArtifacts<Fr>, Stage5KernelError>\n\
                 where\n\
                 \x20   E: Stage5KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage5_prover_with_program(&STAGE5_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage5_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage5CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage5ExecutionArtifacts<Fr>, Stage5KernelError>\n\
                 where\n\
                 \x20   E: Stage5KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage5_program(program, Stage5ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage5<T>(
    proof: &Stage5Proof<Fr>,
    opening_inputs: &[Stage5OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage5_with_program(&STAGE5_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage5_with_program<T>(
    program: &'static Stage5VerifierProgramPlan,
    proof: &Stage5Proof<Fr>,
    opening_inputs: &[Stage5OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage5Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage5ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage5ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage5_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage5ProgramStepKind::TranscriptAbsorbBytes => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage5_bytes(absorb, transcript);
            }
            Stage5ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage5Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage5_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage5_verifier_program() -> &'static Stage5VerifierProgramPlan {
    &STAGE5_PROGRAM
}

fn verify_stage5_squeeze<T>(
    program: &'static Stage5VerifierProgramPlan,
    squeeze: &'static Stage5TranscriptSqueezePlan,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage5ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage5Error::from)?;
    artifacts.challenge_vectors.push(Stage5ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage5_bytes<T>(absorb: &'static Stage5TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage5_driver<T>(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    proof: &Stage5Proof<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage5ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage5Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage5Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage5RelationKind::Stage5Batched => {
            verify_batched_stage5(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage5<T>(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    proof: &Stage5SumcheckOutput<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    bolt_verifier_runtime::verify_batched_sumcheck(
        driver,
        proof,
        program.claims,
        program.batches,
        program.field_exprs,
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        store,
        transcript,
        |store, evals, point, batching_coeffs| {
            expected_batched_output_claim(program, driver, store, evals, point, batching_coeffs)
        },
        |store, verified| observe_stage5_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage5Error::Sumcheck { driver, error },
    )
}

fn observe_stage5_sumcheck_output<F: Field>(
    program: &'static Stage5VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage5SumcheckOutput<F>,
) -> Result<(), VerifyStage5Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "instruction_read_raf" => {
                    point = normalize_instruction_read_raf_point(&point, "stage5.instruction_read_raf.point")?;
                }
                _ => {
                    return Err(VerifyStage5Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage5Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage5Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage5Error::InvalidProof { driver, reason },
        |symbol| VerifyStage5Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage5Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage5Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let Some(relation) = claim.relation else {
            return Err(VerifyStage5Error::InvalidProof {
                driver: driver.symbol,
                reason: "missing claim relation",
            });
        };
        let value = match relation {
            Stage5RelationKind::Stage5InstructionReadRaf => {
                expected_instruction_read_raf(store, evals, local_point)?
            }
            Stage5RelationKind::Stage5RamRaClaimReduction => {
                let output_claim = program
                    .output_claims
                    .iter()
                    .find(|output_claim| output_claim.relation == instance.relation)
                    .ok_or(VerifyStage5Error::UnsupportedRelation {
                        relation: instance.relation,
                    })?;
                bolt_verifier_runtime::evaluate_sumcheck_output_claim(
                    output_claim,
                    program.field_exprs,
                    store,
                    instance.symbol,
                    evals,
                    local_point,
                )?
            }
            Stage5RelationKind::Stage5RegistersValEvaluation => {
                let output_claim = program
                    .output_claims
                    .iter()
                    .find(|output_claim| output_claim.relation == instance.relation)
                    .ok_or(VerifyStage5Error::UnsupportedRelation {
                        relation: instance.relation,
                    })?;
                bolt_verifier_runtime::evaluate_sumcheck_output_claim(
                    output_claim,
                    program.field_exprs,
                    store,
                    instance.symbol,
                    evals,
                    local_point,
                )?
            }
            relation => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_instruction_read_raf(
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    if local_point.len() < LOG_K {
        return Err(VerifyStage5Error::InvalidInputLength {
            input: "stage5.instruction_read_raf.point",
            expected: LOG_K,
            actual: local_point.len(),
        });
    }

    let (r_address_prime, r_cycle) = local_point.split_at(LOG_K);
    let r_cycle_prime = reverse_slice(r_cycle);
    let r_reduction = bolt_verifier_runtime::store_point(store, "stage5.input.stage2.instruction.LookupOutput")?;
    let eq_eval_r_reduction = EqPolynomial::<Fr>::mle(r_reduction, &r_cycle_prime);

    let left_operand_eval = operand_polynomial_eval(r_address_prime, true);
    let right_operand_eval = operand_polynomial_eval(r_address_prime, false);
    let identity_poly_eval = identity_polynomial_eval(r_address_prime);

    let table_flag_claims =
        eval_family_values(evals, &STAGE5_INSTRUCTION_READ_RAF_TABLE_FLAG_EVALS)?;
    let table_values = LookupTableKind::<XLEN>::all()
        .iter()
        .take(table_flag_claims.len())
        .map(|table| table.evaluate_mle::<Fr, Fr>(r_address_prime))
        .collect::<Vec<_>>();
    let val_claim = table_values
        .into_iter()
        .zip(table_flag_claims)
        .map(|(table_value, flag_claim)| table_value * flag_claim)
        .sum::<Fr>();

    let ra_claim = eval_family_values(
        evals,
        &STAGE5_INSTRUCTION_READ_RAF_INSTRUCTION_RA_EVALS,
    )?
    .into_iter()
    .product::<Fr>();
    let raf_flag_claim = eval_by_name(
        evals,
        "stage5.instruction_read_raf.eval.InstructionRafFlag",
    )?;
    let gamma = bolt_verifier_runtime::store_scalar(store, "stage5.instruction_read_raf.gamma")?;

    let raf_claim = (Fr::from_u64(1) - raf_flag_claim)
        * (left_operand_eval + gamma * right_operand_eval)
        + raf_flag_claim * gamma * identity_poly_eval;
    Ok(eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim))
}

"#
            }
        }
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

fn require_supported_symbol(kind: &str, actual: &str, expected: &str) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; expected @{expected}"
        )))
    }
}

fn emit_str_array(name: &str, values: &[String]) -> String {
    if values.is_empty() {
        return format!("pub const {name}: &[&str] = &[];\n\n");
    }
    if let [value] = values {
        return format!("pub const {name}: &[&str] = &[{}];\n\n", rust_str(value));
    }
    let entries = values
        .iter()
        .map(|value| format!("    {},", rust_str(value)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[&str] = &[\n{entries}\n];\n\n")
}

fn emit_usize_array(name: &str, values: &[usize]) -> String {
    let entries = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("pub const {name}: &[usize] = &[{entries}];\n\n")
}

fn intern_str_array(
    source: &mut String,
    arrays: &mut Vec<(Vec<String>, String)>,
    name_prefix: &str,
    values: &[String],
) -> String {
    if let Some((_, name)) = arrays
        .iter()
        .find(|(existing, _)| existing.as_slice() == values)
    {
        return name.clone();
    }
    let name = format!("{name_prefix}_{}", arrays.len());
    source.push_str(&emit_str_array(&name, values));
    arrays.push((values.to_vec(), name.clone()));
    name
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn rust_option_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
}

fn verify_count(kind: &str, symbol: &str, expected: usize, actual: usize) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn symbols<'a>(values: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    values.cloned().collect()
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
        .ok_or_else(|| attr_error(operation, attr, "symbol"))
}

fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&attribute).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

fn parse_symbol_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().strip_prefix('@').map(ToOwned::to_owned))
        .collect()
}

fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn int_array_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<Vec<usize>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "integer array"))?;
    parse_int_array(&attribute).ok_or_else(|| attr_error(operation, attr, "integer array"))
}

fn parse_int_array(attribute: &str) -> Option<Vec<usize>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().parse().ok())
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
