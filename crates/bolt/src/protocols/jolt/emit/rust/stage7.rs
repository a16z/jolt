#![expect(
    clippy::needless_raw_string_hashes,
    reason = "generated Rust templates are kept as raw string blocks for copyable output"
)]

use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::protocols::jolt::verifier_plan::{self, VerifierStagePlan};
use crate::protocols::jolt::verifier_relation_outputs::{
    self, parse_output_eval_family_plan, parse_output_function_family_plan,
    parse_output_product_family_plan, FieldExprDependencies,
    RelationOutputAst as Stage7RelationOutputAst,
    RelationOutputEvalFamilyPlan as Stage7RelationOutputEvalFamilyPlan,
    RelationOutputFunctionFamilyPlan as Stage7RelationOutputFunctionFamilyPlan,
    RelationOutputPlan as Stage7RelationOutputPlan,
    RelationOutputProductFamilyPlan as Stage7RelationOutputProductFamilyPlan,
    StructuredPolynomialEvalPlan as Stage7StructuredPolynomialEvalPlan,
    StructuredPolynomialPointPlan as Stage7StructuredPolynomialPointPlan,
};
use crate::protocols::jolt::verifier_values;
use crate::schema::verify_cpu_schema;

use super::plan_tokens::{
    emit_str_array, emit_usize_array, intern_str_array, require_supported_symbol, rust_option_str,
    rust_str, symbols, verify_count,
};

const STAGE7_KERNEL_ABIS: &[(&str, &str)] = &[
    (
        "jolt.stage7.hamming_weight_claim_reduction",
        "jolt_stage7_hamming_weight_claim_reduction",
    ),
    ("jolt.stage7.batched", "jolt_stage7_batched"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7CpuProgram {
    pub role: Role,
    pub(crate) verifier_plan: Option<VerifierStagePlan>,
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
    pub relation_output_values: Vec<Stage7StructuredPolynomialEvalPlan>,
    pub relation_output_eval_families: Vec<Stage7RelationOutputEvalFamilyPlan>,
    pub relation_output_product_families: Vec<Stage7RelationOutputProductFamilyPlan>,
    pub relation_output_function_families: Vec<Stage7RelationOutputFunctionFamilyPlan>,
    pub relation_outputs: Vec<Stage7RelationOutputPlan>,
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

impl FieldExprDependencies for Stage7FieldExprPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn operands(&self) -> &[String] {
        &self.operands
    }
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

verifier_plan::impl_verifier_plan_source_traits!(
    program = Stage7CpuProgram,
    step = Stage7ProgramStepPlan,
    squeeze = Stage7TranscriptSqueezePlan,
    opening_input = Stage7OpeningInputPlan,
    field_constant = Stage7FieldConstantPlan,
    field_expr = Stage7FieldExprPlan,
    claim = Stage7SumcheckClaimPlan,
    batch = Stage7SumcheckBatchPlan,
    driver = Stage7SumcheckDriverPlan,
    instance = Stage7SumcheckInstanceResultPlan,
    eval = Stage7SumcheckEvalPlan,
    point_slice = Stage7PointSlicePlan,
    point_concat = Stage7PointConcatPlan,
    opening_claim = Stage7OpeningClaimPlan,
    opening_equality = Stage7OpeningClaimEqualityPlan,
    opening_batch = Stage7OpeningBatchPlan,
    absorb = Stage7TranscriptAbsorbBytesPlan,
    point_zero = Stage7PointZeroPlan,
    relation_output_eval_families = relation_output_eval_families,
    relation_output_product_families = relation_output_product_families,
    relation_output_function_families = relation_output_function_families,
);

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
        source: program.emit_source()?,
    })
}

impl Stage7CpuProgram {
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
        let mut relation_output_values = Vec::new();
        let mut relation_output_eval_families = Vec::new();
        let mut relation_output_product_families = Vec::new();
        let mut relation_output_function_families = Vec::new();
        let mut relation_output_asts = Vec::new();
        let mut point_zeros = Vec::new();
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
                    params = Some(Stage7Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage7KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage7ProgramStepPlan {
                        kind: "transcript_squeeze".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_squeezes.push(Stage7TranscriptSqueezePlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.transcript_absorb_bytes" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage7ProgramStepPlan {
                        kind: "transcript_absorb_bytes".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_absorb_bytes.push(Stage7TranscriptAbsorbBytesPlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        payload: string_attr(op, "payload")?,
                    });
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage7OpeningInputPlan {
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
                    field_constants.push(Stage7FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.field_zero" => {
                    field_constants.push(Stage7FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 0,
                    });
                }
                "cpu.field_one" => {
                    field_constants.push(Stage7FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 1,
                    });
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage7FieldExprPlan {
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
                    field_exprs.push(Stage7FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: format!("field.pow:{exponent}"),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.sumcheck_claim" => {
                    claims.push(Stage7SumcheckClaimPlan {
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
                    claims.push(Stage7SumcheckClaimPlan {
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
                    batches.push(Stage7SumcheckBatchPlan {
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
                    steps.push(Stage7ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage7SumcheckDriverPlan {
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
                    steps.push(Stage7ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage7SumcheckDriverPlan {
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
                    instance_results.push(Stage7SumcheckInstanceResultPlan {
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
                    evals.push(Stage7SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.structured_polynomial_eval" => {
                    let symbol = string_attr(op, "sym_name")?;
                    let x_point = Stage7StructuredPolynomialPointPlan::from_cpu(
                        operand_symbol(op, 0)?,
                        string_attr(op, "x_point_segment")?,
                        string_attr(op, "x_point_length")?,
                        string_attr(op, "x_point_order")?,
                    )?;
                    let y_point = Stage7StructuredPolynomialPointPlan::from_cpu(
                        operand_symbol(op, 1)?,
                        string_attr(op, "y_point_segment")?,
                        string_attr(op, "y_point_length")?,
                        string_attr(op, "y_point_order")?,
                    )?;
                    relation_output_values.push(Stage7StructuredPolynomialEvalPlan::from_cpu(
                        symbol,
                        string_attr(op, "polynomial")?,
                        x_point,
                        y_point,
                    )?);
                }
                "cpu.sumcheck_output_eval_family" => {
                    relation_output_eval_families
                        .push(parse_output_eval_family_plan("stage7", op)?);
                }
                "cpu.sumcheck_output_product_family" => {
                    relation_output_product_families
                        .push(parse_output_product_family_plan("stage7", op)?);
                }
                "cpu.sumcheck_output_function_family" => {
                    relation_output_function_families
                        .push(parse_output_function_family_plan("stage7", op)?);
                }
                "cpu.sumcheck_output_claim" => {
                    relation_output_asts.push(Stage7RelationOutputAst {
                        relation: symbol_attr(op, "relation")?,
                        expected_output: operand_symbol(op, 0)?,
                        polynomial_evals: symbol_array_attr(op, "polynomial_evals")?,
                        polynomial_eval_operands: operand_symbols(op, 1)?,
                    });
                }
                "cpu.point_zero" => {
                    point_zeros.push(Stage7PointZeroPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        arity: int_attr(op, "arity")?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(Stage7PointSlicePlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        offset: int_attr(op, "offset")?,
                        length: int_attr(op, "length")?,
                        input: operand_symbol(op, 0)?,
                    });
                }
                "cpu.point_concat" => {
                    point_concats.push(Stage7PointConcatPlan {
                        symbol: string_attr(op, "sym_name")?,
                        layout: string_attr(op, "layout")?,
                        arity: int_attr(op, "arity")?,
                        inputs: operand_symbols(op, 0)?,
                    });
                }
                "cpu.opening_claim" => {
                    opening_claims.push(Stage7OpeningClaimPlan {
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
                    opening_equalities.push(Stage7OpeningClaimEqualityPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: string_attr(op, "mode")?,
                        lhs: operand_symbol(op, 0)?,
                        rhs: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_batch" => {
                    opening_batches.push(Stage7OpeningBatchPlan {
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
        let is_verifier = role == Role::Verifier;
        if role == Role::Prover {
            verifier_relation_outputs::prune_output_only_field_exprs(
                &mut field_exprs,
                claims.iter().map(|claim| claim.claim_value.as_str()),
                relation_output_asts
                    .iter()
                    .map(|claim| claim.expected_output.as_str()),
            );
        }
        let relation_outputs = if role == Role::Verifier {
            verifier_relation_outputs::resolve_relation_outputs(
                "stage7",
                &relation_output_values,
                &relation_output_eval_families,
                &relation_output_product_families,
                &relation_output_function_families,
                &field_exprs,
                relation_output_asts,
            )?
        } else {
            Vec::new()
        };

        let mut program = Self {
            params: params.ok_or_else(|| EmitError::new("missing cpu.params"))?,
            role,
            verifier_plan: None,
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
            relation_output_values,
            relation_output_eval_families,
            relation_output_product_families,
            relation_output_function_families,
            relation_outputs,
            point_zeros,
            point_slices,
            point_concats,
            opening_claims,
            opening_equalities,
            opening_batches,
        };
        if is_verifier {
            program.verifier_plan = Some(program.plan_verifier()?);
        }
        Ok(program)
    }

    fn plan_verifier(&self) -> Result<VerifierStagePlan, EmitError> {
        verifier_plan::stage_plan_from_cpu_sources(self)
    }

    fn verifier_plan(&self) -> Result<&VerifierStagePlan, EmitError> {
        self.verifier_plan
            .as_ref()
            .ok_or_else(|| EmitError::new("missing stage7 verifier plan"))
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
            self.verify_relation_outputs()?;
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
                    "stage7 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage7 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        for absorb in &self.transcript_absorb_bytes {
            if absorb.label.is_empty() {
                return Err(EmitError::new(format!(
                    "stage7 transcript byte absorb @{} has empty label",
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
        let field_values = if self.role == Role::Verifier {
            self.verifier_plan()?.scalar_value_sources()
        } else {
            self.cpu_field_value_sources()
        };
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

    fn cpu_field_value_sources(&self) -> verifier_values::VerifierScalarSourceSet {
        let mut values = verifier_values::VerifierScalarSourceSet::default();
        values.extend(
            self.opening_inputs.iter().map(|input| &input.symbol),
            verifier_values::VerifierScalarSourceKind::OpeningInput,
        );
        values.extend(
            self.field_constants.iter().map(|constant| &constant.symbol),
            verifier_values::VerifierScalarSourceKind::FieldConstant,
        );
        values.extend(
            self.transcript_squeezes
                .iter()
                .filter(|squeeze| matches!(squeeze.kind.as_str(), "challenge_scalar" | "scalar"))
                .map(|squeeze| &squeeze.symbol),
            verifier_values::VerifierScalarSourceKind::TranscriptScalar,
        );
        values.extend(
            self.field_exprs.iter().map(|expr| &expr.symbol),
            verifier_values::VerifierScalarSourceKind::FieldExpr,
        );
        values.extend(
            self.evals.iter().map(|eval| &eval.symbol),
            verifier_values::VerifierScalarSourceKind::SumcheckEval,
        );
        values.extend(
            self.relation_output_values
                .iter()
                .map(|value| &value.symbol),
            verifier_values::VerifierScalarSourceKind::StructuredPolynomialEval,
        );
        values.extend(
            self.relation_output_eval_families
                .iter()
                .map(|family| &family.symbol),
            verifier_values::VerifierScalarSourceKind::OutputEvalFamily,
        );
        values.extend(
            self.relation_output_product_families
                .iter()
                .map(|family| &family.symbol),
            verifier_values::VerifierScalarSourceKind::OutputProductFamily,
        );
        values.extend(
            self.relation_output_function_families
                .iter()
                .map(|family| &family.symbol),
            verifier_values::VerifierScalarSourceKind::OutputFunctionFamily,
        );
        values
    }

    fn verify_kernel_definitions(&self) -> Result<(), EmitError> {
        for kernel in &self.kernels {
            if kernel.backend != "cpu" {
                return Err(EmitError::new(format!(
                    "stage7 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage7 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = stage7_kernel_abi(&kernel.relation).ok_or_else(|| {
                EmitError::new(format!(
                    "unsupported stage7 kernel relation @{}",
                    kernel.relation
                ))
            })?;
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage7 kernel @{} ABI `{}` does not match relation @{}",
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
                "verifier stage7 program must not contain kernels",
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

    fn verify_relation_outputs(&self) -> Result<(), EmitError> {
        let relations = symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.relation),
        );
        let plan = self.verifier_plan()?;
        let field_values = plan.scalar_value_sources();
        let point_values = plan.point_value_sources();
        verifier_relation_outputs::verify_relation_outputs(
            "stage7",
            verifier_relation_outputs::RelationOutputVerification {
                relation_output_values: &self.relation_output_values,
                relation_output_eval_families: &self.relation_output_eval_families,
                relation_output_product_families: &self.relation_output_product_families,
                relation_output_function_families: &self.relation_output_function_families,
                relation_outputs: &self.relation_outputs,
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        )
    }

    fn verify_opening_flow(&self) -> Result<(), EmitError> {
        let point_sources = if self.role == Role::Verifier {
            self.verifier_plan()?.opening_point_sources()
        } else {
            let mut point_sources = symbols(self.drivers.iter().map(|driver| &driver.symbol));
            point_sources.extend(symbols(
                self.instance_results
                    .iter()
                    .map(|instance| &instance.symbol),
            ));
            point_sources.extend(symbols(
                self.opening_inputs.iter().map(|input| &input.symbol),
            ));
            point_sources.extend(symbols(self.point_zeros.iter().map(|zero| &zero.symbol)));
            point_sources.extend(symbols(self.point_slices.iter().map(|slice| &slice.symbol)));
            point_sources.extend(symbols(
                self.point_concats.iter().map(|concat| &concat.symbol),
            ));
            point_sources
        };
        for zero in &self.point_zeros {
            require_supported_symbol("point zero field", &zero.field, "bn254_fr")?;
        }
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
        let eval_sources = if self.role == Role::Verifier {
            self.verifier_plan()?.scalar_value_sources()
        } else {
            self.cpu_field_value_sources()
        };
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
            Role::Prover => "prove_stage7.rs",
            Role::Verifier => "verify_stage7.rs",
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
         use jolt_kernels::stage7::{execute_stage7_program, Stage7CpuProgramPlan, Stage7ExecutionArtifacts, Stage7ExecutionMode, Stage7FieldConstantPlan, Stage7FieldExprPlan, Stage7KernelError, Stage7KernelExecutor, Stage7KernelPlan, Stage7OpeningBatchPlan, Stage7OpeningClaimEqualityPlan, Stage7OpeningClaimPlan, Stage7OpeningInputPlan, Stage7Params, Stage7PointConcatPlan, Stage7PointSlicePlan, Stage7PointZeroPlan, Stage7ProgramStepPlan, Stage7SumcheckBatchPlan, Stage7SumcheckClaimPlan, Stage7SumcheckDriverPlan, Stage7SumcheckEvalPlan, Stage7SumcheckInstanceResultPlan, Stage7TranscriptAbsorbBytesPlan, Stage7TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage7Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use bolt_verifier_runtime::{batch_claims, find_batch, find_plan};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    #[expect(dead_code)]
    fn emit_types() -> &'static str {
        r#"#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7SumcheckClaimPlan {
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
pub struct Stage7SumcheckBatchPlan {
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
pub struct Stage7SumcheckDriverPlan {
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
pub struct Stage7SumcheckInstanceResultPlan {
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
pub struct Stage7SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7PointZeroPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub arity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage7Params,
    pub steps: &'static [Stage7ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage7TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage7TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage7OpeningInputPlan],
    pub field_constants: &'static [Stage7FieldConstantPlan],
    pub field_exprs: &'static [Stage7FieldExprPlan],
    pub kernels: &'static [Stage7KernelPlan],
    pub claims: &'static [Stage7SumcheckClaimPlan],
    pub batches: &'static [Stage7SumcheckBatchPlan],
    pub drivers: &'static [Stage7SumcheckDriverPlan],
    pub instance_results: &'static [Stage7SumcheckInstanceResultPlan],
    pub evals: &'static [Stage7SumcheckEvalPlan],
    pub point_zeros: &'static [Stage7PointZeroPlan],
    pub point_slices: &'static [Stage7PointSlicePlan],
    pub point_concats: &'static [Stage7PointConcatPlan],
    pub opening_claims: &'static [Stage7OpeningClaimPlan],
    pub opening_equalities: &'static [Stage7OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage7OpeningBatchPlan],
}
"#
    }

    fn emit_verifier_type_aliases() -> &'static str {
        r#"pub type Stage7NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage7SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage7ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage7ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage7Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage7OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage7CpuProgramPlan = bolt_verifier_runtime::StageProgramPlan<Stage7RelationKind>;
pub type Stage7SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage7RelationKind>;
pub type Stage7SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage7RelationKind>;
pub type Stage7SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage7RelationKind>;
pub type Stage7RelationOutputPlan = bolt_verifier_runtime::RelationOutputPlan<Stage7RelationKind>;
pub type Stage7StructuredPolynomialEvalPlan = bolt_verifier_runtime::StructuredPolynomialEvalPlan;

pub use super::jolt_relations::JoltRelationKind as Stage7RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage7ClaimKind, FieldConstantPlan as Stage7FieldConstantPlan,
    FieldExprKind as Stage7FieldExprKind,
    FieldExprPlan as Stage7FieldExprPlan,
    KernelPlan as Stage7KernelPlan, OpeningBatchPlan as Stage7OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage7OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage7OpeningClaimPlan, OpeningInputPlan as Stage7OpeningInputPlan,
    OpeningEqualityMode as Stage7OpeningEqualityMode,
    PointConcatPlan as Stage7PointConcatPlan, PointSlicePlan as Stage7PointSlicePlan,
    PointZeroPlan as Stage7PointZeroPlan, ProgramStepKind as Stage7ProgramStepKind,
    ProgramStepPlan as Stage7ProgramStepPlan,
    StageParams as Stage7Params,
    SumcheckBatchPlan as Stage7SumcheckBatchPlan,
    SumcheckEvalPlan as Stage7SumcheckEvalPlan,
    StructuredPolynomialKind as Stage7StructuredPolynomialKind,
    StructuredPolynomialPointLength as Stage7StructuredPolynomialPointLength,
    StructuredPolynomialPointOrder as Stage7StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan as Stage7StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment as Stage7StructuredPolynomialPointSegment,
    TranscriptAbsorbBytesPlan as Stage7TranscriptAbsorbBytesPlan,
    TranscriptSqueezeKind as Stage7TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage7TranscriptSqueezePlan,
};
"#
    }

    fn emit_verifier_types() -> String {
        let mut source = Self::emit_verifier_type_aliases().to_owned();
        source.push_str(
            r#"
pub type DefaultStage7Transcript = Blake2bTranscript<Fr>;
pub type Stage7VerifierProgramPlan = Stage7CpuProgramPlan;

#[derive(Debug)]
pub enum VerifyStage7Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage7RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage7Error);
"#,
        );
        source
    }

    fn emit_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants()?;
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants()?);
        source.push_str(&self.emit_sumcheck_driver_constants()?);
        if self.role == Role::Verifier {
            source.push_str(&self.emit_verifier_relation_output_constants()?);
        }
        source.push_str(&self.emit_tail_constants()?);
        let relation_outputs_field = if self.role == Role::Verifier {
            "    relation_output_values: STAGE7_RELATION_OUTPUT_VALUES,\n    relation_outputs: STAGE7_RELATION_OUTPUTS,\n"
        } else {
            ""
        };
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE7_PROGRAM: {} = Stage7CpuProgramPlan {{\n\
                 \x20   role: {},\n\
                 \x20   params: STAGE7_PARAMS,\n\
                 \x20   steps: STAGE7_PROGRAM_STEPS,\n\
                 \x20   transcript_squeezes: STAGE7_TRANSCRIPT_SQUEEZES,\n\
                 \x20   transcript_absorb_bytes: STAGE7_TRANSCRIPT_ABSORB_BYTES,\n\
                 \x20   opening_inputs: STAGE7_OPENING_INPUTS,\n\
                 \x20   field_constants: STAGE7_FIELD_CONSTANTS,\n\
                 \x20   field_exprs: STAGE7_FIELD_EXPRS,\n\
                 \x20   kernels: STAGE7_KERNELS,\n\
                 \x20   claims: STAGE7_SUMCHECK_CLAIMS,\n\
                 \x20   batches: STAGE7_SUMCHECK_BATCHES,\n\
                 \x20   drivers: STAGE7_SUMCHECK_DRIVERS,\n\
                 \x20   instance_results: STAGE7_SUMCHECK_INSTANCE_RESULTS,\n\
                 \x20   evals: STAGE7_SUMCHECK_EVALS,\n\
                 {relation_outputs_field}\
                 \x20   point_zeros: STAGE7_POINT_ZEROS,\n\
                 \x20   point_slices: STAGE7_POINT_SLICES,\n\
                 \x20   point_concats: STAGE7_POINT_CONCATS,\n\
                 \x20   opening_claims: STAGE7_OPENING_CLAIMS,\n\
                 \x20   opening_equalities: STAGE7_OPENING_EQUALITIES,\n\
                 \x20   opening_batches: STAGE7_OPENING_BATCHES,\n\
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
                "pub const STAGE7_PARAMS: Stage7Params = Stage7Params {{ field: {}, pcs: {}, transcript: {} }};\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript)
            ),
        );
        source.push_str(&self.emit_program_step_constants()?);
        source.push_str(&self.emit_transcript_squeeze_constants()?);
        source.push_str(&self.emit_transcript_absorb_bytes_constants()?);
        source.push_str(&self.emit_opening_input_constants()?);
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_field_expr_constants()?);
        Ok(source)
    }

    fn emit_program_step_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_program_step_constants(
                "Stage7",
                "STAGE7",
                &plan.steps,
            ));
        }
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(format!(
                    "    Stage7ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    super::plan_tokens::role_program_step_kind_expr(
                        "Stage7", &self.role, &step.kind
                    )?,
                    rust_str(&step.symbol),
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE7_PROGRAM_STEPS: &[Stage7ProgramStepPlan] = &[\n{steps}\n];\n\n"
        ))
    }

    fn emit_transcript_squeeze_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_squeeze_constants(
                "Stage7",
                "STAGE7",
                &plan.transcript_squeezes,
            ));
        }
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                Ok(format!(
                    "    Stage7TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    super::plan_tokens::role_transcript_squeeze_kind_expr("Stage7", &self.role, &squeeze.kind)?,
                    squeeze.count,
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE7_TRANSCRIPT_SQUEEZES: &[Stage7TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        ))
    }

    fn emit_transcript_absorb_bytes_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_absorb_bytes_constants(
                "Stage7",
                "STAGE7",
                &plan.transcript_absorb_bytes,
            ));
        }
        let absorbs = self
            .transcript_absorb_bytes
            .iter()
            .map(|absorb| {
                format!(
                    "    Stage7TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                    rust_str(&absorb.symbol),
                    rust_str(&absorb.label),
                    rust_str(&absorb.payload),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "pub const STAGE7_TRANSCRIPT_ABSORB_BYTES: &[Stage7TranscriptAbsorbBytesPlan] = &[\n{absorbs}\n];\n\n"
        ))
    }

    fn emit_opening_input_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_input_constants(
                "Stage7",
                "STAGE7",
                &plan.opening_inputs,
            ));
        }
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                Ok(format!(
                    "    Stage7OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage7", &self.role, &input.claim_kind)?
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE7_OPENING_INPUTS: &[Stage7OpeningInputPlan] = &[\n{inputs}\n];\n\n"
        ))
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage7FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE7_FIELD_CONSTANTS: &[Stage7FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_field_expr_constants_chunked(
                "Stage7",
                "STAGE7",
                "stage7_field_expr",
                &plan.field_exprs,
                8,
            ));
        }

        let mut source = String::new();
        let mut arrays = Vec::new();
        let mut array_refs = Vec::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            let operands = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE7_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE7_FIELD_EXPR_OPERANDS",
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
                    "    Stage7FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
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
                "pub const STAGE7_FIELD_EXPRS: &[Stage7FieldExprPlan] = &[\n{exprs}\n];\n"
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
                    "    Stage7KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE7_KERNELS: &[Stage7KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_claim_constants(
                "Stage7",
                "STAGE7",
                &plan.claims,
            ));
        }

        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                Ok(format!(
                    "    Stage7SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: STAGE7_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_option_str(claim.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage7",
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
                "pub const STAGE7_SUMCHECK_CLAIMS: &[Stage7SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_batch_constants(
                "Stage7",
                "STAGE7",
                &plan.batches,
            ));
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE7_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE7_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage7SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE7_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE7_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE7_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                "pub const STAGE7_SUMCHECK_BATCHES: &[Stage7SumcheckBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_driver_constants(
                "Stage7",
                "STAGE7",
                &plan.drivers,
            ));
        }
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE7_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                Ok(format!(
                    "    Stage7SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE7_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                    rust_str(&driver.symbol),
                    rust_str(&driver.stage),
                    rust_str(&driver.proof_slot),
                    rust_option_str(driver.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage7",
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
                "pub const STAGE7_SUMCHECK_DRIVERS: &[Stage7SumcheckDriverPlan] = &[\n{drivers}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_tail_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants()?);
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_zero_constants()?);
        source.push_str(&self.emit_point_slice_constants()?);
        source.push_str(&self.emit_point_concat_constants()?);
        source.push_str(&self.emit_opening_claim_constants()?);
        source.push_str(&self.emit_opening_claim_equality_constants()?);
        source.push_str(&self.emit_opening_batch_constants()?);
        Ok(source)
    }

    fn emit_sumcheck_instance_result_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_instance_result_constants(
                "Stage7",
                "STAGE7",
                &plan.instance_results,
            ));
        }
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                Ok(format!(
                    "    Stage7SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    super::plan_tokens::role_relation_kind_expr(
                        "Stage7",
                        &self.role,
                        &instance.relation
                    )?,
                    instance.index,
                    instance.point_arity,
                    instance.num_rounds,
                    instance.round_offset,
                    super::plan_tokens::role_sumcheck_point_order_expr(
                        &self.role,
                        &instance.point_order
                    )?,
                    instance.degree
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE7_SUMCHECK_INSTANCE_RESULTS: &[Stage7SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        ))
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let rows = self
            .evals
            .chunks(4)
            .map(|chunk| {
                let evals = chunk
                    .iter()
                    .map(|eval| {
                        format!(
                            "stage7_sumcheck_eval({}, {}, {}, {}, {})",
                            rust_str(&eval.symbol),
                            rust_str(&eval.source),
                            rust_str(&eval.name),
                            eval.index,
                            rust_str(&eval.oracle)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("    {evals},")
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "const fn stage7_sumcheck_eval(symbol: &'static str, source: &'static str, name: &'static str, index: usize, oracle: &'static str) -> Stage7SumcheckEvalPlan {{\n    Stage7SumcheckEvalPlan {{ symbol, source, name, index, oracle }}\n}}\n\n#[rustfmt::skip]\npub const STAGE7_SUMCHECK_EVALS: &[Stage7SumcheckEvalPlan] = &[\n{rows}\n];\n\n"
        )
    }

    fn emit_verifier_relation_output_constants(&self) -> Result<String, EmitError> {
        super::relation_outputs::emit_verifier_relation_output_constants(
            "Stage7",
            &self.role,
            &self.relation_output_values,
            &self.relation_outputs,
        )
    }

    fn emit_point_zero_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_zero_constants(
                "Stage7",
                "STAGE7",
                &plan.point_zeros,
            ));
        }
        let zeros = self
            .point_zeros
            .iter()
            .map(|zero| {
                format!(
                    "    Stage7PointZeroPlan {{ symbol: {}, field: {}, arity: {} }},",
                    rust_str(&zero.symbol),
                    rust_str(&zero.field),
                    zero.arity
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "pub const STAGE7_POINT_ZEROS: &[Stage7PointZeroPlan] = &[\n{zeros}\n];\n\n"
        ))
    }

    fn emit_point_slice_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_slice_constants(
                "Stage7",
                "STAGE7",
                &plan.point_slices,
            ));
        }
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage7PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
                    rust_str(&slice.symbol),
                    rust_str(&slice.source),
                    slice.offset,
                    slice.length,
                    rust_str(&slice.input)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "pub const STAGE7_POINT_SLICES: &[Stage7PointSlicePlan] = &[\n{slices}\n];\n\n"
        ))
    }

    fn emit_point_concat_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_concat_constants(
                "Stage7",
                "STAGE7",
                &plan.point_concats,
            ));
        }

        let mut source = String::new();
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage7PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE7_POINT_CONCAT_{index}_INPUTS }},",
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
                "pub const STAGE7_POINT_CONCATS: &[Stage7PointConcatPlan] = &[\n{concats}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_opening_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_constants(
                "Stage7",
                "STAGE7",
                &plan.opening_claims,
            ));
        }
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(format!(
                    "    Stage7OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage7", &self.role, &claim.claim_kind)?,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE7_OPENING_CLAIMS: &[Stage7OpeningClaimPlan] = &[\n{claims}\n];\n\n"
        ))
    }

    fn emit_opening_claim_equality_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_equality_constants(
                "Stage7",
                "STAGE7",
                &plan.opening_equalities,
            ));
        }
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                Ok(format!(
                    "    Stage7OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    super::plan_tokens::role_opening_equality_mode_expr("Stage7", &self.role, &equality.mode)?,
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE7_OPENING_EQUALITIES: &[Stage7OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        ))
    }

    fn emit_opening_batch_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_batch_constants(
                "Stage7",
                "STAGE7",
                &plan.opening_batches,
            ));
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE7_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage7OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE7_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE7_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
                "pub const STAGE7_OPENING_BATCHES: &[Stage7OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage7_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage7ExecutionArtifacts<Fr>, Stage7KernelError>\n\
                 where\n\
                 \x20   E: Stage7KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage7_prover_with_program(&STAGE7_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage7_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage7CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage7ExecutionArtifacts<Fr>, Stage7KernelError>\n\
                 where\n\
                 \x20   E: Stage7KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage7_program(program, Stage7ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage7<T>(
    proof: &Stage7Proof<Fr>,
    opening_inputs: &[Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage7_with_program(&STAGE7_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage7_with_program<T>(
    program: &'static Stage7VerifierProgramPlan,
    proof: &Stage7Proof<Fr>,
    opening_inputs: &[Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage7Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    store.seed_point_zeros(program.point_zeros);
    let mut artifacts = Stage7ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage7ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage7Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage7_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage7ProgramStepKind::TranscriptAbsorbBytes => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage7Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage7_bytes(absorb, transcript);
            }
            Stage7ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage7Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage7_driver(
                    program,
                    driver,
                    proof,
                    &mut store,
                    transcript,
                    &mut artifacts,
                )?;
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage7_verifier_program() -> &'static Stage7VerifierProgramPlan {
    &STAGE7_PROGRAM
}

fn verify_stage7_squeeze<T>(
    program: &'static Stage7VerifierProgramPlan,
    squeeze: &'static Stage7TranscriptSqueezePlan,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage7ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage7Error::from)?;
    artifacts.challenge_vectors.push(Stage7ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage7_bytes<T>(absorb: &'static Stage7TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage7_driver<T>(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    proof: &Stage7Proof<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage7ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage7Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage7Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage7RelationKind::Stage7Batched => {
            verify_batched_stage7(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage7Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage7<T>(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    proof: &Stage7SumcheckOutput<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage7Error::InvalidInputLength {
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
        |store, verified| observe_stage7_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage7Error::Sumcheck { driver, error },
    )
}

fn observe_stage7_sumcheck_output<F: Field>(
    program: &'static Stage7VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage7SumcheckOutput<F>,
) -> Result<(), VerifyStage7Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                _ => {
                    return Err(VerifyStage7Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage7Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage7Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage7Error::InvalidProof { driver, reason },
        |symbol| VerifyStage7Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage7Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let Some(relation) = claim.relation else {
            return Err(VerifyStage7Error::InvalidProof {
                driver: driver.symbol,
                reason: "missing claim relation",
            });
        };
        let value = match relation {
            Stage7RelationKind::Stage7HammingWeightClaimReduction => {
                bolt_verifier_runtime::evaluate_relation_output_for_instance(
                    program.relation_outputs,
        program.relation_output_values,
                    program.field_exprs,
                    store,
                    instance,
                    evals, &[], &[], local_point,
                )?
            }
            relation => return Err(VerifyStage7Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
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
            Role::Prover => "Stage7CpuProgramPlan",
            Role::Verifier => "Stage7VerifierProgramPlan",
        }
    }
}

fn stage7_kernel_abi(relation: &str) -> Option<&'static str> {
    STAGE7_KERNEL_ABIS
        .iter()
        .find_map(|(candidate, abi)| (*candidate == relation).then_some(*abi))
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

#[cfg(test)]
mod tests {
    use super::{stage7_kernel_abi, STAGE7_KERNEL_ABIS};

    #[test]
    fn stage7_kernel_abi_contracts_cover_supported_relations() {
        assert_eq!(STAGE7_KERNEL_ABIS.len(), 2);
        assert_eq!(
            stage7_kernel_abi("jolt.stage7.hamming_weight_claim_reduction"),
            Some("jolt_stage7_hamming_weight_claim_reduction")
        );
        assert_eq!(
            stage7_kernel_abi("jolt.stage7.batched"),
            Some("jolt_stage7_batched")
        );
        assert_eq!(stage7_kernel_abi("jolt.stage6.batched"), None);
    }
}
