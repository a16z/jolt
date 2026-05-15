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
use crate::protocols::jolt::stage6_bytecode_read_raf_plan::{
    emit_stage6_bytecode_read_raf_plan_constants, stage6_bytecode_read_raf_output_claim_plan,
    STAGE6_BYTECODE_RA_EVAL_FAMILY,
};
use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;
use crate::protocols::jolt::verifier_output_claims::{
    self, parse_output_eval_family_plan, parse_output_function_family_plan,
    parse_output_product_family_plan, FieldExprDependencies,
    StructuredPolynomialEvalPlan as Stage6StructuredPolynomialEvalPlan,
    StructuredPolynomialPointPlan as Stage6StructuredPolynomialPointPlan,
    SumcheckOutputClaimAst as Stage6SumcheckOutputClaimAst,
    SumcheckOutputClaimPlan as Stage6SumcheckOutputClaimPlan,
    SumcheckOutputEvalFamilyPlan as Stage6SumcheckOutputEvalFamilyPlan,
    SumcheckOutputFunctionFamilyPlan as Stage6SumcheckOutputFunctionFamilyPlan,
    SumcheckOutputProductFamilyPlan as Stage6SumcheckOutputProductFamilyPlan,
};
use crate::protocols::jolt::verifier_plan::{self, VerifierStagePlan};
use crate::protocols::jolt::verifier_values;
use crate::schema::verify_cpu_schema;

use super::plan_tokens::{
    emit_str_array, emit_usize_array, intern_str_array, require_supported_symbol, rust_option_str,
    rust_str, symbols, verify_count,
};

const STAGE6_KERNEL_ABIS: &[(&str, &str)] = &[
    (
        "jolt.stage6.bytecode_read_raf",
        "jolt_stage6_bytecode_read_raf",
    ),
    ("jolt.stage6.booleanity", "jolt_stage6_booleanity"),
    (
        "jolt.stage6.hamming_booleanity",
        "jolt_stage6_hamming_booleanity",
    ),
    ("jolt.stage6.ram_ra_virtual", "jolt_stage6_ram_ra_virtual"),
    (
        "jolt.stage6.instruction_ra_virtual",
        "jolt_stage6_instruction_ra_virtual",
    ),
    (
        "jolt.stage6.inc_claim_reduction",
        "jolt_stage6_inc_claim_reduction",
    ),
    ("jolt.stage6.batched", "jolt_stage6_batched"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6CpuProgram {
    pub role: Role,
    pub(crate) verifier_plan: Option<VerifierStagePlan>,
    pub(crate) indexed_eval_families: Vec<IndexedEvalFamilyPlan>,
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
    pub output_values: Vec<Stage6StructuredPolynomialEvalPlan>,
    pub output_families: Vec<Stage6SumcheckOutputEvalFamilyPlan>,
    pub output_product_families: Vec<Stage6SumcheckOutputProductFamilyPlan>,
    pub output_function_families: Vec<Stage6SumcheckOutputFunctionFamilyPlan>,
    pub output_claims: Vec<Stage6SumcheckOutputClaimPlan>,
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

impl FieldExprDependencies for Stage6FieldExprPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn operands(&self) -> &[String] {
        &self.operands
    }
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

verifier_plan::impl_verifier_plan_source_traits!(
    program = Stage6CpuProgram,
    step = Stage6ProgramStepPlan,
    squeeze = Stage6TranscriptSqueezePlan,
    opening_input = Stage6OpeningInputPlan,
    field_expr = Stage6FieldExprPlan,
    claim = Stage6SumcheckClaimPlan,
    batch = Stage6SumcheckBatchPlan,
    driver = Stage6SumcheckDriverPlan,
    instance = Stage6SumcheckInstanceResultPlan,
    point_slice = Stage6PointSlicePlan,
    point_concat = Stage6PointConcatPlan,
    opening_claim = Stage6OpeningClaimPlan,
    opening_equality = Stage6OpeningClaimEqualityPlan,
    opening_batch = Stage6OpeningBatchPlan,
    absorb = Stage6TranscriptAbsorbBytesPlan,
    point_zero = Stage6PointZeroPlan,
    indexed_eval_families = indexed_eval_families,
);

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
        source: program.emit_source()?,
    })
}

impl Stage6CpuProgram {
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
        let mut indexed_eval_families = Vec::new();
        let mut output_values = Vec::new();
        let mut output_families = Vec::new();
        let mut output_product_families = Vec::new();
        let mut output_function_families = Vec::new();
        let mut output_claim_asts = Vec::new();
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
                    params = Some(Stage6Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage6KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage6ProgramStepPlan {
                        kind: "transcript_squeeze".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_squeezes.push(Stage6TranscriptSqueezePlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.transcript_absorb_bytes" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage6ProgramStepPlan {
                        kind: "transcript_absorb_bytes".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_absorb_bytes.push(Stage6TranscriptAbsorbBytesPlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        payload: string_attr(op, "payload")?,
                    });
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage6OpeningInputPlan {
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
                    field_constants.push(Stage6FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.field_zero" => {
                    field_constants.push(Stage6FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 0,
                    });
                }
                "cpu.field_one" => {
                    field_constants.push(Stage6FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 1,
                    });
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage6FieldExprPlan {
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
                    field_exprs.push(Stage6FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: format!("field.pow:{exponent}"),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.sumcheck_claim" => {
                    claims.push(Stage6SumcheckClaimPlan {
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
                    claims.push(Stage6SumcheckClaimPlan {
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
                    batches.push(Stage6SumcheckBatchPlan {
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
                    steps.push(Stage6ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage6SumcheckDriverPlan {
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
                    steps.push(Stage6ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage6SumcheckDriverPlan {
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
                    instance_results.push(Stage6SumcheckInstanceResultPlan {
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
                    evals.push(Stage6SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.sumcheck_eval_family" => {
                    indexed_eval_families.push(parse_indexed_eval_family(op)?);
                }
                "cpu.structured_polynomial_eval" => {
                    let symbol = string_attr(op, "sym_name")?;
                    let x_point = Stage6StructuredPolynomialPointPlan::from_cpu(
                        operand_symbol(op, 0)?,
                        string_attr(op, "x_point_segment")?,
                        string_attr(op, "x_point_length")?,
                        string_attr(op, "x_point_order")?,
                    )?;
                    let y_point = Stage6StructuredPolynomialPointPlan::from_cpu(
                        operand_symbol(op, 1)?,
                        string_attr(op, "y_point_segment")?,
                        string_attr(op, "y_point_length")?,
                        string_attr(op, "y_point_order")?,
                    )?;
                    output_values.push(Stage6StructuredPolynomialEvalPlan::from_cpu(
                        symbol,
                        string_attr(op, "polynomial")?,
                        x_point,
                        y_point,
                    )?);
                }
                "cpu.sumcheck_output_eval_family" => {
                    output_families.push(parse_output_eval_family_plan("stage6", op)?);
                }
                "cpu.sumcheck_output_product_family" => {
                    output_product_families.push(parse_output_product_family_plan("stage6", op)?);
                }
                "cpu.sumcheck_output_function_family" => {
                    output_function_families.push(parse_output_function_family_plan("stage6", op)?);
                }
                "cpu.sumcheck_output_claim" => {
                    output_claim_asts.push(Stage6SumcheckOutputClaimAst {
                        relation: symbol_attr(op, "relation")?,
                        claim_value: operand_symbol(op, 0)?,
                        polynomial_evals: symbol_array_attr(op, "polynomial_evals")?,
                        polynomial_eval_operands: operand_symbols(op, 1)?,
                    });
                }
                "cpu.point_zero" => {
                    point_zeros.push(Stage6PointZeroPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        arity: int_attr(op, "arity")?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(Stage6PointSlicePlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        offset: int_attr(op, "offset")?,
                        length: int_attr(op, "length")?,
                        input: operand_symbol(op, 0)?,
                    });
                }
                "cpu.point_concat" => {
                    point_concats.push(Stage6PointConcatPlan {
                        symbol: string_attr(op, "sym_name")?,
                        layout: string_attr(op, "layout")?,
                        arity: int_attr(op, "arity")?,
                        inputs: operand_symbols(op, 0)?,
                    });
                }
                "cpu.opening_claim" => {
                    opening_claims.push(Stage6OpeningClaimPlan {
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
                    opening_equalities.push(Stage6OpeningClaimEqualityPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: string_attr(op, "mode")?,
                        lhs: operand_symbol(op, 0)?,
                        rhs: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_batch" => {
                    opening_batches.push(Stage6OpeningBatchPlan {
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
            verifier_output_claims::prune_output_only_field_exprs(
                &mut field_exprs,
                claims.iter().map(|claim| claim.claim_value.as_str()),
                output_claim_asts
                    .iter()
                    .map(|claim| claim.claim_value.as_str()),
            );
        }
        let mut output_claims = if role == Role::Verifier {
            verifier_output_claims::resolve_output_claims(
                "stage6",
                &output_values,
                &output_families,
                &output_product_families,
                &output_function_families,
                &field_exprs,
                output_claim_asts,
            )?
        } else {
            Vec::new()
        };
        if role == Role::Verifier {
            let bytecode_ra_evals = stage6_bytecode_read_raf_eval_family(&indexed_eval_families)?;
            output_claims.push(stage6_bytecode_read_raf_output_claim_plan(
                bytecode_ra_evals,
            ));
        }

        let mut program = Self {
            params: params.ok_or_else(|| EmitError::new("missing cpu.params"))?,
            role,
            verifier_plan: None,
            indexed_eval_families,
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
            output_families,
            output_product_families,
            output_function_families,
            output_claims,
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
            .ok_or_else(|| EmitError::new("missing stage6 verifier plan"))
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
                    "stage6 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage6 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        for absorb in &self.transcript_absorb_bytes {
            if absorb.label.is_empty() {
                return Err(EmitError::new(format!(
                    "stage6 transcript byte absorb @{} has empty label",
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

    fn field_value_symbols(&self) -> verifier_values::VerifierScalarSourceSet {
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
            self.output_values.iter().map(|value| &value.symbol),
            verifier_values::VerifierScalarSourceKind::StructuredPolynomialEval,
        );
        values.extend(
            self.output_claims
                .iter()
                .flat_map(|claim| claim.polynomial_evals.iter().map(|value| &value.symbol)),
            verifier_values::VerifierScalarSourceKind::StructuredPolynomialEval,
        );
        values.extend(
            self.output_families.iter().map(|family| &family.symbol),
            verifier_values::VerifierScalarSourceKind::OutputEvalFamily,
        );
        values.extend(
            self.output_product_families
                .iter()
                .map(|family| &family.symbol),
            verifier_values::VerifierScalarSourceKind::OutputProductFamily,
        );
        values.extend(
            self.output_function_families
                .iter()
                .map(|family| &family.symbol),
            verifier_values::VerifierScalarSourceKind::OutputFunctionFamily,
        );
        values.extend(
            self.output_claims
                .iter()
                .flat_map(|claim| claim.eval_families.iter().map(|family| &family.symbol)),
            verifier_values::VerifierScalarSourceKind::OutputEvalFamily,
        );
        values.extend(
            self.output_claims
                .iter()
                .flat_map(|claim| claim.product_families.iter().map(|family| &family.symbol)),
            verifier_values::VerifierScalarSourceKind::OutputProductFamily,
        );
        values.extend(
            self.output_claims
                .iter()
                .flat_map(|claim| claim.function_families.iter().map(|family| &family.symbol)),
            verifier_values::VerifierScalarSourceKind::OutputFunctionFamily,
        );
        values.extend(
            self.field_exprs.iter().map(|expr| &expr.symbol),
            verifier_values::VerifierScalarSourceKind::FieldExpr,
        );
        values.extend(
            self.output_claims
                .iter()
                .flat_map(|claim| claim.local_scalars.iter()),
            verifier_values::VerifierScalarSourceKind::PointDerived,
        );
        values.extend(
            self.evals.iter().map(|eval| &eval.symbol),
            verifier_values::VerifierScalarSourceKind::SumcheckEval,
        );
        values
    }

    fn point_value_symbols(&self) -> verifier_values::VerifierPointSourceSet {
        let mut values = verifier_values::VerifierPointSourceSet::default();
        values.extend(
            self.instance_results
                .iter()
                .map(|instance| &instance.symbol),
            verifier_values::VerifierPointSourceKind::SumcheckInstance,
        );
        values.extend(
            self.opening_inputs.iter().map(|input| &input.symbol),
            verifier_values::VerifierPointSourceKind::OpeningInput,
        );
        values.extend(
            self.point_zeros.iter().map(|zero| &zero.symbol),
            verifier_values::VerifierPointSourceKind::PointZero,
        );
        values.extend(
            self.point_slices.iter().map(|slice| &slice.symbol),
            verifier_values::VerifierPointSourceKind::PointSlice,
        );
        values.extend(
            self.point_concats.iter().map(|concat| &concat.symbol),
            verifier_values::VerifierPointSourceKind::PointConcat,
        );
        values
    }

    fn verify_kernel_definitions(&self) -> Result<(), EmitError> {
        for kernel in &self.kernels {
            if kernel.backend != "cpu" {
                return Err(EmitError::new(format!(
                    "stage6 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage6 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = stage6_kernel_abi(&kernel.relation).ok_or_else(|| {
                EmitError::new(format!(
                    "unsupported stage6 kernel relation @{}",
                    kernel.relation
                ))
            })?;
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage6 kernel @{} ABI `{}` does not match relation @{}",
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
                "verifier stage6 program must not contain kernels",
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
        verifier_output_claims::verify_output_claims(
            "stage6",
            verifier_output_claims::OutputClaimVerification {
                output_values: &self.output_values,
                output_families: &self.output_families,
                output_product_families: &self.output_product_families,
                output_function_families: &self.output_function_families,
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
        point_sources.extend(symbols(self.point_zeros.iter().map(|zero| &zero.symbol)));
        point_sources.extend(symbols(self.point_slices.iter().map(|slice| &slice.symbol)));
        point_sources.extend(symbols(
            self.point_concats.iter().map(|concat| &concat.symbol),
        ));
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
            Role::Prover => "prove_stage6.rs",
            Role::Verifier => "verify_stage6.rs",
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
         use jolt_kernels::stage6::{execute_stage6_program, Stage6CpuProgramPlan, Stage6ExecutionArtifacts, Stage6ExecutionMode, Stage6FieldConstantPlan, Stage6FieldExprPlan, Stage6KernelError, Stage6KernelExecutor, Stage6KernelPlan, Stage6OpeningBatchPlan, Stage6OpeningClaimEqualityPlan, Stage6OpeningClaimPlan, Stage6OpeningInputPlan, Stage6Params, Stage6PointConcatPlan, Stage6PointSlicePlan, Stage6PointZeroPlan, Stage6ProgramStepPlan, Stage6SumcheckBatchPlan, Stage6SumcheckClaimPlan, Stage6SumcheckDriverPlan, Stage6SumcheckEvalPlan, Stage6SumcheckInstanceResultPlan, Stage6TranscriptAbsorbBytesPlan, Stage6TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use bolt_verifier_runtime::{batch_claims, find_batch, find_plan};\n\
         use super::jolt_relations::{evaluate_stage67_bytecode_read_raf_output_scalars, normalize_bytecode_read_raf_point, stage67_trace_rounds, Stage67BytecodeEntry, Stage67BytecodeFlag, Stage67BytecodeOutputTermPlan, Stage67BytecodeReadRafPlan, Stage67BytecodeRegister, Stage67BytecodeRegisterSymbols, Stage67BytecodeStagePlan, Stage67BytecodeTermPlan, Stage67RelationSymbols};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    #[expect(dead_code)]
    fn emit_types() -> &'static str {
        r#"#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckClaimPlan {
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
pub struct Stage6SumcheckBatchPlan {
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
pub struct Stage6SumcheckDriverPlan {
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
pub struct Stage6SumcheckInstanceResultPlan {
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
pub struct Stage6SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointZeroPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub arity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage6Params,
    pub steps: &'static [Stage6ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage6TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage6TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage6OpeningInputPlan],
    pub field_constants: &'static [Stage6FieldConstantPlan],
    pub field_exprs: &'static [Stage6FieldExprPlan],
    pub kernels: &'static [Stage6KernelPlan],
    pub claims: &'static [Stage6SumcheckClaimPlan],
    pub batches: &'static [Stage6SumcheckBatchPlan],
    pub drivers: &'static [Stage6SumcheckDriverPlan],
    pub instance_results: &'static [Stage6SumcheckInstanceResultPlan],
    pub evals: &'static [Stage6SumcheckEvalPlan],
    pub point_zeros: &'static [Stage6PointZeroPlan],
    pub point_slices: &'static [Stage6PointSlicePlan],
    pub point_concats: &'static [Stage6PointConcatPlan],
    pub opening_claims: &'static [Stage6OpeningClaimPlan],
    pub opening_equalities: &'static [Stage6OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage6OpeningBatchPlan],
}
"#
    }

    fn emit_verifier_type_aliases() -> &'static str {
        r#"pub type Stage6NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage6SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage6ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage6ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage6Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage6OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage6CpuProgramPlan = bolt_verifier_runtime::StageProgramPlan<Stage6RelationKind>;
pub type Stage6SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage6RelationKind>;
pub type Stage6SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage6RelationKind>;
pub type Stage6SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage6RelationKind>;
pub type Stage6SumcheckOutputClaimPlan = bolt_verifier_runtime::SumcheckOutputClaimPlan<Stage6RelationKind>;
pub type Stage6StructuredPolynomialEvalPlan = bolt_verifier_runtime::StructuredPolynomialEvalPlan;

pub use super::jolt_relations::JoltRelationKind as Stage6RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage6ClaimKind, FieldConstantPlan as Stage6FieldConstantPlan,
    FieldExprKind as Stage6FieldExprKind,
    FieldExprPlan as Stage6FieldExprPlan,
    KernelPlan as Stage6KernelPlan, OpeningBatchPlan as Stage6OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage6OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage6OpeningClaimPlan, OpeningInputPlan as Stage6OpeningInputPlan,
    OpeningEqualityMode as Stage6OpeningEqualityMode,
    PointConcatPlan as Stage6PointConcatPlan, PointSlicePlan as Stage6PointSlicePlan,
    PointZeroPlan as Stage6PointZeroPlan, ProgramStepKind as Stage6ProgramStepKind,
    ProgramStepPlan as Stage6ProgramStepPlan,
    StageParams as Stage6Params,
    SumcheckBatchPlan as Stage6SumcheckBatchPlan,
    SumcheckEvalPlan as Stage6SumcheckEvalPlan,
    StructuredPolynomialPointLength as Stage6StructuredPolynomialPointLength,
    StructuredPolynomialPointOrder as Stage6StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan as Stage6StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment as Stage6StructuredPolynomialPointSegment,
    StructuredPolynomialKind as Stage6StructuredPolynomialKind,
    TranscriptAbsorbBytesPlan as Stage6TranscriptAbsorbBytesPlan,
    TranscriptSqueezeKind as Stage6TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage6TranscriptSqueezePlan,
};
"#
    }

    fn emit_verifier_types() -> String {
        let mut source = Self::emit_verifier_type_aliases().to_owned();
        source.push_str(
            r#"
pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;
pub type Stage6VerifierProgramPlan = Stage6CpuProgramPlan;

#[derive(Clone, Debug)]
pub struct Stage6BytecodeEntry {
    pub address: Fr,
    pub imm: Fr,
    pub circuit_flags: [bool; 14],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

impl Stage67BytecodeEntry for Stage6BytecodeEntry {
    fn address(&self) -> Fr { self.address }
    fn imm(&self) -> Fr { self.imm }
    fn circuit_flags(&self) -> &[bool; 14] { &self.circuit_flags }
    fn rd(&self) -> Option<usize> { self.rd }
    fn rs1(&self) -> Option<usize> { self.rs1 }
    fn rs2(&self) -> Option<usize> { self.rs2 }
    fn lookup_table(&self) -> Option<usize> { self.lookup_table }
    fn is_interleaved(&self) -> bool { self.is_interleaved }
    fn is_branch(&self) -> bool { self.is_branch }
    fn left_is_rs1(&self) -> bool { self.left_is_rs1 }
    fn left_is_pc(&self) -> bool { self.left_is_pc }
    fn right_is_rs2(&self) -> bool { self.right_is_rs2 }
    fn right_is_imm(&self) -> bool { self.right_is_imm }
    fn is_noop(&self) -> bool { self.is_noop }
}


#[derive(Clone, Debug)]
pub struct Stage6BytecodeReadRafData {
    pub entries: Vec<Stage6BytecodeEntry>,
    pub entry_bytecode_index: usize,
    pub num_lookup_tables: usize,
}

#[derive(Clone, Debug)]
pub struct Stage6VerifierData {
    pub bytecode_read_raf: Option<Stage6BytecodeReadRafData>,
}

const STAGE6_RELATION_SYMBOLS: Stage67RelationSymbols = Stage67RelationSymbols {
    hamming_booleanity_instance: "stage6.hamming_booleanity.instance",
};
"#,
        );
        source.push_str(
            r#"
#[derive(Debug)]
pub enum VerifyStage6Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage6RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage6Error);
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
        source.push_str(&self.emit_tail_constants()?);
        if self.role == Role::Verifier {
            let bytecode_ra_evals =
                stage6_bytecode_read_raf_eval_family(&self.verifier_plan()?.indexed_eval_families)?;
            source.push_str(&emit_stage6_bytecode_read_raf_plan_constants(
                bytecode_ra_evals,
            ));
            source.push_str(&self.emit_verifier_output_claim_constants()?);
        }
        let output_claims_field = if self.role == Role::Verifier {
            "    output_claims: STAGE6_SUMCHECK_OUTPUT_CLAIMS,\n"
        } else {
            ""
        };
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE6_PROGRAM: {} = Stage6CpuProgramPlan {{\n\
                 \x20   role: {},\n\
                 \x20   params: STAGE6_PARAMS,\n\
                 \x20   steps: STAGE6_PROGRAM_STEPS,\n\
                 \x20   transcript_squeezes: STAGE6_TRANSCRIPT_SQUEEZES,\n\
                 \x20   transcript_absorb_bytes: STAGE6_TRANSCRIPT_ABSORB_BYTES,\n\
                 \x20   opening_inputs: STAGE6_OPENING_INPUTS,\n\
                 \x20   field_constants: STAGE6_FIELD_CONSTANTS,\n\
                 \x20   field_exprs: STAGE6_FIELD_EXPRS,\n\
                 \x20   kernels: STAGE6_KERNELS,\n\
                 \x20   claims: STAGE6_SUMCHECK_CLAIMS,\n\
                 \x20   batches: STAGE6_SUMCHECK_BATCHES,\n\
                 \x20   drivers: STAGE6_SUMCHECK_DRIVERS,\n\
                 \x20   instance_results: STAGE6_SUMCHECK_INSTANCE_RESULTS,\n\
                 \x20   evals: STAGE6_SUMCHECK_EVALS,\n\
                 {output_claims_field}\
                 \x20   point_zeros: STAGE6_POINT_ZEROS,\n\
                 \x20   point_slices: STAGE6_POINT_SLICES,\n\
                 \x20   point_concats: STAGE6_POINT_CONCATS,\n\
                 \x20   opening_claims: STAGE6_OPENING_CLAIMS,\n\
                 \x20   opening_equalities: STAGE6_OPENING_EQUALITIES,\n\
                 \x20   opening_batches: STAGE6_OPENING_BATCHES,\n\
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
                "pub const STAGE6_PARAMS: Stage6Params = Stage6Params {{ field: {}, pcs: {}, transcript: {} }};\n",
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
                "Stage6",
                "STAGE6",
                &plan.steps,
            ));
        }
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(format!(
                    "    Stage6ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    super::plan_tokens::role_program_step_kind_expr(
                        "Stage6", &self.role, &step.kind
                    )?,
                    rust_str(&step.symbol),
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE6_PROGRAM_STEPS: &[Stage6ProgramStepPlan] = &[\n{steps}\n];\n\n"
        ))
    }

    fn emit_transcript_squeeze_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_squeeze_constants(
                "Stage6",
                "STAGE6",
                &plan.transcript_squeezes,
            ));
        }
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                Ok(format!(
                    "    Stage6TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    super::plan_tokens::role_transcript_squeeze_kind_expr("Stage6", &self.role, &squeeze.kind)?,
                    squeeze.count,
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE6_TRANSCRIPT_SQUEEZES: &[Stage6TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        ))
    }

    fn emit_transcript_absorb_bytes_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_absorb_bytes_constants(
                "Stage6",
                "STAGE6",
                &plan.transcript_absorb_bytes,
            ));
        }
        let absorbs = self
            .transcript_absorb_bytes
            .iter()
            .map(|absorb| {
                format!(
                    "    Stage6TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                    rust_str(&absorb.symbol),
                    rust_str(&absorb.label),
                    rust_str(&absorb.payload),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "pub const STAGE6_TRANSCRIPT_ABSORB_BYTES: &[Stage6TranscriptAbsorbBytesPlan] = &[\n{absorbs}\n];\n\n"
        ))
    }

    fn emit_opening_input_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_input_constants(
                "Stage6",
                "STAGE6",
                &plan.opening_inputs,
            ));
        }
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                Ok(format!(
                    "    Stage6OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage6", &self.role, &input.claim_kind)?
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE6_OPENING_INPUTS: &[Stage6OpeningInputPlan] = &[\n{inputs}\n];\n\n"
        ))
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage6FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE6_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_field_expr_constants_chunked(
                "Stage6",
                "STAGE6",
                "stage6_field_expr",
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
                "STAGE6_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE6_FIELD_EXPR_OPERANDS",
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
                    "    Stage6FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
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
                "pub const STAGE6_FIELD_EXPRS: &[Stage6FieldExprPlan] = &[\n{exprs}\n];\n"
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
                    "    Stage6KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE6_KERNELS: &[Stage6KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_claim_constants(
                "Stage6",
                "STAGE6",
                &plan.claims,
            ));
        }

        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                Ok(format!(
                    "    Stage6SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: STAGE6_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_option_str(claim.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage6",
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
                "pub const STAGE6_SUMCHECK_CLAIMS: &[Stage6SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_batch_constants(
                "Stage6",
                "STAGE6",
                &plan.batches,
            ));
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE6_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE6_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage6SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE6_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE6_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE6_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                "pub const STAGE6_SUMCHECK_BATCHES: &[Stage6SumcheckBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_driver_constants(
                "Stage6",
                "STAGE6",
                &plan.drivers,
            ));
        }
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE6_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                Ok(format!(
                    "    Stage6SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE6_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                    rust_str(&driver.symbol),
                    rust_str(&driver.stage),
                    rust_str(&driver.proof_slot),
                    rust_option_str(driver.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage6",
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
                "pub const STAGE6_SUMCHECK_DRIVERS: &[Stage6SumcheckDriverPlan] = &[\n{drivers}\n];\n"
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
                "Stage6",
                "STAGE6",
                &plan.instance_results,
            ));
        }
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                Ok(format!(
                    "    Stage6SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    super::plan_tokens::role_relation_kind_expr(
                        "Stage6",
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
            "pub const STAGE6_SUMCHECK_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
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
                            "stage6_sumcheck_eval({}, {}, {}, {}, {})",
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
            "const fn stage6_sumcheck_eval(symbol: &'static str, source: &'static str, name: &'static str, index: usize, oracle: &'static str) -> Stage6SumcheckEvalPlan {{\n    Stage6SumcheckEvalPlan {{ symbol, source, name, index, oracle }}\n}}\n\n#[rustfmt::skip]\npub const STAGE6_SUMCHECK_EVALS: &[Stage6SumcheckEvalPlan] = &[\n{rows}\n];\n\n"
        )
    }

    fn emit_verifier_output_claim_constants(&self) -> Result<String, EmitError> {
        super::output_claims::emit_verifier_output_claim_constants(
            "Stage6",
            &self.role,
            &self.output_claims,
        )
    }

    fn emit_point_zero_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_zero_constants(
                "Stage6",
                "STAGE6",
                &plan.point_zeros,
            ));
        }
        let zeros = self
            .point_zeros
            .iter()
            .map(|zero| {
                format!(
                    "    Stage6PointZeroPlan {{ symbol: {}, field: {}, arity: {} }},",
                    rust_str(&zero.symbol),
                    rust_str(&zero.field),
                    zero.arity
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "pub const STAGE6_POINT_ZEROS: &[Stage6PointZeroPlan] = &[\n{zeros}\n];\n\n"
        ))
    }

    fn emit_point_slice_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_slice_constants(
                "Stage6",
                "STAGE6",
                &plan.point_slices,
            ));
        }
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage6PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
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
            "pub const STAGE6_POINT_SLICES: &[Stage6PointSlicePlan] = &[\n{slices}\n];\n\n"
        ))
    }

    fn emit_point_concat_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_concat_constants(
                "Stage6",
                "STAGE6",
                &plan.point_concats,
            ));
        }

        let mut source = String::new();
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage6PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE6_POINT_CONCAT_{index}_INPUTS }},",
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
                "pub const STAGE6_POINT_CONCATS: &[Stage6PointConcatPlan] = &[\n{concats}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_opening_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_constants(
                "Stage6",
                "STAGE6",
                &plan.opening_claims,
            ));
        }
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(format!(
                    "    Stage6OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage6", &self.role, &claim.claim_kind)?,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE6_OPENING_CLAIMS: &[Stage6OpeningClaimPlan] = &[\n{claims}\n];\n\n"
        ))
    }

    fn emit_opening_claim_equality_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_equality_constants(
                "Stage6",
                "STAGE6",
                &plan.opening_equalities,
            ));
        }
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                Ok(format!(
                    "    Stage6OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    super::plan_tokens::role_opening_equality_mode_expr("Stage6", &self.role, &equality.mode)?,
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE6_OPENING_EQUALITIES: &[Stage6OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        ))
    }

    fn emit_opening_batch_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_batch_constants(
                "Stage6",
                "STAGE6",
                &plan.opening_batches,
            ));
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE6_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage6OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE6_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE6_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
                "pub const STAGE6_OPENING_BATCHES: &[Stage6OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage6_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>\n\
                 where\n\
                 \x20   E: Stage6KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage6_prover_with_program(&STAGE6_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage6_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage6CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>\n\
                 where\n\
                 \x20   E: Stage6KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage6_program(program, Stage6ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage6<T>(
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage6_with_program(&STAGE6_PROGRAM, proof, opening_inputs, verifier_data, transcript)
}

pub fn verify_stage6_with_program<T>(
    program: &'static Stage6VerifierProgramPlan,
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage6Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    store.seed_point_zeros(program.point_zeros);
    let mut artifacts = Stage6ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage6ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage6_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage6ProgramStepKind::TranscriptAbsorbBytes => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage6_bytes(absorb, transcript);
            }
            Stage6ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage6Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage6_driver(
                    program,
                    driver,
                    proof,
                    verifier_data,
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

pub fn stage6_verifier_program() -> &'static Stage6VerifierProgramPlan {
    &STAGE6_PROGRAM
}

fn verify_stage6_squeeze<T>(
    program: &'static Stage6VerifierProgramPlan,
    squeeze: &'static Stage6TranscriptSqueezePlan,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = if squeeze.symbol == "stage6.booleanity.gamma" {
        transcript.challenge_vector_optimized(squeeze.count)
    } else {
        transcript.challenge_vector(squeeze.count)
    };
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage6Error::from)?;
    artifacts.challenge_vectors.push(Stage6ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage6_bytes<T>(absorb: &'static Stage6TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage6_driver<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6Proof<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage6Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage6Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage6RelationKind::Stage6Batched => {
            verify_batched_stage6(program, driver, proof, verifier_data, store, transcript)?
        }
        relation => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage6<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6SumcheckOutput<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
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
            expected_batched_output_claim(
                program,
                driver,
                verifier_data,
                store,
                evals,
                point,
                batching_coeffs,
            )
        },
        |store, verified| observe_stage6_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage6Error::Sumcheck { driver, error },
    )
}

fn observe_stage6_sumcheck_output<F: Field>(
    program: &'static Stage6VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage6SumcheckOutput<F>,
) -> Result<(), VerifyStage6Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                bolt_verifier_runtime::SumcheckPointOrder::BytecodeReadRaf => point = normalize_bytecode_read_raf_point(&point, stage6_trace_rounds(program)?, "stage6.bytecode_read_raf.point")?,
                bolt_verifier_runtime::SumcheckPointOrder::Stage6Booleanity => {}
                _ => {
                    return Err(VerifyStage6Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage6Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage6Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage6Error::InvalidProof { driver, reason },
        |symbol| VerifyStage6Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    verifier_data: Option<&Stage6VerifierData>,
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage6Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let Some(relation) = claim.relation else {
            return Err(VerifyStage6Error::InvalidProof {
                driver: driver.symbol,
                reason: "missing claim relation",
            });
        };
        let value = match relation {
            Stage6RelationKind::Stage6BytecodeReadRaf => {
                let data = verifier_data
                    .and_then(|data| data.bytecode_read_raf.as_ref())
                    .ok_or(VerifyStage6Error::MissingValue {
                        symbol: "stage6.bytecode_read_raf.data",
                })?;
                let log_t = stage6_trace_rounds(program)?;
                let local_scalars = evaluate_stage67_bytecode_read_raf_output_scalars(
                    &STAGE6_BYTECODE_PLAN,
                    &data.entries,
                    data.entry_bytecode_index,
                    data.num_lookup_tables,
                    store,
                    local_point,
                    log_t,
                )?;
                expected_plan_output_claim(program, instance, store, evals, &local_scalars, local_point)?
            }
            Stage6RelationKind::Stage6Booleanity
            | Stage6RelationKind::Stage6HammingBooleanity
            | Stage6RelationKind::Stage6RamRaVirtual
            | Stage6RelationKind::Stage6InstructionRaVirtual
            | Stage6RelationKind::Stage6IncClaimReduction => expected_plan_output_claim(
                program,
                instance,
                store,
                evals,
                &[],
                local_point,
            )?,
            relation => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_plan_output_claim(
    program: &'static Stage6VerifierProgramPlan,
    instance: &'static Stage6SumcheckInstanceResultPlan,
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_scalars: &[bolt_verifier_runtime::NamedScalar<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(bolt_verifier_runtime::evaluate_sumcheck_instance_output_claim(
        program.output_claims,
        program.field_exprs,
        store,
        instance,
        evals,
        local_scalars,
        &[], local_point,
    )?)
}

fn stage6_trace_rounds(
    program: &'static Stage6VerifierProgramPlan,
) -> Result<usize, VerifyStage6Error> {
    Ok(stage67_trace_rounds(program.instance_results, &STAGE6_RELATION_SYMBOLS)?)
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
            Role::Prover => "Stage6CpuProgramPlan",
            Role::Verifier => "Stage6VerifierProgramPlan",
        }
    }
}

fn stage6_kernel_abi(relation: &str) -> Option<&'static str> {
    STAGE6_KERNEL_ABIS
        .iter()
        .find_map(|(candidate, abi)| (*candidate == relation).then_some(*abi))
}

fn parse_indexed_eval_family(
    operation: OperationRef<'_, '_>,
) -> Result<IndexedEvalFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let evals = symbol_array_attr(operation, "evals")?;
    verify_count(
        "indexed eval family",
        &symbol,
        int_attr(operation, "count")?,
        evals.len(),
    )?;
    Ok(IndexedEvalFamilyPlan { symbol, evals })
}

fn stage6_bytecode_read_raf_eval_family(
    eval_families: &[IndexedEvalFamilyPlan],
) -> Result<&IndexedEvalFamilyPlan, EmitError> {
    IndexedEvalFamilyPlan::find(eval_families, STAGE6_BYTECODE_RA_EVAL_FAMILY)
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
    use crate::emit::rust::EmitError;
    use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;

    use super::{
        stage6_bytecode_read_raf_eval_family, stage6_kernel_abi, STAGE6_BYTECODE_RA_EVAL_FAMILY,
        STAGE6_KERNEL_ABIS,
    };

    #[test]
    fn stage6_kernel_abi_contracts_cover_supported_relations() {
        assert_eq!(STAGE6_KERNEL_ABIS.len(), 7);
        assert_eq!(
            stage6_kernel_abi("jolt.stage6.bytecode_read_raf"),
            Some("jolt_stage6_bytecode_read_raf")
        );
        assert_eq!(
            stage6_kernel_abi("jolt.stage6.batched"),
            Some("jolt_stage6_batched")
        );
        assert_eq!(stage6_kernel_abi("jolt.stage7.batched"), None);
    }

    #[test]
    fn stage6_bytecode_read_raf_eval_family_uses_explicit_row_order() -> Result<(), EmitError> {
        let families = [IndexedEvalFamilyPlan {
            symbol: STAGE6_BYTECODE_RA_EVAL_FAMILY.to_owned(),
            evals: vec![
                "stage6.bytecode_read_raf.eval.BytecodeRa_2".to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_0".to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_1".to_owned(),
            ],
        }];
        let family = stage6_bytecode_read_raf_eval_family(&families)?;
        assert_eq!(
            family.evals,
            vec![
                "stage6.bytecode_read_raf.eval.BytecodeRa_2".to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_0".to_owned(),
                "stage6.bytecode_read_raf.eval.BytecodeRa_1".to_owned(),
            ]
        );
        Ok(())
    }

    #[test]
    fn stage6_bytecode_read_raf_eval_family_requires_explicit_plan_row() {
        let families = [IndexedEvalFamilyPlan {
            symbol: "stage6.other.eval.BytecodeRa".to_owned(),
            evals: vec!["stage6.other.eval.BytecodeRa_0".to_owned()],
        }];

        let error = stage6_bytecode_read_raf_eval_family(&families)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert!(error.contains(&format!(
            "missing eval family `{STAGE6_BYTECODE_RA_EVAL_FAMILY}`"
        )));
    }
}
