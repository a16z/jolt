use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::lower_party_to_compute;

const R1CS_INPUT_ORACLES: [&str; 35] = [
    "LeftInstructionInput",
    "RightInstructionInput",
    "Product",
    "ShouldBranch",
    "PC",
    "UnexpandedPC",
    "Imm",
    "RamAddress",
    "Rs1Value",
    "Rs2Value",
    "RdWriteValue",
    "RamReadValue",
    "RamWriteValue",
    "LeftLookupOperand",
    "RightLookupOperand",
    "NextUnexpandedPC",
    "NextPC",
    "NextIsVirtual",
    "NextIsFirstInSequence",
    "LookupOutput",
    "ShouldJump",
    "OpFlagAddOperands",
    "OpFlagSubtractOperands",
    "OpFlagMultiplyOperands",
    "OpFlagLoad",
    "OpFlagStore",
    "OpFlagJump",
    "OpFlagWriteLookupOutputToRD",
    "OpFlagVirtualInstruction",
    "OpFlagAssert",
    "OpFlagDoNotUpdateUnexpandedPC",
    "OpFlagAdvice",
    "OpFlagIsCompressed",
    "OpFlagIsFirstInSequence",
    "OpFlagIsLastInSequence",
];
const OUTER_UNISKIP_FIRST_ROUND_DEGREE_BOUND: usize = 27;

pub fn build_stage1_outer_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage1_outer", None);
    oracles::append_foundation_ops(context, &module, params)?;
    context.append_op_with_owned_attrs(
        &module,
        "protocol.params",
        Some("jolt.params"),
        &params.attrs(),
    )?;
    context.append_op(
        &module,
        "protocol.boundary",
        Some("jolt.stage1_outer"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage1_virtual_oracles(context, &module, params)?;
    append_stage1_relations(context, &module, params)?;

    let fs0 = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs0"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs0, "transcript.state")?;
    let tau = context.append_typed_op(
        &module,
        "transcript.squeeze",
        Some("stage1.tau"),
        &[
            ("label", r#""outer_tau""#),
            ("kind", r#""challenge_vector""#),
            ("count", &int_attr(params.log_t + 2)),
        ],
        &[state],
        &["!transcript.state_type", "!poly.point"],
    )?;
    let state = first_result(tau, "transcript.squeeze")?;

    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage1"),
        &[
            ("name", r#""spartan_outer""#),
            ("order", "1 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;
    let zero_claim = append_field_zero(context, &module, "stage1.zero")?;

    let (state, uniskip_opening, uniskip_eval) =
        append_uniskip_sumcheck(context, &module, params, state, stage, zero_claim)?;
    let _state = append_remaining_sumcheck(
        context,
        &module,
        params,
        state,
        stage,
        uniskip_eval,
        uniskip_opening,
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage1_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(
        context,
        module,
        "jolt.stage1_outer",
        "jolt.stage1_outer",
        "stage1",
    )
}

pub fn resolve_compute_kernels<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    crate::pass::resolve_compute_kernels_with(context, module, kernel_spec)
}

fn append_stage1_virtual_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.stage1_uniskip_domain"),
        &[("field", "@bn254_fr"), ("log_size", "1 : i64")],
    )?;
    append_virtual_oracle(
        context,
        module,
        "UnivariateSkip",
        "jolt.stage1_uniskip_domain",
    )?;
    for oracle in R1CS_INPUT_ORACLES {
        append_virtual_oracle(context, module, oracle, "jolt.trace_domain")?;
    }
    context.append_op(
        module,
        "piop.oracle_family",
        Some("jolt.stage1_r1cs_virtuals"),
        &[
            ("ordered_oracles", &symbol_array_attr(&R1CS_INPUT_ORACLES)),
            ("count", &int_attr(params.num_r1cs_inputs)),
            ("domain", "@jolt.trace_domain"),
            ("visibility", r#""virtual""#),
        ],
    )
}

fn append_virtual_oracle<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
    domain: &str,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.oracle",
        Some(symbol),
        &[
            ("field", "@bn254_fr"),
            ("domain", &format!("@{domain}")),
            ("commit_domain", &format!("@{domain}")),
            ("visibility", r#""virtual""#),
            ("layout", r#""virtual""#),
        ],
    )
}

fn append_stage1_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.relation",
        Some("jolt.stage1.outer.uniskip"),
        &[
            ("kind", r#""sumcheck""#),
            ("domain", "@jolt.stage1_uniskip_domain"),
            ("num_rounds", "1 : i64"),
            ("degree", &int_attr(OUTER_UNISKIP_FIRST_ROUND_DEGREE_BOUND)),
            ("output_count", "1 : i64"),
        ],
    )?;
    context.append_op(
        module,
        "piop.relation",
        Some("jolt.stage1.outer.remaining"),
        &[
            ("kind", r#""sumcheck""#),
            ("domain", "@jolt.trace_domain"),
            ("num_rounds", &int_attr(params.log_t + 1)),
            ("degree", "3 : i64"),
            ("output_count", &int_attr(R1CS_INPUT_ORACLES.len())),
        ],
    )
}

fn append_field_zero<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.zero",
        Some(symbol),
        &[("field", "@bn254_fr")],
        &[],
        &["!field.scalar"],
    )?;
    first_result(op, "field.zero")
}

fn append_uniskip_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    zero_claim: Value<'c, 'a>,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let claim = context.append_typed_op(
        module,
        "piop.sumcheck_claim",
        Some("stage1.uniskip.input"),
        &[
            ("stage", "@stage1"),
            ("domain", "@jolt.stage1_uniskip_domain"),
            ("num_rounds", "1 : i64"),
            ("degree", &int_attr(OUTER_UNISKIP_FIRST_ROUND_DEGREE_BOUND)),
            ("claim", "@stage1.zero"),
            ("relation", "@jolt.stage1.outer.uniskip"),
        ],
        &[zero_claim],
        &["!piop.sumcheck_claim_type"],
    )?;
    let claim = first_result(claim, "piop.sumcheck_claim")?;
    let batch = context.append_typed_op(
        module,
        "piop.sumcheck_batch",
        Some("stage1.uniskip.batch"),
        &[
            ("stage", "@stage1"),
            ("proof_slot", "@stage1.uni_skip_first_round"),
            ("policy", r#""single_instance""#),
            ("count", "1 : i64"),
            ("ordered_claims", "[@stage1.uniskip.input]"),
            ("claim_label", r#""uniskip_claim""#),
            ("round_label", r#""uniskip_poly""#),
            ("round_schedule", "[1]"),
        ],
        &[stage, claim],
        &["!piop.sumcheck_batch_type"],
    )?;
    let batch = first_result(batch, "piop.sumcheck_batch")?;
    let sumcheck = context.append_typed_op(
        module,
        "piop.sumcheck",
        Some("stage1.uniskip.sumcheck"),
        &[
            ("stage", "@stage1"),
            ("proof_slot", "@stage1.uni_skip_first_round"),
            ("relation", "@jolt.stage1.outer.uniskip"),
            ("policy", r#""univariate_skip""#),
            ("round_schedule", "[1]"),
            ("claim_label", r#""uniskip_claim""#),
            ("round_label", r#""uniskip_poly""#),
            ("num_rounds", "1 : i64"),
            ("degree", &int_attr(OUTER_UNISKIP_FIRST_ROUND_DEGREE_BOUND)),
        ],
        &[state, batch],
        &[
            "!transcript.state_type",
            "!poly.point",
            "!piop.sumcheck_result_type",
            "!piop.sumcheck_proof_type",
        ],
    )?;
    let state = result(sumcheck, 0, "piop.sumcheck")?;
    let point = result(sumcheck, 1, "piop.sumcheck")?;
    let result_value = result(sumcheck, 2, "piop.sumcheck")?;
    let (point, result_value) = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage1.uniskip.instance",
            source: "stage1.uniskip.sumcheck",
            claim: "stage1.uniskip.input",
            relation: "jolt.stage1.outer.uniskip",
            index: 0,
            point_arity: 1,
            num_rounds: 1,
            round_offset: 0,
            point_order: "as_is",
            degree: OUTER_UNISKIP_FIRST_ROUND_DEGREE_BOUND,
        },
        point,
        result_value,
    )?;
    let eval = append_sumcheck_eval(
        context,
        module,
        "stage1.uniskip.eval",
        "stage1.uniskip.sumcheck",
        "UnivariateSkip",
        0,
        result_value,
    )?;
    let opening = append_piop_opening_claim(
        context,
        module,
        point,
        eval,
        OpeningClaimSpec {
            symbol: "stage1.uniskip.opening",
            oracle: "UnivariateSkip",
            domain: "jolt.stage1_uniskip_domain",
            point_arity: 1,
        },
    )?;
    let _ = params;
    Ok((state, opening, eval))
}

fn append_remaining_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    input_claim: Value<'c, 'a>,
    uniskip_opening: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let num_rounds = params.log_t + 1;
    let claim = context.append_typed_op(
        module,
        "piop.sumcheck_claim",
        Some("stage1.outer_remaining.input"),
        &[
            ("stage", "@stage1"),
            ("domain", "@jolt.trace_domain"),
            ("num_rounds", &int_attr(num_rounds)),
            ("degree", "3 : i64"),
            ("claim", "@stage1.uniskip.eval"),
            ("relation", "@jolt.stage1.outer.remaining"),
        ],
        &[input_claim, uniskip_opening],
        &["!piop.sumcheck_claim_type"],
    )?;
    let claim = first_result(claim, "piop.sumcheck_claim")?;
    let batch = context.append_typed_op(
        module,
        "piop.sumcheck_batch",
        Some("stage1.outer_remaining.batch"),
        &[
            ("stage", "@stage1"),
            ("proof_slot", "@stage1.sumcheck"),
            ("policy", r#""jolt_core_front_loaded""#),
            ("count", "1 : i64"),
            ("ordered_claims", "[@stage1.outer_remaining.input]"),
            ("claim_label", r#""sumcheck_claim""#),
            ("round_label", r#""sumcheck_poly""#),
            ("round_schedule", &format!("[{}]", num_rounds)),
        ],
        &[stage, claim],
        &["!piop.sumcheck_batch_type"],
    )?;
    let batch = first_result(batch, "piop.sumcheck_batch")?;
    let sumcheck = context.append_typed_op(
        module,
        "piop.sumcheck",
        Some("stage1.outer_remaining.sumcheck"),
        &[
            ("stage", "@stage1"),
            ("proof_slot", "@stage1.sumcheck"),
            ("relation", "@jolt.stage1.outer.remaining"),
            ("policy", r#""jolt_core_front_loaded""#),
            ("round_schedule", &format!("[{}]", num_rounds)),
            ("claim_label", r#""sumcheck_claim""#),
            ("round_label", r#""sumcheck_poly""#),
            ("num_rounds", &int_attr(num_rounds)),
            ("degree", "3 : i64"),
        ],
        &[state, batch],
        &[
            "!transcript.state_type",
            "!poly.point",
            "!piop.sumcheck_result_type",
            "!piop.sumcheck_proof_type",
        ],
    )?;
    let state = result(sumcheck, 0, "piop.sumcheck")?;
    let point = result(sumcheck, 1, "piop.sumcheck")?;
    let result_value = result(sumcheck, 2, "piop.sumcheck")?;
    let (point, result_value) = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage1.outer_remaining.instance",
            source: "stage1.outer_remaining.sumcheck",
            claim: "stage1.outer_remaining.input",
            relation: "jolt.stage1.outer.remaining",
            index: 0,
            point_arity: params.log_t,
            num_rounds,
            round_offset: 1,
            point_order: "reverse",
            degree: 3,
        },
        point,
        result_value,
    )?;
    let mut claims = Vec::with_capacity(R1CS_INPUT_ORACLES.len());
    for (index, oracle) in R1CS_INPUT_ORACLES.iter().enumerate() {
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!("stage1.outer_remaining.eval.{oracle}"),
            "stage1.outer_remaining.sumcheck",
            oracle,
            index,
            result_value,
        )?;
        claims.push(append_piop_opening_claim(
            context,
            module,
            point,
            eval,
            OpeningClaimSpec {
                symbol: &format!("stage1.outer_remaining.opening.{oracle}"),
                oracle,
                domain: "jolt.trace_domain",
                point_arity: params.log_t,
            },
        )?);
    }
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage1.outer_remaining.openings"),
        &[
            ("stage", "@stage1"),
            ("proof_slot", "@stage1.virtual_openings"),
            ("policy", r#""jolt_r1cs_input_order""#),
            ("count", &int_attr(R1CS_INPUT_ORACLES.len())),
            ("ordered_claims", &opening_claim_attr()),
        ],
        &claims,
        &["!piop.opening_batch_type"],
    )?;
    Ok(state)
}

fn append_sumcheck_eval<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    source: &str,
    oracle: &str,
    index: usize,
    result_value: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_eval",
        Some(symbol),
        &[
            ("source", &format!("@{source}")),
            ("name", &format!("@{symbol}")),
            ("index", &int_attr(index)),
            ("oracle", &format!("@{oracle}")),
        ],
        &[result_value],
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_eval")
}

fn append_sumcheck_instance_result<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: SumcheckInstanceResultSpec<'_>,
    point: Value<'c, 'a>,
    result_value: Value<'c, 'a>,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_instance_result",
        Some(spec.symbol),
        &[
            ("source", &format!("@{}", spec.source)),
            ("claim", &format!("@{}", spec.claim)),
            ("relation", &format!("@{}", spec.relation)),
            ("index", &int_attr(spec.index)),
            ("point_arity", &int_attr(spec.point_arity)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("round_offset", &int_attr(spec.round_offset)),
            ("point_order", &format!("\"{}\"", spec.point_order)),
            ("degree", &int_attr(spec.degree)),
        ],
        &[point, result_value],
        &["!poly.point", "!piop.sumcheck_result_type"],
    )?;
    Ok((
        result(op, 0, "piop.sumcheck_instance_result")?,
        result(op, 1, "piop.sumcheck_instance_result")?,
    ))
}

fn append_piop_opening_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
    spec: OpeningClaimSpec<'_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.opening_claim",
        Some(spec.symbol),
        &[
            ("oracle", &format!("@{}", spec.oracle)),
            ("domain", &format!("@{}", spec.domain)),
            ("point_arity", &int_attr(spec.point_arity)),
            ("claim_kind", r#""virtual""#),
        ],
        &[point, eval],
        &["!piop.opening_claim_type"],
    )?;
    first_result(op, "piop.opening_claim")
}

struct OpeningClaimSpec<'a> {
    symbol: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
}

struct SumcheckInstanceResultSpec<'a> {
    symbol: &'a str,
    source: &'a str,
    claim: &'a str,
    relation: &'a str,
    index: usize,
    point_arity: usize,
    num_rounds: usize,
    round_offset: usize,
    point_order: &'a str,
    degree: usize,
}

fn first_result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    operation_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    result(operation, 0, operation_name)
}

fn result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    index: usize,
    operation_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    operation
        .result(index)
        .map(Into::into)
        .map_err(|_| schema_error(format!("{operation_name} requires result {index}")))
}

fn kernel_spec(relation: &str) -> Result<crate::pass::ComputeKernelSpec, MlirError> {
    match relation {
        "jolt.stage1.outer.uniskip" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage1.outer.uniskip",
            "sumcheck",
            "cpu",
            "jolt_stage1_outer_uniskip",
        )),
        "jolt.stage1.outer.remaining" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage1.outer.remaining",
            "sumcheck",
            "cpu",
            "jolt_stage1_outer_remaining",
        )),
        "jolt.stage2.product_virtual.uniskip" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage2.product_virtual.uniskip",
            "sumcheck",
            "cpu",
            "jolt_stage2_product_virtual_uniskip",
        )),
        "jolt.stage2.ram.read_write" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage2.ram.read_write",
            "sumcheck",
            "cpu",
            "jolt_stage2_ram_read_write",
        )),
        "jolt.stage2.product_virtual.remainder" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage2.product_virtual.remainder",
            "sumcheck",
            "cpu",
            "jolt_stage2_product_virtual_remainder",
        )),
        "jolt.stage2.instruction_lookup.claim_reduction" => {
            Ok(crate::pass::ComputeKernelSpec::new(
                "jolt.cpu.stage2.instruction_lookup.claim_reduction",
                "sumcheck",
                "cpu",
                "jolt_stage2_instruction_lookup_claim_reduction",
            ))
        }
        "jolt.stage2.ram.raf_evaluation" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage2.ram.raf_evaluation",
            "sumcheck",
            "cpu",
            "jolt_stage2_ram_raf_evaluation",
        )),
        "jolt.stage2.ram.output_check" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage2.ram.output_check",
            "sumcheck",
            "cpu",
            "jolt_stage2_ram_output_check",
        )),
        "jolt.stage2.batched" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage2.batched",
            "sumcheck",
            "cpu",
            "jolt_stage2_batched",
        )),
        "jolt.stage3.spartan_shift" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage3.spartan_shift",
            "sumcheck",
            "cpu",
            "jolt_stage3_spartan_shift",
        )),
        "jolt.stage3.instruction_input" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage3.instruction_input",
            "sumcheck",
            "cpu",
            "jolt_stage3_instruction_input",
        )),
        "jolt.stage3.registers_claim_reduction" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage3.registers_claim_reduction",
            "sumcheck",
            "cpu",
            "jolt_stage3_registers_claim_reduction",
        )),
        "jolt.stage3.batched" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage3.batched",
            "sumcheck",
            "cpu",
            "jolt_stage3_batched",
        )),
        "jolt.stage4.registers_read_write" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage4.registers_read_write",
            "sumcheck",
            "cpu",
            "jolt_stage4_registers_read_write",
        )),
        "jolt.stage4.ram_val_check" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage4.ram_val_check",
            "sumcheck",
            "cpu",
            "jolt_stage4_ram_val_check",
        )),
        "jolt.stage4.batched" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage4.batched",
            "sumcheck",
            "cpu",
            "jolt_stage4_batched",
        )),
        "jolt.stage5.instruction_read_raf" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage5.instruction_read_raf",
            "sumcheck",
            "cpu",
            "jolt_stage5_instruction_read_raf",
        )),
        "jolt.stage5.ram_ra_claim_reduction" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage5.ram_ra_claim_reduction",
            "sumcheck",
            "cpu",
            "jolt_stage5_ram_ra_claim_reduction",
        )),
        "jolt.stage5.registers_val_evaluation" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage5.registers_val_evaluation",
            "sumcheck",
            "cpu",
            "jolt_stage5_registers_val_evaluation",
        )),
        "jolt.stage5.batched" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage5.batched",
            "sumcheck",
            "cpu",
            "jolt_stage5_batched",
        )),
        "jolt.stage6.bytecode_read_raf" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.bytecode_read_raf",
            "sumcheck",
            "cpu",
            "jolt_stage6_bytecode_read_raf",
        )),
        "jolt.stage6.booleanity" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.booleanity",
            "sumcheck",
            "cpu",
            "jolt_stage6_booleanity",
        )),
        "jolt.stage6.hamming_booleanity" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.hamming_booleanity",
            "sumcheck",
            "cpu",
            "jolt_stage6_hamming_booleanity",
        )),
        "jolt.stage6.ram_ra_virtual" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.ram_ra_virtual",
            "sumcheck",
            "cpu",
            "jolt_stage6_ram_ra_virtual",
        )),
        "jolt.stage6.instruction_ra_virtual" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.instruction_ra_virtual",
            "sumcheck",
            "cpu",
            "jolt_stage6_instruction_ra_virtual",
        )),
        "jolt.stage6.inc_claim_reduction" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.inc_claim_reduction",
            "sumcheck",
            "cpu",
            "jolt_stage6_inc_claim_reduction",
        )),
        "jolt.stage6.batched" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage6.batched",
            "sumcheck",
            "cpu",
            "jolt_stage6_batched",
        )),
        "jolt.stage7.hamming_weight_claim_reduction" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage7.hamming_weight_claim_reduction",
            "sumcheck",
            "cpu",
            "jolt_stage7_hamming_weight_claim_reduction",
        )),
        "jolt.stage7.batched" => Ok(crate::pass::ComputeKernelSpec::new(
            "jolt.cpu.stage7.batched",
            "sumcheck",
            "cpu",
            "jolt_stage7_batched",
        )),
        _ => Err(schema_error(format!(
            "unsupported compute relation @{relation}"
        ))),
    }
}

fn int_attr(value: usize) -> String {
    format!("{value} : i64")
}

fn symbol_array_attr(values: &[&str]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn opening_claim_attr() -> String {
    let values = R1CS_INPUT_ORACLES
        .iter()
        .map(|oracle| format!("@stage1.outer_remaining.opening.{oracle}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn schema_error(message: impl Into<String>) -> MlirError {
    let error = SchemaError::new(message);
    error.into()
}
