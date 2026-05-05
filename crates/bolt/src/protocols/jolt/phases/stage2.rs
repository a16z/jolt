use std::collections::BTreeSet;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_protocol_schema, SchemaError};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::{lower_party_to_compute, transcript_squeeze_protocol_result_type};

const PRODUCT_UNISKIP_DEGREE_BOUND: usize = 6;
const PRODUCT_UNISKIP_DOMAIN_START: isize = -1;
const PRODUCT_UNISKIP_DOMAIN_SIZE: usize = 3;
const RAM_RW_DEGREE: usize = 3;
const PRODUCT_REMAINDER_DEGREE: usize = 3;
const INSTRUCTION_CLAIM_REDUCTION_DEGREE: usize = 2;
const RAM_RAF_DEGREE: usize = 2;
const RAM_OUTPUT_DEGREE: usize = 3;

const STAGE1_PRODUCT_OPENINGS: [&str; 3] = ["Product", "ShouldBranch", "ShouldJump"];
const STAGE2_RAM_RW_INPUTS: [&str; 2] = ["RamReadValue", "RamWriteValue"];
const STAGE2_INSTRUCTION_INPUTS: [&str; 5] = [
    "LookupOutput",
    "LeftLookupOperand",
    "RightLookupOperand",
    "LeftInstructionInput",
    "RightInstructionInput",
];
const PRODUCT_REMAINDER_OUTPUTS: [&str; 8] = [
    "LeftInstructionInput",
    "RightInstructionInput",
    "OpFlagJump",
    "OpFlagWriteLookupOutputToRD",
    "LookupOutput",
    "InstructionFlagBranch",
    "NextIsNoop",
    "OpFlagVirtualInstruction",
];

pub fn build_stage2_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage2", None);
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
        Some("jolt.stage2"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage2_domains(context, &module, params)?;
    append_stage2_oracles(context, &module)?;
    append_stage2_relations(context, &module, params)?;
    let inputs = append_stage2_opening_inputs(context, &module, params)?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage1"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs, "transcript.state")?;
    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage2"),
        &[
            ("name", r#""product_virtual_and_ram""#),
            ("order", "2 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;
    let (state, tau_high) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage2.product_virtual.tau_high",
        "product_virtual_tau_high",
        "challenge_scalar",
        1,
    )?;
    let (state, uniskip) =
        append_product_uniskip(context, &module, params, state, stage, &inputs, tau_high)?;
    let (state, ram_read_write_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage2.ram_read_write.gamma",
        "ram_read_write_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, instruction_lookup_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage2.instruction_lookup.gamma",
        "instruction_lookup_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, _ram_output_address) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage2.ram_output.r_address",
        "ram_output_r_address",
        "challenge_vector",
        params.log_k_ram,
    )?;
    let _state = append_stage2_batched_sumcheck(
        context,
        &module,
        params,
        Stage2BatchedSumcheckInputs {
            state,
            stage,
            openings: &inputs,
            uniskip,
            ram_read_write_gamma,
            instruction_lookup_gamma,
        },
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage2_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    lower_party_to_compute(context, module, "jolt.stage2", "jolt.stage2", "stage2")
}

fn append_stage2_domains<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.stage2_uniskip_domain"),
        &[("field", "@bn254_fr"), ("log_size", "1 : i64")],
    )?;
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.stage2_ram_rw_domain"),
        &[
            ("field", "@bn254_fr"),
            ("log_size", &int_attr(stage2_max_rounds(params))),
        ],
    )?;
    context.append_op(
        module,
        "poly.domain",
        Some("jolt.ram_address_domain"),
        &[
            ("field", "@bn254_fr"),
            ("log_size", &int_attr(params.log_k_ram)),
        ],
    )
}

fn append_stage2_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
) -> Result<(), MlirError> {
    let mut trace_oracles = BTreeSet::new();
    trace_oracles.extend(STAGE1_PRODUCT_OPENINGS);
    trace_oracles.extend(STAGE2_RAM_RW_INPUTS);
    trace_oracles.extend(STAGE2_INSTRUCTION_INPUTS);
    trace_oracles.extend(PRODUCT_REMAINDER_OUTPUTS);
    let _ = trace_oracles.insert("RamAddress");
    for oracle in trace_oracles {
        append_virtual_oracle(context, module, oracle, "jolt.trace_domain")?;
    }
    append_virtual_oracle(
        context,
        module,
        "UnivariateSkip",
        "jolt.stage2_uniskip_domain",
    )?;
    append_virtual_oracle(context, module, "RamVal", "jolt.stage2_ram_rw_domain")?;
    append_virtual_oracle(context, module, "RamRa", "jolt.stage2_ram_rw_domain")?;
    append_virtual_oracle(context, module, "RamValFinal", "jolt.ram_address_domain")?;
    context.append_op(
        module,
        "piop.oracle",
        Some("RamInc"),
        &[
            ("field", "@bn254_fr"),
            ("domain", "@jolt.trace_domain"),
            ("commit_domain", "@jolt.main_witness_commit_domain"),
            ("visibility", r#""committed""#),
            ("layout", r#""dense_trace""#),
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

fn append_stage2_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    let max_rounds = stage2_max_rounds(params);
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.product_virtual.uniskip",
            kind: "sumcheck",
            domain: "jolt.stage2_uniskip_domain",
            num_rounds: 1,
            degree: PRODUCT_UNISKIP_DEGREE_BOUND,
            output_count: 1,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.ram.read_write",
            kind: "sumcheck",
            domain: "jolt.stage2_ram_rw_domain",
            num_rounds: max_rounds,
            degree: RAM_RW_DEGREE,
            output_count: 3,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.product_virtual.remainder",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: PRODUCT_REMAINDER_DEGREE,
            output_count: PRODUCT_REMAINDER_OUTPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.instruction_lookup.claim_reduction",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: INSTRUCTION_CLAIM_REDUCTION_DEGREE,
            output_count: STAGE2_INSTRUCTION_INPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.ram.raf_evaluation",
            kind: "sumcheck",
            domain: "jolt.ram_address_domain",
            num_rounds: params.log_k_ram,
            degree: RAM_RAF_DEGREE,
            output_count: 1,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.ram.output_check",
            kind: "sumcheck",
            domain: "jolt.ram_address_domain",
            num_rounds: params.log_k_ram,
            degree: RAM_OUTPUT_DEGREE,
            output_count: 1,
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage2.batched",
            kind: "batched_sumcheck",
            domain: "jolt.stage2_ram_rw_domain",
            num_rounds: max_rounds,
            degree: RAM_RW_DEGREE,
            output_count: 18,
        },
    )
}

fn append_relation<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    spec: RelationSpec<'_>,
) -> Result<(), MlirError> {
    context.append_op(
        module,
        "piop.relation",
        Some(spec.symbol),
        &[
            ("kind", &format!("\"{}\"", spec.kind)),
            ("domain", &format!("@{}", spec.domain)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("degree", &int_attr(spec.degree)),
            ("output_count", &int_attr(spec.output_count)),
        ],
    )
}

fn append_stage2_opening_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<Stage2OpeningInputs<'c, 'a>, MlirError> {
    let product = append_stage1_opening_input(context, module, params, "Product")?;
    let should_branch = append_stage1_opening_input(context, module, params, "ShouldBranch")?;
    let should_jump = append_stage1_opening_input(context, module, params, "ShouldJump")?;
    let ram_read_value = append_stage1_opening_input(context, module, params, "RamReadValue")?;
    let ram_write_value = append_stage1_opening_input(context, module, params, "RamWriteValue")?;
    let lookup_output = append_stage1_opening_input(context, module, params, "LookupOutput")?;
    let left_lookup_operand =
        append_stage1_opening_input(context, module, params, "LeftLookupOperand")?;
    let right_lookup_operand =
        append_stage1_opening_input(context, module, params, "RightLookupOperand")?;
    let left_instruction_input =
        append_stage1_opening_input(context, module, params, "LeftInstructionInput")?;
    let right_instruction_input =
        append_stage1_opening_input(context, module, params, "RightInstructionInput")?;
    let ram_address = append_stage1_opening_input(context, module, params, "RamAddress")?;

    Ok(Stage2OpeningInputs {
        product,
        should_branch,
        should_jump,
        ram_read_value,
        ram_write_value,
        lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        left_instruction_input,
        right_instruction_input,
        ram_address,
    })
}

fn append_stage1_opening_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    oracle: &str,
) -> Result<Stage2OpeningInput<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.opening_input",
        Some(&format!("stage2.input.stage1.{oracle}")),
        &[
            ("source_stage", "@stage1"),
            (
                "source_claim",
                &format!("@stage1.outer_remaining.opening.{oracle}"),
            ),
            ("oracle", &format!("@{oracle}")),
            ("domain", "@jolt.trace_domain"),
            ("point_arity", &int_attr(params.log_t)),
            ("claim_kind", r#""virtual""#),
        ],
        &[],
        &["!poly.point", "!field.scalar", "!piop.opening_claim_type"],
    )?;
    Ok(Stage2OpeningInput {
        point: result(op, 0, "piop.opening_input")?,
        eval: result(op, 1, "piop.opening_input")?,
        claim: result(op, 2, "piop.opening_input")?,
    })
}

fn append_transcript_squeeze<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    state: Value<'c, 'a>,
    symbol: &str,
    label: &str,
    kind: &str,
    count: usize,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let op = context.append_typed_op(
        module,
        "transcript.squeeze",
        Some(symbol),
        &[
            ("label", &format!("\"{label}\"")),
            ("kind", &format!("\"{kind}\"")),
            ("count", &int_attr(count)),
        ],
        &[state],
        &[
            "!transcript.state_type",
            transcript_squeeze_protocol_result_type(kind)?,
        ],
    )?;
    Ok((
        result(op, 0, "transcript.squeeze")?,
        result(op, 1, "transcript.squeeze")?,
    ))
}

fn append_field_const<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    value: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.const",
        Some(symbol),
        &[("field", "@bn254_fr"), ("value", &int_attr(value))],
        &[],
        &["!field.scalar"],
    )?;
    first_result(op, "field.const")
}

fn append_field_binary<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    op_name: &str,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        op_name,
        Some(symbol),
        &[],
        &[lhs, rhs],
        &["!field.scalar"],
    )?;
    first_result(op, op_name)
}

fn append_field_add<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    append_field_binary(context, module, "field.add", symbol, lhs, rhs)
}

fn append_field_mul<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    append_field_binary(context, module, "field.mul", symbol, lhs, rhs)
}

fn append_lagrange_basis_eval<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    point: Value<'c, 'a>,
    index: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.lagrange_basis_eval",
        Some(symbol),
        &[
            (
                "domain_start",
                &int_attr_signed(PRODUCT_UNISKIP_DOMAIN_START),
            ),
            ("domain_size", &int_attr(PRODUCT_UNISKIP_DOMAIN_SIZE)),
            ("index", &int_attr(index)),
        ],
        &[point],
        &["!field.scalar"],
    )?;
    first_result(op, "poly.lagrange_basis_eval")
}

fn append_product_uniskip<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    _params: &JoltProtocolParams,
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    inputs: &Stage2OpeningInputs<'c, 'a>,
    tau_high: Value<'c, 'a>,
) -> Result<(Value<'c, 'a>, Stage2UniskipOutput<'c, 'a>), MlirError> {
    let product_weight = append_lagrange_basis_eval(
        context,
        module,
        "stage2.product_virtual.uniskip.weight.Product",
        tau_high,
        0,
    )?;
    let branch_weight = append_lagrange_basis_eval(
        context,
        module,
        "stage2.product_virtual.uniskip.weight.ShouldBranch",
        tau_high,
        1,
    )?;
    let jump_weight = append_lagrange_basis_eval(
        context,
        module,
        "stage2.product_virtual.uniskip.weight.ShouldJump",
        tau_high,
        2,
    )?;
    let product_term = append_field_mul(
        context,
        module,
        "stage2.product_virtual.uniskip.term.Product",
        product_weight,
        inputs.product.eval,
    )?;
    let branch_term = append_field_mul(
        context,
        module,
        "stage2.product_virtual.uniskip.term.ShouldBranch",
        branch_weight,
        inputs.should_branch.eval,
    )?;
    let jump_term = append_field_mul(
        context,
        module,
        "stage2.product_virtual.uniskip.term.ShouldJump",
        jump_weight,
        inputs.should_jump.eval,
    )?;
    let product_branch_sum = append_field_add(
        context,
        module,
        "stage2.product_virtual.uniskip.partial.ProductShouldBranch",
        product_term,
        branch_term,
    )?;
    let input_claim = append_field_add(
        context,
        module,
        "stage2.product_virtual.uniskip.claim_expr",
        product_branch_sum,
        jump_term,
    )?;
    let claim = append_sumcheck_claim(
        context,
        module,
        SumcheckClaimSpec {
            symbol: "stage2.product_virtual.uniskip.input",
            stage: "stage2",
            domain: "jolt.stage2_uniskip_domain",
            num_rounds: 1,
            degree: PRODUCT_UNISKIP_DEGREE_BOUND,
            claim: "stage2.product_virtual.weighted_stage1_outputs",
            relation: "jolt.stage2.product_virtual.uniskip",
        },
        input_claim,
        &[
            inputs.product.claim,
            inputs.should_branch.claim,
            inputs.should_jump.claim,
        ],
    )?;
    let batch = append_sumcheck_batch(
        context,
        module,
        stage,
        &[claim],
        SumcheckBatchSpec {
            symbol: "stage2.product_virtual.uniskip.batch",
            stage: "stage2",
            proof_slot: "stage2.product_virtual.uni_skip_first_round",
            policy: "single_instance",
            ordered_claims: &["stage2.product_virtual.uniskip.input"],
            claim_label: "uniskip_claim",
            round_label: "uniskip_poly",
            round_schedule: "[1]".to_owned(),
        },
    )?;
    let (state, point, result_value) = append_sumcheck(
        context,
        module,
        state,
        batch,
        SumcheckDriverSpec {
            symbol: "stage2.product_virtual.uniskip.sumcheck",
            stage: "stage2",
            proof_slot: "stage2.product_virtual.uni_skip_first_round",
            relation: "jolt.stage2.product_virtual.uniskip",
            policy: "univariate_skip",
            round_schedule: "[1]".to_owned(),
            claim_label: "uniskip_claim",
            round_label: "uniskip_poly",
            num_rounds: 1,
            degree: PRODUCT_UNISKIP_DEGREE_BOUND,
        },
    )?;
    let (point, result_value) = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage2.product_virtual.uniskip.instance",
            source: "stage2.product_virtual.uniskip.sumcheck",
            claim: "stage2.product_virtual.uniskip.input",
            relation: "jolt.stage2.product_virtual.uniskip",
            index: 0,
            point_arity: 1,
            num_rounds: 1,
            round_offset: 0,
            point_order: "as_is",
            degree: PRODUCT_UNISKIP_DEGREE_BOUND,
        },
        point,
        result_value,
    )?;
    let eval = append_sumcheck_eval(
        context,
        module,
        "stage2.product_virtual.uniskip.eval.UnivariateSkip",
        "stage2.product_virtual.uniskip.sumcheck",
        "UnivariateSkip",
        0,
        result_value,
    )?;
    let opening = append_opening_claim(
        context,
        module,
        point,
        eval,
        OpeningClaimSpec {
            symbol: "stage2.product_virtual.uniskip.opening.UnivariateSkip",
            oracle: "UnivariateSkip",
            domain: "jolt.stage2_uniskip_domain",
            point_arity: 1,
            claim_kind: "virtual",
        },
    )?;
    Ok((state, Stage2UniskipOutput { opening, eval }))
}

fn append_stage2_batched_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage2BatchedSumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let inputs = spec.openings;
    let uniskip = spec.uniskip;
    let max_rounds = stage2_max_rounds(params);
    let product_offset = max_rounds - params.log_t;
    let ram_offset = params.log_t;
    let ram_write_term = append_field_mul(
        context,
        module,
        "stage2.ram_read_write.term.RamWriteValue",
        spec.ram_read_write_gamma,
        inputs.ram_write_value.eval,
    )?;
    let ram_read_write_claim = append_field_add(
        context,
        module,
        "stage2.ram_read_write.claim_expr",
        inputs.ram_read_value.eval,
        ram_write_term,
    )?;
    let product_remainder_claim = uniskip.eval;
    let gamma2 = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.gamma2",
        spec.instruction_lookup_gamma,
        spec.instruction_lookup_gamma,
    )?;
    let gamma3 = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.gamma3",
        gamma2,
        spec.instruction_lookup_gamma,
    )?;
    let gamma4 = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.gamma4",
        gamma2,
        gamma2,
    )?;
    let left_lookup_term = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.term.LeftLookupOperand",
        spec.instruction_lookup_gamma,
        inputs.left_lookup_operand.eval,
    )?;
    let right_lookup_term = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.term.RightLookupOperand",
        gamma2,
        inputs.right_lookup_operand.eval,
    )?;
    let left_input_term = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.term.LeftInstructionInput",
        gamma3,
        inputs.left_instruction_input.eval,
    )?;
    let right_input_term = append_field_mul(
        context,
        module,
        "stage2.instruction_lookup.term.RightInstructionInput",
        gamma4,
        inputs.right_instruction_input.eval,
    )?;
    let instruction_sum_0 = append_field_add(
        context,
        module,
        "stage2.instruction_lookup.partial.LookupOutputLeftOperand",
        inputs.lookup_output.eval,
        left_lookup_term,
    )?;
    let instruction_sum_1 = append_field_add(
        context,
        module,
        "stage2.instruction_lookup.partial.RightOperand",
        instruction_sum_0,
        right_lookup_term,
    )?;
    let instruction_sum_2 = append_field_add(
        context,
        module,
        "stage2.instruction_lookup.partial.LeftInstructionInput",
        instruction_sum_1,
        left_input_term,
    )?;
    let instruction_claim = append_field_add(
        context,
        module,
        "stage2.instruction_lookup.claim_reduction.claim_expr",
        instruction_sum_2,
        right_input_term,
    )?;
    let ram_raf_claim = inputs.ram_address.eval;
    let ram_output_claim = append_field_const(context, module, "stage2.ram_output.zero", 0)?;
    let claims = [
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage2.ram_read_write.input",
                stage: "stage2",
                domain: "jolt.stage2_ram_rw_domain",
                num_rounds: max_rounds,
                degree: RAM_RW_DEGREE,
                claim: "stage2.ram_read_write.weighted_values",
                relation: "jolt.stage2.ram.read_write",
            },
            ram_read_write_claim,
            &[inputs.ram_read_value.claim, inputs.ram_write_value.claim],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage2.product_virtual.remainder.input",
                stage: "stage2",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: PRODUCT_REMAINDER_DEGREE,
                claim: "stage2.product_virtual.uniskip.opening",
                relation: "jolt.stage2.product_virtual.remainder",
            },
            product_remainder_claim,
            &[uniskip.opening],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage2.instruction_lookup.claim_reduction.input",
                stage: "stage2",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: INSTRUCTION_CLAIM_REDUCTION_DEGREE,
                claim: "stage2.instruction_lookup.weighted_operands",
                relation: "jolt.stage2.instruction_lookup.claim_reduction",
            },
            instruction_claim,
            &[
                inputs.lookup_output.claim,
                inputs.left_lookup_operand.claim,
                inputs.right_lookup_operand.claim,
                inputs.left_instruction_input.claim,
                inputs.right_instruction_input.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage2.ram_raf.input",
                stage: "stage2",
                domain: "jolt.ram_address_domain",
                num_rounds: params.log_k_ram,
                degree: RAM_RAF_DEGREE,
                claim: "stage2.ram_raf.ram_address",
                relation: "jolt.stage2.ram.raf_evaluation",
            },
            ram_raf_claim,
            &[inputs.ram_address.claim],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage2.ram_output.input",
                stage: "stage2",
                domain: "jolt.ram_address_domain",
                num_rounds: params.log_k_ram,
                degree: RAM_OUTPUT_DEGREE,
                claim: "zero",
                relation: "jolt.stage2.ram.output_check",
            },
            ram_output_claim,
            &[],
        )?,
    ];
    let batch = append_sumcheck_batch(
        context,
        module,
        spec.stage,
        &claims,
        SumcheckBatchSpec {
            symbol: "stage2.batch",
            stage: "stage2",
            proof_slot: "stage2.sumcheck",
            policy: "jolt_core_stage2_aligned",
            ordered_claims: &[
                "stage2.ram_read_write.input",
                "stage2.product_virtual.remainder.input",
                "stage2.instruction_lookup.claim_reduction.input",
                "stage2.ram_raf.input",
                "stage2.ram_output.input",
            ],
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            round_schedule: format!("[{}, {}]", params.log_t, params.log_k_ram),
        },
    )?;
    let (state, point, result_value) = append_sumcheck(
        context,
        module,
        spec.state,
        batch,
        SumcheckDriverSpec {
            symbol: "stage2.sumcheck",
            stage: "stage2",
            proof_slot: "stage2.sumcheck",
            relation: "jolt.stage2.batched",
            policy: "jolt_core_stage2_aligned",
            round_schedule: format!("[{}, {}]", params.log_t, params.log_k_ram),
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: max_rounds,
            degree: RAM_RW_DEGREE,
        },
    )?;
    let ram_rw = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage2.ram_read_write.instance",
            source: "stage2.sumcheck",
            claim: "stage2.ram_read_write.input",
            relation: "jolt.stage2.ram.read_write",
            index: 0,
            point_arity: max_rounds,
            num_rounds: max_rounds,
            round_offset: 0,
            point_order: "as_is",
            degree: RAM_RW_DEGREE,
        },
        point,
        result_value,
    )?;
    let product = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage2.product_virtual.remainder.instance",
            source: "stage2.sumcheck",
            claim: "stage2.product_virtual.remainder.input",
            relation: "jolt.stage2.product_virtual.remainder",
            index: 1,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: product_offset,
            point_order: "reverse",
            degree: PRODUCT_REMAINDER_DEGREE,
        },
        point,
        result_value,
    )?;
    let instruction = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage2.instruction_lookup.claim_reduction.instance",
            source: "stage2.sumcheck",
            claim: "stage2.instruction_lookup.claim_reduction.input",
            relation: "jolt.stage2.instruction_lookup.claim_reduction",
            index: 2,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: product_offset,
            point_order: "reverse",
            degree: INSTRUCTION_CLAIM_REDUCTION_DEGREE,
        },
        point,
        result_value,
    )?;
    let ram_raf = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage2.ram_raf.instance",
            source: "stage2.sumcheck",
            claim: "stage2.ram_raf.input",
            relation: "jolt.stage2.ram.raf_evaluation",
            index: 3,
            point_arity: params.log_k_ram,
            num_rounds: params.log_k_ram,
            round_offset: ram_offset,
            point_order: "reverse",
            degree: RAM_RAF_DEGREE,
        },
        point,
        result_value,
    )?;
    let ram_output = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage2.ram_output.instance",
            source: "stage2.sumcheck",
            claim: "stage2.ram_output.input",
            relation: "jolt.stage2.ram.output_check",
            index: 4,
            point_arity: params.log_k_ram,
            num_rounds: params.log_k_ram,
            round_offset: ram_offset,
            point_order: "reverse",
            degree: RAM_OUTPUT_DEGREE,
        },
        point,
        result_value,
    )?;
    append_stage2_output_openings(
        context,
        module,
        params,
        Stage2OutputOpeningSpec {
            outputs: &[
                InstanceOutput {
                    prefix: "stage2.product_virtual.remainder",
                    instance: product,
                    eval_source: "stage2.sumcheck",
                    outputs: &PRODUCT_REMAINDER_OUTPUTS,
                    domain: "jolt.trace_domain",
                    point_arity: params.log_t,
                    claim_kind: "virtual",
                },
                InstanceOutput {
                    prefix: "stage2.instruction_lookup.claim_reduction",
                    instance: instruction,
                    eval_source: "stage2.sumcheck",
                    outputs: &STAGE2_INSTRUCTION_INPUTS,
                    domain: "jolt.trace_domain",
                    point_arity: params.log_t,
                    claim_kind: "virtual",
                },
            ],
            ram_rw,
            ram_raf,
            ram_output,
            stage1_ram_address_point: inputs.ram_address.point,
        },
    )?;
    Ok(state)
}

fn append_stage2_output_openings<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage2OutputOpeningSpec<'c, 'a, '_>,
) -> Result<(), MlirError> {
    let mut claims = Vec::new();
    let mut claim_symbols = Vec::new();

    for (index, &oracle) in ["RamVal", "RamRa"].iter().enumerate() {
        let symbol = format!("stage2.ram_read_write.opening.{oracle}");
        let eval = append_sumcheck_eval(
            context,
            module,
            &format!("stage2.ram_read_write.eval.{oracle}"),
            "stage2.sumcheck",
            oracle,
            index,
            spec.ram_rw.1,
        )?;
        claim_symbols.push(symbol.clone());
        claims.push(append_opening_claim(
            context,
            module,
            spec.ram_rw.0,
            eval,
            OpeningClaimSpec {
                symbol: &symbol,
                oracle,
                domain: "jolt.stage2_ram_rw_domain",
                point_arity: stage2_max_rounds(params),
                claim_kind: "virtual",
            },
        )?);
    }
    let ram_inc_point = append_point_slice(
        context,
        module,
        "stage2.ram_read_write.point.RamInc",
        "stage2.ram_read_write.instance",
        params.log_k_ram,
        params.log_t,
        spec.ram_rw.0,
    )?;
    let ram_inc_eval = append_sumcheck_eval(
        context,
        module,
        "stage2.ram_read_write.eval.RamInc",
        "stage2.sumcheck",
        "RamInc",
        2,
        spec.ram_rw.1,
    )?;
    claim_symbols.push("stage2.ram_read_write.opening.RamInc".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        ram_inc_point,
        ram_inc_eval,
        OpeningClaimSpec {
            symbol: "stage2.ram_read_write.opening.RamInc",
            oracle: "RamInc",
            domain: "jolt.trace_domain",
            point_arity: params.log_t,
            claim_kind: "committed",
        },
    )?);

    for output in spec.outputs {
        for (index, &oracle) in output.outputs.iter().enumerate() {
            let symbol = format!("{}.opening.{oracle}", output.prefix);
            let eval = append_sumcheck_eval(
                context,
                module,
                &format!("{}.eval.{oracle}", output.prefix),
                output.eval_source,
                oracle,
                index,
                output.instance.1,
            )?;
            claim_symbols.push(symbol.clone());
            claims.push(append_opening_claim(
                context,
                module,
                output.instance.0,
                eval,
                OpeningClaimSpec {
                    symbol: &symbol,
                    oracle,
                    domain: output.domain,
                    point_arity: output.point_arity,
                    claim_kind: output.claim_kind,
                },
            )?);
        }
    }

    let ram_raf_point = append_point_concat(
        context,
        module,
        "stage2.ram_raf.point.RamRa",
        "address_then_cycle",
        params.log_k_ram + params.log_t,
        &[spec.ram_raf.0, spec.stage1_ram_address_point],
    )?;
    let ram_raf_eval = append_sumcheck_eval(
        context,
        module,
        "stage2.ram_raf.eval.RamRa",
        "stage2.sumcheck",
        "RamRa",
        0,
        spec.ram_raf.1,
    )?;
    claim_symbols.push("stage2.ram_raf.opening.RamRa".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        ram_raf_point,
        ram_raf_eval,
        OpeningClaimSpec {
            symbol: "stage2.ram_raf.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: params.log_k_ram + params.log_t,
            claim_kind: "virtual",
        },
    )?);

    let ram_output_eval = append_sumcheck_eval(
        context,
        module,
        "stage2.ram_output.eval.RamValFinal",
        "stage2.sumcheck",
        "RamValFinal",
        0,
        spec.ram_output.1,
    )?;
    claim_symbols.push("stage2.ram_output.opening.RamValFinal".to_owned());
    claims.push(append_opening_claim(
        context,
        module,
        spec.ram_output.0,
        ram_output_eval,
        OpeningClaimSpec {
            symbol: "stage2.ram_output.opening.RamValFinal",
            oracle: "RamValFinal",
            domain: "jolt.ram_address_domain",
            point_arity: params.log_k_ram,
            claim_kind: "virtual",
        },
    )?);

    let claim_names = claim_symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage2.openings"),
        &[
            ("stage", "@stage2"),
            ("proof_slot", "@stage2.openings"),
            ("policy", r#""jolt_stage2_output_order""#),
            ("count", &int_attr(claims.len())),
            ("ordered_claims", &symbol_array_attr(&claim_names)),
        ],
        &claims,
        &["!piop.opening_batch_type"],
    )?;
    Ok(())
}

fn append_sumcheck_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: SumcheckClaimSpec<'_>,
    input_claim: Value<'c, 'a>,
    inputs: &[Value<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let mut operands = Vec::with_capacity(inputs.len() + 1);
    operands.push(input_claim);
    operands.extend_from_slice(inputs);
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_claim",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("domain", &format!("@{}", spec.domain)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("degree", &int_attr(spec.degree)),
            ("claim", &format!("@{}", spec.claim)),
            ("relation", &format!("@{}", spec.relation)),
        ],
        &operands,
        &["!piop.sumcheck_claim_type"],
    )?;
    first_result(op, "piop.sumcheck_claim")
}

fn append_sumcheck_batch<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    stage: Value<'c, 'a>,
    claims: &[Value<'c, 'a>],
    spec: SumcheckBatchSpec<'_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let mut operands = Vec::with_capacity(claims.len() + 1);
    operands.push(stage);
    operands.extend_from_slice(claims);
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_batch",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("proof_slot", &format!("@{}", spec.proof_slot)),
            ("policy", &format!("\"{}\"", spec.policy)),
            ("count", &int_attr(spec.ordered_claims.len())),
            ("ordered_claims", &symbol_array_attr(spec.ordered_claims)),
            ("claim_label", &format!("\"{}\"", spec.claim_label)),
            ("round_label", &format!("\"{}\"", spec.round_label)),
            ("round_schedule", &spec.round_schedule),
        ],
        &operands,
        &["!piop.sumcheck_batch_type"],
    )?;
    first_result(op, "piop.sumcheck_batch")
}

fn append_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    state: Value<'c, 'a>,
    batch: Value<'c, 'a>,
    spec: SumcheckDriverSpec<'_>,
) -> Result<(Value<'c, 'a>, Value<'c, 'a>, Value<'c, 'a>), MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("proof_slot", &format!("@{}", spec.proof_slot)),
            ("relation", &format!("@{}", spec.relation)),
            ("policy", &format!("\"{}\"", spec.policy)),
            ("round_schedule", &spec.round_schedule),
            ("claim_label", &format!("\"{}\"", spec.claim_label)),
            ("round_label", &format!("\"{}\"", spec.round_label)),
            ("num_rounds", &int_attr(spec.num_rounds)),
            ("degree", &int_attr(spec.degree)),
        ],
        &[state, batch],
        &[
            "!transcript.state_type",
            "!poly.point",
            "!piop.sumcheck_result_type",
            "!piop.sumcheck_proof_type",
        ],
    )?;
    Ok((
        result(op, 0, "piop.sumcheck")?,
        result(op, 1, "piop.sumcheck")?,
        result(op, 2, "piop.sumcheck")?,
    ))
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

fn append_opening_claim<'c, 'a>(
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
            ("claim_kind", &format!("\"{}\"", spec.claim_kind)),
        ],
        &[point, eval],
        &["!piop.opening_claim_type"],
    )?;
    first_result(op, "piop.opening_claim")
}

fn append_point_slice<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    source: &str,
    offset: usize,
    length: usize,
    point: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_slice",
        Some(symbol),
        &[
            ("source", &format!("@{source}")),
            ("offset", &int_attr(offset)),
            ("length", &int_attr(length)),
        ],
        &[point],
        &["!poly.point"],
    )?;
    first_result(op, "poly.point_slice")
}

fn append_point_concat<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    layout: &str,
    arity: usize,
    points: &[Value<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "poly.point_concat",
        Some(symbol),
        &[
            ("layout", &format!("\"{layout}\"")),
            ("arity", &int_attr(arity)),
        ],
        points,
        &["!poly.point"],
    )?;
    first_result(op, "poly.point_concat")
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

fn stage2_max_rounds(params: &JoltProtocolParams) -> usize {
    params.log_t + params.log_k_ram
}

fn int_attr(value: usize) -> String {
    format!("{value} : i64")
}

fn int_attr_signed(value: isize) -> String {
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

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}

#[derive(Clone, Copy)]
struct Stage2OpeningInput<'c, 'a> {
    point: Value<'c, 'a>,
    eval: Value<'c, 'a>,
    claim: Value<'c, 'a>,
}

struct Stage2OpeningInputs<'c, 'a> {
    product: Stage2OpeningInput<'c, 'a>,
    should_branch: Stage2OpeningInput<'c, 'a>,
    should_jump: Stage2OpeningInput<'c, 'a>,
    ram_read_value: Stage2OpeningInput<'c, 'a>,
    ram_write_value: Stage2OpeningInput<'c, 'a>,
    lookup_output: Stage2OpeningInput<'c, 'a>,
    left_lookup_operand: Stage2OpeningInput<'c, 'a>,
    right_lookup_operand: Stage2OpeningInput<'c, 'a>,
    left_instruction_input: Stage2OpeningInput<'c, 'a>,
    right_instruction_input: Stage2OpeningInput<'c, 'a>,
    ram_address: Stage2OpeningInput<'c, 'a>,
}

#[derive(Clone, Copy)]
struct Stage2UniskipOutput<'c, 'a> {
    opening: Value<'c, 'a>,
    eval: Value<'c, 'a>,
}

struct Stage2BatchedSumcheckInputs<'c, 'a, 'b> {
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    openings: &'b Stage2OpeningInputs<'c, 'a>,
    uniskip: Stage2UniskipOutput<'c, 'a>,
    ram_read_write_gamma: Value<'c, 'a>,
    instruction_lookup_gamma: Value<'c, 'a>,
}

struct RelationSpec<'a> {
    symbol: &'a str,
    kind: &'a str,
    domain: &'a str,
    num_rounds: usize,
    degree: usize,
    output_count: usize,
}

struct SumcheckClaimSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    domain: &'a str,
    num_rounds: usize,
    degree: usize,
    claim: &'a str,
    relation: &'a str,
}

struct SumcheckBatchSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    proof_slot: &'a str,
    policy: &'a str,
    ordered_claims: &'a [&'a str],
    claim_label: &'a str,
    round_label: &'a str,
    round_schedule: String,
}

struct SumcheckDriverSpec<'a> {
    symbol: &'a str,
    stage: &'a str,
    proof_slot: &'a str,
    relation: &'a str,
    policy: &'a str,
    round_schedule: String,
    claim_label: &'a str,
    round_label: &'a str,
    num_rounds: usize,
    degree: usize,
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

struct OpeningClaimSpec<'a> {
    symbol: &'a str,
    oracle: &'a str,
    domain: &'a str,
    point_arity: usize,
    claim_kind: &'a str,
}

struct Stage2OutputOpeningSpec<'c, 'a, 'b> {
    outputs: &'b [InstanceOutput<'c, 'a, 'b>],
    ram_rw: (Value<'c, 'a>, Value<'c, 'a>),
    ram_raf: (Value<'c, 'a>, Value<'c, 'a>),
    ram_output: (Value<'c, 'a>, Value<'c, 'a>),
    stage1_ram_address_point: Value<'c, 'a>,
}

struct InstanceOutput<'c, 'a, 'b> {
    prefix: &'b str,
    instance: (Value<'c, 'a>, Value<'c, 'a>),
    eval_source: &'b str,
    outputs: &'b [&'b str],
    domain: &'b str,
    point_arity: usize,
    claim_kind: &'b str,
}
