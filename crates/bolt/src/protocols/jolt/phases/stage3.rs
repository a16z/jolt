use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationRef;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Party, Protocol, Role};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{
    operation_name, symbol_attr, verify_compute_schema, verify_party_schema,
    verify_protocol_schema, SchemaError,
};

use super::super::oracles;
use super::super::params::JoltProtocolParams;
use super::lowering::{
    copy_attrs, field_lowering_attrs as field_compute_attrs, string_attr,
    transcript_squeeze_compute_result_types, transcript_squeeze_protocol_result_type,
};

const SPARTAN_SHIFT_DEGREE: usize = 2;
const INSTRUCTION_INPUT_DEGREE: usize = 3;
const REGISTERS_CLAIM_REDUCTION_DEGREE: usize = 2;
const FIELD_REG_CLAIM_REDUCTION_DEGREE: usize = 2;
const STAGE3_BATCHED_DEGREE: usize = 3;

const STAGE3_SHIFT_INPUTS: [&str; 4] = [
    "NextUnexpandedPC",
    "NextPC",
    "NextIsVirtual",
    "NextIsFirstInSequence",
];
const STAGE3_SHIFT_OUTPUTS: [&str; 5] = [
    "UnexpandedPC",
    "PC",
    "OpFlagVirtualInstruction",
    "OpFlagIsFirstInSequence",
    "InstructionFlagIsNoop",
];
const STAGE3_INSTRUCTION_INPUT_OUTPUTS: [&str; 8] = [
    "InstructionFlagLeftOperandIsRs1Value",
    "Rs1Value",
    "InstructionFlagLeftOperandIsPC",
    "UnexpandedPC",
    "InstructionFlagRightOperandIsRs2Value",
    "Rs2Value",
    "InstructionFlagRightOperandIsImm",
    "Imm",
];
const STAGE3_REGISTER_INPUTS: [&str; 3] = ["RdWriteValue", "Rs1Value", "Rs2Value"];
const STAGE3_FIELD_REG_INPUTS: [&str; 3] = ["FieldRdWriteValue", "FieldRs1Value", "FieldRs2Value"];

pub fn build_stage3_protocol<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> Result<BoltModule<'c, Protocol>, MlirError> {
    let module = context.new_module::<Protocol>("jolt.stage3", None);
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
        Some("jolt.stage3"),
        &[("roles", r#"["prover", "verifier"]"#)],
    )?;
    append_stage3_oracles(context, &module)?;
    append_stage3_relations(context, &module, params)?;
    let inputs = append_stage3_opening_inputs(context, &module, params)?;

    let fs = context.append_typed_op(
        &module,
        "transcript.state",
        Some("fs_after_stage2"),
        &[("scheme", "@blake2b_transcript")],
        &[],
        &["!transcript.state_type"],
    )?;
    let state = first_result(fs, "transcript.state")?;
    let stage = context.append_typed_op(
        &module,
        "piop.stage",
        Some("stage3"),
        &[
            ("name", r#""shift_instruction_input_and_registers""#),
            ("order", "3 : i64"),
            ("roles", r#"["prover", "verifier"]"#),
        ],
        &[],
        &["!piop.stage_type"],
    )?;
    let stage = first_result(stage, "piop.stage")?;

    let (state, shift_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage3.spartan_shift.gamma",
        "spartan_shift_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, instruction_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage3.instruction_input.gamma",
        "instruction_input_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, registers_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage3.registers.gamma",
        "registers_gamma",
        "challenge_scalar",
        1,
    )?;
    let (state, field_reg_gamma) = append_transcript_squeeze(
        context,
        &module,
        state,
        "stage3.field_reg.gamma",
        "field_reg_gamma",
        "challenge_scalar",
        1,
    )?;
    let _state = append_stage3_batched_sumcheck(
        context,
        &module,
        params,
        Stage3BatchedSumcheckInputs {
            state,
            stage,
            openings: &inputs,
            shift_gamma,
            instruction_gamma,
            registers_gamma,
            field_reg_gamma,
        },
    )?;

    verify_module(&module)?;
    verify_protocol_schema(&module)?;
    Ok(module)
}

pub fn lower_stage3_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    verify_party_schema(module)?;
    let role = module
        .role()
        .ok_or_else(|| schema_error("stage3 lowering requires party role"))?;
    let params = stage_params(module)?;
    let compute = context.new_module::<Compute>(&module.name(), Some(role.clone()));
    context.append_op_with_owned_attrs(
        &compute,
        "compute.params",
        Some("jolt.compute_params"),
        &[
            ("field".to_owned(), symbol_ref(&params.field)),
            ("pcs".to_owned(), symbol_ref(&params.pcs)),
            ("transcript".to_owned(), symbol_ref(&params.transcript)),
        ],
    )?;
    context.append_op(
        &compute,
        "compute.function",
        Some("jolt.stage3"),
        &[("source", "@jolt.stage3")],
    )?;

    let mut value_map = BTreeMap::new();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        match operation_name(op).as_str() {
            "piop.relation" => {
                let attrs = copy_attrs(
                    op,
                    &["kind", "domain", "num_rounds", "degree", "output_count"],
                )?;
                let symbol = string_attr(op, "sym_name")?;
                context.append_op_with_owned_attrs(
                    &compute,
                    "compute.relation",
                    Some(&symbol),
                    &attrs,
                )?;
            }
            "transcript.state" => {
                let attrs = copy_attrs(op, &["scheme"])?;
                let symbol = string_attr(op, "sym_name")?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.transcript_init",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!compute.transcript_state"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "transcript.absorb_bytes" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["label", "payload"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.transcript_absorb_bytes",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.transcript_state"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "transcript.squeeze" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["label", "kind", "count"])?;
                let result_types = transcript_squeeze_compute_result_types(op)?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.transcript_squeeze",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &result_types,
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
                insert_result_mapping(&mut value_map, op, operation, 1, 1)?;
            }
            "field.const" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["field", "value"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.field_const",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "field.zero" | "field.one" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["field"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    &format!("compute.{}", operation_name(op).replace('.', "_")),
                    Some(&symbol),
                    &attrs,
                    &[],
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "field.add" | "field.sub" | "field.mul" | "field.neg" | "field.pow" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = field_compute_attrs(op)?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    &format!("compute.{}", operation_name(op).replace('.', "_")),
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "poly.lagrange_basis_eval" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["domain_start", "domain_size", "index"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.poly_lagrange_basis_eval",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.opening_input" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "source_stage",
                        "source_claim",
                        "oracle",
                        "domain",
                        "point_arity",
                        "claim_kind",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.opening_input",
                    Some(&symbol),
                    &attrs,
                    &[],
                    &[
                        "!compute.point",
                        "!compute.field_value",
                        "!compute.opening_claim_type",
                    ],
                )?;
                for index in 0..3 {
                    insert_result_mapping(&mut value_map, op, operation, index, index)?;
                }
            }
            "poly.point_slice" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["source", "offset", "length"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.point_slice",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.point"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "poly.point_concat" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["layout", "arity"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.point_concat",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.point"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_claim" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "domain",
                        "num_rounds",
                        "degree",
                        "claim",
                        "relation",
                    ],
                )?;
                let target_op = match &role {
                    Role::Prover => "compute.sumcheck_claim",
                    Role::Verifier => "compute.sumcheck_verify_claim",
                };
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    target_op,
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.sumcheck_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_batch" => {
                let operands = lowered_operands(op, &value_map, 1)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "proof_slot",
                        "policy",
                        "count",
                        "ordered_claims",
                        "claim_label",
                        "round_label",
                        "round_schedule",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_batch",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.sumcheck_batch_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "stage",
                        "proof_slot",
                        "relation",
                        "policy",
                        "round_schedule",
                        "claim_label",
                        "round_label",
                        "num_rounds",
                        "degree",
                    ],
                )?;
                let target_op = match &role {
                    Role::Prover => "compute.sumcheck_driver",
                    Role::Verifier => "compute.sumcheck_verify",
                };
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    target_op,
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[
                        "!compute.transcript_state",
                        "!compute.point",
                        "!compute.sumcheck_result_type",
                        "!compute.sumcheck_proof_type",
                    ],
                )?;
                for index in 0..4 {
                    insert_result_mapping(&mut value_map, op, operation, index, index)?;
                }
            }
            "piop.sumcheck_eval" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["source", "name", "index", "oracle"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_eval",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_instance_result" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "source",
                        "claim",
                        "relation",
                        "index",
                        "point_arity",
                        "num_rounds",
                        "round_offset",
                        "point_order",
                        "degree",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_instance_result",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.point", "!compute.sumcheck_result_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
                insert_result_mapping(&mut value_map, op, operation, 1, 1)?;
            }
            "piop.opening_claim" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["oracle", "domain", "point_arity", "claim_kind"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.opening_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.opening_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.opening_claim_equal" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["mode"])?;
                let _operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.opening_claim_equal",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[],
                )?;
            }
            "piop.opening_batch" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &["stage", "proof_slot", "policy", "count", "ordered_claims"],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.opening_batch",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.opening_batch_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            _ => {}
        }
    }

    verify_module(&compute)?;
    verify_compute_schema(&compute)?;
    Ok(compute)
}

fn append_stage3_oracles<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
) -> Result<(), MlirError> {
    let mut trace_oracles = BTreeSet::new();
    trace_oracles.extend(STAGE3_SHIFT_INPUTS);
    trace_oracles.extend(STAGE3_SHIFT_OUTPUTS);
    trace_oracles.extend(STAGE3_INSTRUCTION_INPUT_OUTPUTS);
    trace_oracles.extend(STAGE3_REGISTER_INPUTS);
    trace_oracles.extend(STAGE3_FIELD_REG_INPUTS);
    trace_oracles.extend([
        "LeftInstructionInput",
        "RightInstructionInput",
        "NextIsNoop",
    ]);
    for oracle in trace_oracles {
        append_virtual_oracle(context, module, oracle, "jolt.trace_domain")?;
    }
    Ok(())
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

fn append_stage3_relations<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<(), MlirError> {
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage3.spartan_shift",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: SPARTAN_SHIFT_DEGREE,
            output_count: STAGE3_SHIFT_OUTPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage3.instruction_input",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: INSTRUCTION_INPUT_DEGREE,
            output_count: STAGE3_INSTRUCTION_INPUT_OUTPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage3.registers_claim_reduction",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: REGISTERS_CLAIM_REDUCTION_DEGREE,
            output_count: STAGE3_REGISTER_INPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage3.field_reg_claim_reduction",
            kind: "sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: FIELD_REG_CLAIM_REDUCTION_DEGREE,
            output_count: STAGE3_FIELD_REG_INPUTS.len(),
        },
    )?;
    append_relation(
        context,
        module,
        RelationSpec {
            symbol: "jolt.stage3.batched",
            kind: "batched_sumcheck",
            domain: "jolt.trace_domain",
            num_rounds: params.log_t,
            degree: STAGE3_BATCHED_DEGREE,
            output_count: stage3_output_count(),
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

fn append_stage3_opening_inputs<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
) -> Result<Stage3OpeningInputs<'c, 'a>, MlirError> {
    Ok(Stage3OpeningInputs {
        next_unexpanded_pc: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.NextUnexpandedPC",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.NextUnexpandedPC",
                oracle: "NextUnexpandedPC",
            },
        )?,
        next_pc: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.NextPC",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.NextPC",
                oracle: "NextPC",
            },
        )?,
        next_is_virtual: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.NextIsVirtual",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.NextIsVirtual",
                oracle: "NextIsVirtual",
            },
        )?,
        next_is_first_in_sequence: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.NextIsFirstInSequence",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.NextIsFirstInSequence",
                oracle: "NextIsFirstInSequence",
            },
        )?,
        product_next_is_noop: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage2.product_virtual.NextIsNoop",
                source_stage: "stage2",
                source_claim: "stage2.product_virtual.remainder.opening.NextIsNoop",
                oracle: "NextIsNoop",
            },
        )?,
        product_left_instruction_input: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage2.product_virtual.LeftInstructionInput",
                source_stage: "stage2",
                source_claim: "stage2.product_virtual.remainder.opening.LeftInstructionInput",
                oracle: "LeftInstructionInput",
            },
        )?,
        product_right_instruction_input: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage2.product_virtual.RightInstructionInput",
                source_stage: "stage2",
                source_claim: "stage2.product_virtual.remainder.opening.RightInstructionInput",
                oracle: "RightInstructionInput",
            },
        )?,
        instruction_left_instruction_input: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage2.instruction_lookup.LeftInstructionInput",
                source_stage: "stage2",
                source_claim:
                    "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput",
                oracle: "LeftInstructionInput",
            },
        )?,
        instruction_right_instruction_input: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage2.instruction_lookup.RightInstructionInput",
                source_stage: "stage2",
                source_claim:
                    "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput",
                oracle: "RightInstructionInput",
            },
        )?,
        rd_write_value: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.RdWriteValue",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.RdWriteValue",
                oracle: "RdWriteValue",
            },
        )?,
        rs1_value: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.Rs1Value",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.Rs1Value",
                oracle: "Rs1Value",
            },
        )?,
        rs2_value: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.Rs2Value",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.Rs2Value",
                oracle: "Rs2Value",
            },
        )?,
        field_rd_write_value: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.FieldRdWriteValue",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.FieldRdWriteValue",
                oracle: "FieldRdWriteValue",
            },
        )?,
        field_rs1_value: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.FieldRs1Value",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.FieldRs1Value",
                oracle: "FieldRs1Value",
            },
        )?,
        field_rs2_value: append_stage_input(
            context,
            module,
            params,
            StageOpeningInputSpec {
                symbol: "stage3.input.stage1.FieldRs2Value",
                source_stage: "stage1",
                source_claim: "stage1.outer_remaining.opening.FieldRs2Value",
                oracle: "FieldRs2Value",
            },
        )?,
    })
}

fn append_stage_input<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: StageOpeningInputSpec<'_>,
) -> Result<Stage3OpeningInput<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.opening_input",
        Some(spec.symbol),
        &[
            ("source_stage", &format!("@{}", spec.source_stage)),
            ("source_claim", &format!("@{}", spec.source_claim)),
            ("oracle", &format!("@{}", spec.oracle)),
            ("domain", "@jolt.trace_domain"),
            ("point_arity", &int_attr(params.log_t)),
            ("claim_kind", r#""virtual""#),
        ],
        &[],
        &["!poly.point", "!field.scalar", "!piop.opening_claim_type"],
    )?;
    Ok(Stage3OpeningInput {
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

fn append_field_one<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.one",
        Some(symbol),
        &[("field", "@bn254_fr")],
        &[],
        &["!field.scalar"],
    )?;
    first_result(op, "field.one")
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

fn append_field_sub<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    lhs: Value<'c, 'a>,
    rhs: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    append_field_binary(context, module, "field.sub", symbol, lhs, rhs)
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

fn append_field_pow<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    symbol: &str,
    base: Value<'c, 'a>,
    exponent: usize,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "field.pow",
        Some(symbol),
        &[("exponent", &int_attr(exponent))],
        &[base],
        &["!field.scalar"],
    )?;
    first_result(op, "field.pow")
}

fn append_opening_claim_equal<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
    symbol: &str,
    left: Value<'c, '_>,
    right: Value<'c, '_>,
) -> Result<(), MlirError> {
    let _operation = context.append_typed_op(
        module,
        "piop.opening_claim_equal",
        Some(symbol),
        &[("mode", r#""point_and_eval""#)],
        &[left, right],
        &[],
    )?;
    Ok(())
}

fn append_stage3_batched_sumcheck<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    params: &JoltProtocolParams,
    spec: Stage3BatchedSumcheckInputs<'c, 'a, '_>,
) -> Result<Value<'c, 'a>, MlirError> {
    let inputs = spec.openings;
    let shift_gamma2 = append_field_pow(
        context,
        module,
        "stage3.spartan_shift.gamma2",
        spec.shift_gamma,
        2,
    )?;
    let shift_gamma3 = append_field_mul(
        context,
        module,
        "stage3.spartan_shift.gamma3",
        shift_gamma2,
        spec.shift_gamma,
    )?;
    let shift_gamma4 = append_field_mul(
        context,
        module,
        "stage3.spartan_shift.gamma4",
        shift_gamma2,
        shift_gamma2,
    )?;
    let one = append_field_one(context, module, "stage3.field.one")?;
    let next_pc_term = append_field_mul(
        context,
        module,
        "stage3.spartan_shift.term.NextPC",
        spec.shift_gamma,
        inputs.next_pc.eval,
    )?;
    let next_virtual_term = append_field_mul(
        context,
        module,
        "stage3.spartan_shift.term.NextIsVirtual",
        shift_gamma2,
        inputs.next_is_virtual.eval,
    )?;
    let next_first_term = append_field_mul(
        context,
        module,
        "stage3.spartan_shift.term.NextIsFirstInSequence",
        shift_gamma3,
        inputs.next_is_first_in_sequence.eval,
    )?;
    let one_minus_noop = append_field_sub(
        context,
        module,
        "stage3.spartan_shift.one_minus.NextIsNoop",
        one,
        inputs.product_next_is_noop.eval,
    )?;
    let next_noop_term = append_field_mul(
        context,
        module,
        "stage3.spartan_shift.term.NextIsNoop",
        shift_gamma4,
        one_minus_noop,
    )?;
    let shift_sum0 = append_field_add(
        context,
        module,
        "stage3.spartan_shift.partial.NextUnexpandedPCNextPC",
        inputs.next_unexpanded_pc.eval,
        next_pc_term,
    )?;
    let shift_sum1 = append_field_add(
        context,
        module,
        "stage3.spartan_shift.partial.NextIsVirtual",
        shift_sum0,
        next_virtual_term,
    )?;
    let shift_sum2 = append_field_add(
        context,
        module,
        "stage3.spartan_shift.partial.NextIsFirstInSequence",
        shift_sum1,
        next_first_term,
    )?;
    let shift_claim = append_field_add(
        context,
        module,
        "stage3.spartan_shift.claim_expr",
        shift_sum2,
        next_noop_term,
    )?;
    append_opening_claim_equal(
        context,
        module,
        "stage3.instruction_input.left_claim_consistency",
        inputs.product_left_instruction_input.claim,
        inputs.instruction_left_instruction_input.claim,
    )?;
    append_opening_claim_equal(
        context,
        module,
        "stage3.instruction_input.right_claim_consistency",
        inputs.product_right_instruction_input.claim,
        inputs.instruction_right_instruction_input.claim,
    )?;
    let instruction_left_term = append_field_mul(
        context,
        module,
        "stage3.instruction_input.term.LeftInstructionInput",
        spec.instruction_gamma,
        inputs.product_left_instruction_input.eval,
    )?;
    let instruction_claim = append_field_add(
        context,
        module,
        "stage3.instruction_input.claim_expr",
        inputs.product_right_instruction_input.eval,
        instruction_left_term,
    )?;
    let registers_gamma2 = append_field_pow(
        context,
        module,
        "stage3.registers.gamma2",
        spec.registers_gamma,
        2,
    )?;
    let rs1_term = append_field_mul(
        context,
        module,
        "stage3.registers.term.Rs1Value",
        spec.registers_gamma,
        inputs.rs1_value.eval,
    )?;
    let rs2_term = append_field_mul(
        context,
        module,
        "stage3.registers.term.Rs2Value",
        registers_gamma2,
        inputs.rs2_value.eval,
    )?;
    let registers_sum = append_field_add(
        context,
        module,
        "stage3.registers.partial.RdWriteValueRs1Value",
        inputs.rd_write_value.eval,
        rs1_term,
    )?;
    let registers_claim = append_field_add(
        context,
        module,
        "stage3.registers.claim_expr",
        registers_sum,
        rs2_term,
    )?;
    let field_reg_gamma2 = append_field_pow(
        context,
        module,
        "stage3.field_reg.gamma2",
        spec.field_reg_gamma,
        2,
    )?;
    let field_rs1_term = append_field_mul(
        context,
        module,
        "stage3.field_reg.term.FieldRs1Value",
        spec.field_reg_gamma,
        inputs.field_rs1_value.eval,
    )?;
    let field_rs2_term = append_field_mul(
        context,
        module,
        "stage3.field_reg.term.FieldRs2Value",
        field_reg_gamma2,
        inputs.field_rs2_value.eval,
    )?;
    let field_reg_sum = append_field_add(
        context,
        module,
        "stage3.field_reg.partial.FieldRdWriteValueFieldRs1Value",
        inputs.field_rd_write_value.eval,
        field_rs1_term,
    )?;
    let field_reg_claim = append_field_add(
        context,
        module,
        "stage3.field_reg.claim_expr",
        field_reg_sum,
        field_rs2_term,
    )?;

    let claims = [
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage3.spartan_shift.input",
                stage: "stage3",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: SPARTAN_SHIFT_DEGREE,
                claim: "stage3.spartan_shift.weighted_next_values",
                relation: "jolt.stage3.spartan_shift",
            },
            shift_claim,
            &[
                inputs.next_unexpanded_pc.claim,
                inputs.next_pc.claim,
                inputs.next_is_virtual.claim,
                inputs.next_is_first_in_sequence.claim,
                inputs.product_next_is_noop.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage3.instruction_input.input",
                stage: "stage3",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: INSTRUCTION_INPUT_DEGREE,
                claim: "stage3.instruction_input.weighted_inputs",
                relation: "jolt.stage3.instruction_input",
            },
            instruction_claim,
            &[
                inputs.product_right_instruction_input.claim,
                inputs.product_left_instruction_input.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage3.registers_claim_reduction.input",
                stage: "stage3",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: REGISTERS_CLAIM_REDUCTION_DEGREE,
                claim: "stage3.registers.weighted_register_values",
                relation: "jolt.stage3.registers_claim_reduction",
            },
            registers_claim,
            &[
                inputs.rd_write_value.claim,
                inputs.rs1_value.claim,
                inputs.rs2_value.claim,
            ],
        )?,
        append_sumcheck_claim(
            context,
            module,
            SumcheckClaimSpec {
                symbol: "stage3.field_reg_claim_reduction.input",
                stage: "stage3",
                domain: "jolt.trace_domain",
                num_rounds: params.log_t,
                degree: FIELD_REG_CLAIM_REDUCTION_DEGREE,
                claim: "stage3.field_reg.weighted_field_reg_values",
                relation: "jolt.stage3.field_reg_claim_reduction",
            },
            field_reg_claim,
            &[
                inputs.field_rd_write_value.claim,
                inputs.field_rs1_value.claim,
                inputs.field_rs2_value.claim,
            ],
        )?,
    ];
    let batch = append_sumcheck_batch(
        context,
        module,
        spec.stage,
        &claims,
        SumcheckBatchSpec {
            symbol: "stage3.batch",
            stage: "stage3",
            proof_slot: "stage3.sumcheck",
            policy: "jolt_core_stage3_aligned",
            ordered_claims: &[
                "stage3.spartan_shift.input",
                "stage3.instruction_input.input",
                "stage3.registers_claim_reduction.input",
                "stage3.field_reg_claim_reduction.input",
            ],
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            round_schedule: format!("[{}]", params.log_t),
        },
    )?;
    let (state, point, result_value) = append_sumcheck(
        context,
        module,
        spec.state,
        batch,
        SumcheckDriverSpec {
            symbol: "stage3.sumcheck",
            stage: "stage3",
            proof_slot: "stage3.sumcheck",
            relation: "jolt.stage3.batched",
            policy: "jolt_core_stage3_aligned",
            round_schedule: format!("[{}]", params.log_t),
            claim_label: "sumcheck_claim",
            round_label: "sumcheck_poly",
            num_rounds: params.log_t,
            degree: STAGE3_BATCHED_DEGREE,
        },
    )?;

    let shift = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage3.spartan_shift.instance",
            source: "stage3.sumcheck",
            claim: "stage3.spartan_shift.input",
            relation: "jolt.stage3.spartan_shift",
            index: 0,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: 0,
            point_order: "reverse",
            degree: SPARTAN_SHIFT_DEGREE,
        },
        point,
        result_value,
    )?;
    let instruction = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage3.instruction_input.instance",
            source: "stage3.sumcheck",
            claim: "stage3.instruction_input.input",
            relation: "jolt.stage3.instruction_input",
            index: 1,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: 0,
            point_order: "reverse",
            degree: INSTRUCTION_INPUT_DEGREE,
        },
        point,
        result_value,
    )?;
    let registers = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage3.registers_claim_reduction.instance",
            source: "stage3.sumcheck",
            claim: "stage3.registers_claim_reduction.input",
            relation: "jolt.stage3.registers_claim_reduction",
            index: 2,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: 0,
            point_order: "reverse",
            degree: REGISTERS_CLAIM_REDUCTION_DEGREE,
        },
        point,
        result_value,
    )?;
    let field_reg = append_sumcheck_instance_result(
        context,
        module,
        SumcheckInstanceResultSpec {
            symbol: "stage3.field_reg_claim_reduction.instance",
            source: "stage3.sumcheck",
            claim: "stage3.field_reg_claim_reduction.input",
            relation: "jolt.stage3.field_reg_claim_reduction",
            index: 3,
            point_arity: params.log_t,
            num_rounds: params.log_t,
            round_offset: 0,
            point_order: "reverse",
            degree: FIELD_REG_CLAIM_REDUCTION_DEGREE,
        },
        point,
        result_value,
    )?;
    append_stage3_output_openings(
        context,
        module,
        &[
            InstanceOutput {
                prefix: "stage3.spartan_shift",
                instance: shift,
                outputs: &STAGE3_SHIFT_OUTPUTS,
                degree_offset: 0,
            },
            InstanceOutput {
                prefix: "stage3.instruction_input",
                instance: instruction,
                outputs: &STAGE3_INSTRUCTION_INPUT_OUTPUTS,
                degree_offset: STAGE3_SHIFT_OUTPUTS.len(),
            },
            InstanceOutput {
                prefix: "stage3.registers_claim_reduction",
                instance: registers,
                outputs: &STAGE3_REGISTER_INPUTS,
                degree_offset: STAGE3_SHIFT_OUTPUTS.len() + STAGE3_INSTRUCTION_INPUT_OUTPUTS.len(),
            },
            InstanceOutput {
                prefix: "stage3.field_reg_claim_reduction",
                instance: field_reg,
                outputs: &STAGE3_FIELD_REG_INPUTS,
                degree_offset: STAGE3_SHIFT_OUTPUTS.len()
                    + STAGE3_INSTRUCTION_INPUT_OUTPUTS.len()
                    + STAGE3_REGISTER_INPUTS.len(),
            },
        ],
        params.log_t,
    )?;
    Ok(state)
}

fn append_stage3_output_openings<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    outputs: &[InstanceOutput<'c, 'a, '_>],
    point_arity: usize,
) -> Result<(), MlirError> {
    let mut claims = Vec::new();
    let mut claim_symbols = Vec::new();

    for output in outputs {
        for (index, &oracle) in output.outputs.iter().enumerate() {
            let symbol = format!("{}.opening.{oracle}", output.prefix);
            let eval = append_sumcheck_eval(
                context,
                module,
                &format!("{}.eval.{oracle}", output.prefix),
                "stage3.sumcheck",
                oracle,
                output.degree_offset + index,
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
                    domain: "jolt.trace_domain",
                    point_arity,
                    claim_kind: "virtual",
                },
            )?);
        }
    }

    let claim_names = claim_symbols.iter().map(String::as_str).collect::<Vec<_>>();
    let _batch = context.append_typed_op(
        module,
        "piop.opening_batch",
        Some("stage3.openings"),
        &[
            ("stage", "@stage3"),
            ("proof_slot", "@stage3.openings"),
            ("policy", r#""jolt_stage3_output_order""#),
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

#[derive(Clone, Debug)]
struct StageParamsAst {
    field: String,
    pcs: String,
    transcript: String,
}

fn stage_params(module: &BoltModule<'_, Party>) -> Result<StageParamsAst, MlirError> {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if operation_name(op) == "protocol.params" {
            return Ok(StageParamsAst {
                field: symbol_attr(op, "field")?,
                pcs: symbol_attr(op, "pcs")?,
                transcript: symbol_attr(op, "transcript")?,
            });
        }
    }
    Err(schema_error("stage3 lowering requires protocol.params"))
}

fn operation_result_key_at(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, MlirError> {
    let result = operation.result(index).map_err(|_| {
        schema_error(format!(
            "{} requires result {index}",
            operation_name(operation)
        ))
    })?;
    result_key(result.owner(), result.result_number())
}

fn result_key(operation: OperationRef<'_, '_>, result_number: usize) -> Result<String, MlirError> {
    Ok(format!(
        "{}#{result_number}",
        string_attr(operation, "sym_name")?
    ))
}

fn operand_key(operation: OperationRef<'_, '_>, index: usize) -> Result<String, MlirError> {
    let operand = operation.operand(index).map_err(|_| {
        schema_error(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        schema_error(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    result_key(owner.owner(), owner.result_number()).map_err(|_| {
        schema_error(format!(
            "{} operand {index} owner missing sym_name",
            operation_name(operation)
        ))
    })
}

fn lowered_operands<'c, 'a>(
    operation: OperationRef<'_, '_>,
    value_map: &BTreeMap<String, Value<'c, 'a>>,
    start_index: usize,
) -> Result<Vec<Value<'c, 'a>>, MlirError> {
    (start_index..operation.operand_count())
        .map(|index| {
            let key = operand_key(operation, index)?;
            value_map.get(&key).copied().ok_or_else(|| {
                schema_error(format!(
                    "{} operand {index} was not lowered",
                    operation_name(operation)
                ))
            })
        })
        .collect()
}

fn insert_result_mapping<'c, 'a>(
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    source: OperationRef<'_, '_>,
    target: OperationRef<'c, 'a>,
    source_index: usize,
    target_index: usize,
) -> Result<(), MlirError> {
    let key = operation_result_key_at(source, source_index)?;
    let value = target.result(target_index).map(Into::into).map_err(|_| {
        schema_error(format!(
            "{} requires result {target_index}",
            operation_name(target)
        ))
    })?;
    let inserted = value_map.insert(key, value);
    debug_assert!(inserted.is_none());
    Ok(())
}

fn symbol_ref(symbol: &str) -> String {
    format!("@{symbol}")
}

fn stage3_output_count() -> usize {
    STAGE3_SHIFT_OUTPUTS.len()
        + STAGE3_INSTRUCTION_INPUT_OUTPUTS.len()
        + STAGE3_REGISTER_INPUTS.len()
        + STAGE3_FIELD_REG_INPUTS.len()
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

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}

#[derive(Clone, Copy)]
struct Stage3OpeningInput<'c, 'a> {
    eval: Value<'c, 'a>,
    claim: Value<'c, 'a>,
}

struct Stage3OpeningInputs<'c, 'a> {
    next_unexpanded_pc: Stage3OpeningInput<'c, 'a>,
    next_pc: Stage3OpeningInput<'c, 'a>,
    next_is_virtual: Stage3OpeningInput<'c, 'a>,
    next_is_first_in_sequence: Stage3OpeningInput<'c, 'a>,
    product_next_is_noop: Stage3OpeningInput<'c, 'a>,
    product_left_instruction_input: Stage3OpeningInput<'c, 'a>,
    product_right_instruction_input: Stage3OpeningInput<'c, 'a>,
    instruction_left_instruction_input: Stage3OpeningInput<'c, 'a>,
    instruction_right_instruction_input: Stage3OpeningInput<'c, 'a>,
    rd_write_value: Stage3OpeningInput<'c, 'a>,
    rs1_value: Stage3OpeningInput<'c, 'a>,
    rs2_value: Stage3OpeningInput<'c, 'a>,
    field_rd_write_value: Stage3OpeningInput<'c, 'a>,
    field_rs1_value: Stage3OpeningInput<'c, 'a>,
    field_rs2_value: Stage3OpeningInput<'c, 'a>,
}

struct StageOpeningInputSpec<'a> {
    symbol: &'a str,
    source_stage: &'a str,
    source_claim: &'a str,
    oracle: &'a str,
}

struct Stage3BatchedSumcheckInputs<'c, 'a, 'b> {
    state: Value<'c, 'a>,
    stage: Value<'c, 'a>,
    openings: &'b Stage3OpeningInputs<'c, 'a>,
    shift_gamma: Value<'c, 'a>,
    instruction_gamma: Value<'c, 'a>,
    registers_gamma: Value<'c, 'a>,
    field_reg_gamma: Value<'c, 'a>,
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

struct InstanceOutput<'c, 'a, 'b> {
    prefix: &'b str,
    instance: (Value<'c, 'a>, Value<'c, 'a>),
    outputs: &'b [&'b str],
    degree_offset: usize,
}
