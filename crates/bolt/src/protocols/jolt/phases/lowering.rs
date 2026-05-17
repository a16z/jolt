use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationResult;
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::Value;

use crate::ir::{string_attribute_value, BoltModule, Compute, Party, Role};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{operation_name, verify_compute_schema, verify_party_schema, SchemaError};

pub(super) fn copy_attrs(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
) -> Result<Vec<(String, String)>, MlirError> {
    attrs
        .iter()
        .filter_map(|attr| {
            operation
                .attribute(attr)
                .ok()
                .map(|value| Ok(((*attr).to_owned(), value.to_string())))
        })
        .collect()
}

pub(super) fn string_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, MlirError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| {
            schema_error(format!(
                "{} attr `{attr}` is not a string",
                operation_name(operation)
            ))
        })
}

pub(super) fn transcript_squeeze_protocol_result_type(
    kind: &str,
) -> Result<&'static str, MlirError> {
    transcript_squeeze_value_type(kind, "!poly.point", "!field.scalar")
}

pub(super) fn transcript_squeeze_compute_result_types(
    operation: OperationRef<'_, '_>,
) -> Result<[&'static str; 2], MlirError> {
    Ok([
        "!compute.transcript_state",
        transcript_squeeze_value_type(
            string_attr(operation, "kind")?.as_str(),
            "!compute.point",
            "!compute.field_value",
        )?,
    ])
}

pub(super) fn transcript_squeeze_cpu_result_types(
    operation: OperationRef<'_, '_>,
) -> Result<[&'static str; 2], MlirError> {
    Ok([
        "!cpu.transcript_state",
        transcript_squeeze_value_type(
            string_attr(operation, "kind")?.as_str(),
            "!cpu.point",
            "!cpu.field_value",
        )?,
    ])
}

pub(super) fn field_lowering_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    match operation_name(operation).as_str() {
        "field.pow" | "compute.field_pow" => copy_attrs(operation, &["exponent"]),
        "poly.lagrange_basis_eval" | "compute.poly_lagrange_basis_eval" => {
            copy_attrs(operation, &["domain_start", "domain_size", "index"])
        }
        _ => Ok(Vec::new()),
    }
}

pub(super) fn lower_party_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
    function_symbol: &str,
    source_symbol: &str,
    stage_label: &str,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    verify_party_schema(module)?;
    let role = module
        .role()
        .ok_or_else(|| schema_error(format!("{stage_label} lowering requires party role")))?;
    let compute = context.new_module::<Compute>(&module.name(), Some(role.clone()));
    let params_attrs = compute_params_attrs(module)?;
    context.append_op_with_owned_attrs(
        &compute,
        "compute.params",
        Some("jolt.compute_params"),
        &params_attrs,
    )?;
    context.append_op(
        &compute,
        "compute.function",
        Some(function_symbol),
        &[("source", &format!("@{source_symbol}"))],
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
                let attrs = field_lowering_attrs(op)?;
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
            "poly.point_zero" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["field", "arity"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.point_zero",
                    Some(&symbol),
                    &attrs,
                    &[],
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
            "piop.sumcheck_eval_family" => {
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["source", "oracle_family", "count", "evals"])?;
                context.append_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_eval_family",
                    Some(&symbol),
                    &attrs,
                )?;
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
            "piop.structured_polynomial_eval" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "polynomial",
                        "x_point_segment",
                        "x_point_length",
                        "x_point_order",
                        "y_point_segment",
                        "y_point_length",
                        "y_point_order",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.structured_polynomial_eval",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_output_eval_family" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "power_stride",
                        "value_term_offsets",
                        "shared_term_offsets",
                        "item_term_offsets",
                        "evals",
                        "shared_terms",
                        "item_terms",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_output_eval_family",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_output_product_family" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "gamma",
                        "term_gamma_power_offsets",
                        "term_eval_counts",
                        "term_factor_counts",
                        "evals",
                        "factors",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_output_product_family",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_output_function_family" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(
                    op,
                    &[
                        "gamma",
                        "term_gamma_power_offsets",
                        "term_functions",
                        "term_factor_counts",
                        "evals",
                        "factors",
                    ],
                )?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_output_function_family",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.field_value"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "piop.sumcheck_output_claim" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["stage", "relation", "count", "polynomial_evals"])?;
                let _operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.sumcheck_output_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &[],
                )?;
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
            "pcs.opening_claim" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["oracle", "family", "domain", "point_arity"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.pcs_opening_claim",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.opening_claim_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "pcs.opening_batch" => {
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["proof_slot", "policy", "count", "ordered_claims"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    "compute.pcs_opening_batch",
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.opening_batch_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
            }
            "pcs.batch_open" | "pcs.batch_verify" => {
                let target_op = match &role {
                    Role::Prover => "compute.pcs_batch_open",
                    Role::Verifier => "compute.pcs_batch_verify",
                };
                let operands = lowered_operands(op, &value_map, 0)?;
                let symbol = string_attr(op, "sym_name")?;
                let attrs = copy_attrs(op, &["pcs", "proof_slot", "transcript_label"])?;
                let operation = context.append_typed_op_with_owned_attrs(
                    &compute,
                    target_op,
                    Some(&symbol),
                    &attrs,
                    &operands,
                    &["!compute.transcript_state", "!compute.opening_proof_type"],
                )?;
                insert_result_mapping(&mut value_map, op, operation, 0, 0)?;
                insert_result_mapping(&mut value_map, op, operation, 1, 1)?;
            }
            _ => {}
        }
    }

    verify_module(&compute)?;
    verify_compute_schema(&compute)?;
    Ok(compute)
}

fn compute_params_attrs<P: crate::ir::Phase>(
    module: &BoltModule<'_, P>,
) -> Result<Vec<(String, String)>, MlirError> {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if operation_name(op) == "protocol.params" {
            return copy_attrs(op, &["field", "pcs", "transcript"]);
        }
    }
    Err(schema_error("module missing protocol.params"))
}

fn transcript_squeeze_value_type(
    kind: &str,
    point_type: &'static str,
    scalar_type: &'static str,
) -> Result<&'static str, MlirError> {
    match kind {
        "challenge_vector" => Ok(point_type),
        "challenge_scalar" | "scalar" => Ok(scalar_type),
        kind => Err(schema_error(format!(
            "unsupported transcript squeeze kind `{kind}`"
        ))),
    }
}

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
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
    result_key(operation, result.result_number()).map_err(|_| {
        schema_error(format!(
            "{} result {index} owner missing sym_name",
            operation_name(operation)
        ))
    })
}

fn result_key(operation: OperationRef<'_, '_>, result_number: usize) -> Result<String, MlirError> {
    let symbol = string_attr(operation, "sym_name")?;
    Ok(format!("{symbol}#{result_number}"))
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
    let _ = value_map.insert(key, value);
    Ok(())
}
