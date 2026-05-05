use melior::ir::operation::OperationRef;

use super::attrs::{int_attr, string_attr, symbol_attr};
use super::names::operation_name;
use super::symbols::operand_owner_result;
use crate::schema::SchemaError;

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpeningClaimMetadata {
    owner: String,
    oracle: String,
    domain: String,
    point_arity: usize,
    claim_kind: String,
}

pub(in crate::schema) fn require_opening_claim_equality(
    operation: OperationRef<'_, '_>,
) -> Result<(), SchemaError> {
    let mode = string_attr(operation, "mode")?;
    if mode != "point_and_eval" {
        return Err(SchemaError::new(format!(
            "{} attr `mode` expected \"point_and_eval\", got \"{mode}\"",
            operation_name(operation)
        )));
    }

    let left = opening_claim_metadata(operation, 0)?;
    let right = opening_claim_metadata(operation, 1)?;
    if left.oracle != right.oracle
        || left.domain != right.domain
        || left.point_arity != right.point_arity
        || left.claim_kind != right.claim_kind
    {
        return Err(SchemaError::new(format!(
            "{} compares incompatible claims @{} and @{}",
            operation_name(operation),
            left.owner,
            right.owner
        )));
    }
    Ok(())
}

fn opening_claim_metadata(
    equality_op: OperationRef<'_, '_>,
    operand_index: usize,
) -> Result<OpeningClaimMetadata, SchemaError> {
    let owner = operand_owner_result(equality_op, operand_index)?;
    let operation = owner.owner();
    let result_number = owner.result_number();
    let expected_result = match operation_name(operation).as_str() {
        "piop.opening_input" | "compute.opening_input" | "cpu.opening_input" => 2,
        "piop.opening_claim" | "compute.opening_claim" | "cpu.opening_claim" => 0,
        name => {
            return Err(SchemaError::new(format!(
                "{} operand {operand_index} must be an opening claim, got result from `{name}`",
                operation_name(equality_op)
            )));
        }
    };
    if result_number != expected_result {
        return Err(SchemaError::new(format!(
            "{} operand {operand_index} must use opening claim result {expected_result}, got result {result_number}",
            operation_name(equality_op)
        )));
    }

    Ok(OpeningClaimMetadata {
        owner: string_attr(operation, "sym_name")?,
        oracle: symbol_attr(operation, "oracle")?,
        domain: symbol_attr(operation, "domain")?,
        point_arity: int_attr(operation, "point_arity")?,
        claim_kind: string_attr(operation, "claim_kind")?,
    })
}
