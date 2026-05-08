use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::OperationRef;

use crate::ir::{string_attribute_value, BoltModule, Concrete, Party, Protocol};
use crate::schema::{
    find_symbol, int_attr, missing_module_op, missing_symbol, require_symbol_attr_eq,
    symbol_array_attr, symbol_attr, verify_concrete_schema, verify_party_schema,
    verify_protocol_schema, SchemaError,
};

use super::oracles::{MAIN_WITNESS_FAMILY_SYMBOL, PCS_SYMBOL};
use super::params::ParsedJoltProtocolParams;

pub fn verify_jolt_protocol_schema(module: &BoltModule<'_, Protocol>) -> Result<(), SchemaError> {
    verify_protocol_schema(module)?;
    validate_jolt_shape(module)
}

pub fn verify_jolt_concrete_schema(module: &BoltModule<'_, Concrete>) -> Result<(), SchemaError> {
    verify_concrete_schema(module)?;
    validate_jolt_shape(module)
}

pub fn verify_jolt_party_schema(module: &BoltModule<'_, Party>) -> Result<(), SchemaError> {
    verify_party_schema(module)?;
    validate_jolt_shape(module)
}

fn validate_jolt_shape<P>(module: &BoltModule<'_, P>) -> Result<(), SchemaError>
where
    P: crate::ir::Phase,
{
    let params_op =
        find_symbol(module, "jolt.params").ok_or_else(|| missing_module_op("protocol.params"))?;
    let params = ParsedJoltProtocolParams::from_op(params_op)?;
    params.validate()?;

    require_symbol(module, &params.field)?;
    require_symbol(module, &params.pcs)?;
    require_symbol(module, &params.transcript)?;

    let witness_family = find_symbol(module, MAIN_WITNESS_FAMILY_SYMBOL)
        .ok_or_else(|| missing_symbol(MAIN_WITNESS_FAMILY_SYMBOL))?;
    let witness_count = int_attr(witness_family, "count")?;
    if witness_count != params.num_committed {
        return Err(SchemaError::new(format!(
            "main witness count {witness_count} does not match num_committed {}",
            params.num_committed
        )));
    }
    let ordered_oracles = symbol_array_attr(witness_family, "ordered_oracles")?;
    let expected_oracles = params.main_witness_oracles();
    if ordered_oracles != expected_oracles {
        return Err(SchemaError::new(format!(
            "main witness ordered_oracles mismatch: expected [{}], got [{}]",
            expected_oracles.join(", "),
            ordered_oracles.join(", ")
        )));
    }
    if ordered_oracles.len() != witness_count {
        return Err(SchemaError::new(format!(
            "main witness ordered_oracles length {} does not match count {witness_count}",
            ordered_oracles.len()
        )));
    }
    for oracle in &ordered_oracles {
        let oracle_op = find_symbol(module, oracle).ok_or_else(|| missing_symbol(oracle))?;
        require_symbol_attr_eq(oracle_op, "field", &params.field)?;
    }

    let commit = find_symbol(module, "jolt.main_witness_commitments")
        .ok_or_else(|| missing_symbol("jolt.main_witness_commitments"))?;
    require_symbol_attr_eq(commit, "oracle_family", MAIN_WITNESS_FAMILY_SYMBOL)?;

    let pcs = find_symbol(module, "jolt.dory_main_witness_commit")
        .ok_or_else(|| missing_symbol("jolt.dory_main_witness_commit"))?;
    require_operand_owner_symbol_eq(pcs, 0, "jolt.main_witness_commitments")?;
    let scheme = symbol_attr(pcs, "scheme")?;
    if scheme != PCS_SYMBOL || scheme != params.pcs {
        return Err(SchemaError::new(format!(
            "PCS scheme `{scheme}` does not match params pcs `{}`",
            params.pcs
        )));
    }

    Ok(())
}

fn require_operand_owner_symbol_eq(
    operation: OperationRef<'_, '_>,
    index: usize,
    expected: &str,
) -> Result<(), SchemaError> {
    let operand = operation.operand(index).map_err(|_| {
        SchemaError::new(format!(
            "{} missing required operand {index}",
            crate::schema::operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        SchemaError::new(format!(
            "{} operand {index} must be an op result",
            crate::schema::operation_name(operation)
        ))
    })?;
    let actual = owner
        .owner()
        .attribute("sym_name")
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| {
            SchemaError::new(format!(
                "{} operand {index} owner missing sym_name",
                crate::schema::operation_name(operation)
            ))
        })?;
    if actual == expected {
        Ok(())
    } else {
        Err(SchemaError::new(format!(
            "{} operand {index} expected @{expected}, got @{actual}",
            crate::schema::operation_name(operation)
        )))
    }
}

fn require_symbol<P>(module: &BoltModule<'_, P>, symbol: &str) -> Result<(), SchemaError>
where
    P: crate::ir::Phase,
{
    find_symbol(module, symbol)
        .map(|_| ())
        .ok_or_else(|| missing_symbol(symbol))
}
