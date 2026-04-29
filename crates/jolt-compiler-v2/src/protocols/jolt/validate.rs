use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::OperationRef;

use crate::ir::{string_attribute_value, BoltModule, Concrete, Party, Protocol};
use crate::schema::{
    find_symbol, int_attr, missing_module_op, missing_symbol, require_attrs,
    require_symbol_attr_eq, symbol_array_attr, symbol_attr, verify_concrete_schema,
    verify_party_schema, verify_protocol_schema, SchemaError,
};

use super::oracles::{MAIN_WITNESS_FAMILY_SYMBOL, PCS_SYMBOL};

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
    require_jolt_params_attrs(params_op)?;
    let params = ParsedJoltParams::from_op(params_op)?;
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

fn require_jolt_params_attrs(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    require_attrs(
        operation,
        &[
            "sym_name",
            "field",
            "pcs",
            "transcript",
            "xlen",
            "log_t",
            "trace_length",
            "log_k_bytecode",
            "bytecode_k",
            "log_k_ram",
            "ram_k",
            "log_k_chunk",
            "k_chunk",
            "instruction_log_k",
            "register_log_k",
            "lookup_table_count",
            "instruction_d",
            "bytecode_d",
            "ram_d",
            "num_committed",
            "num_r1cs_constraints",
            "num_r1cs_inputs",
            "num_vars_padded",
        ],
    )
}

#[derive(Clone, Debug)]
struct ParsedJoltParams {
    field: String,
    pcs: String,
    transcript: String,
    log_t: usize,
    trace_length: usize,
    log_k_bytecode: usize,
    bytecode_k: usize,
    log_k_ram: usize,
    ram_k: usize,
    log_k_chunk: usize,
    k_chunk: usize,
    instruction_log_k: usize,
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    num_committed: usize,
}

impl ParsedJoltParams {
    fn from_op(operation: OperationRef<'_, '_>) -> Result<Self, SchemaError> {
        Ok(Self {
            field: symbol_attr(operation, "field")?,
            pcs: symbol_attr(operation, "pcs")?,
            transcript: symbol_attr(operation, "transcript")?,
            log_t: int_attr(operation, "log_t")?,
            trace_length: int_attr(operation, "trace_length")?,
            log_k_bytecode: int_attr(operation, "log_k_bytecode")?,
            bytecode_k: int_attr(operation, "bytecode_k")?,
            log_k_ram: int_attr(operation, "log_k_ram")?,
            ram_k: int_attr(operation, "ram_k")?,
            log_k_chunk: int_attr(operation, "log_k_chunk")?,
            k_chunk: int_attr(operation, "k_chunk")?,
            instruction_log_k: int_attr(operation, "instruction_log_k")?,
            instruction_d: int_attr(operation, "instruction_d")?,
            bytecode_d: int_attr(operation, "bytecode_d")?,
            ram_d: int_attr(operation, "ram_d")?,
            num_committed: int_attr(operation, "num_committed")?,
        })
    }

    fn validate(&self) -> Result<(), SchemaError> {
        require_power_relation("trace_length", self.trace_length, "log_t", self.log_t)?;
        require_power_relation(
            "bytecode_k",
            self.bytecode_k,
            "log_k_bytecode",
            self.log_k_bytecode,
        )?;
        require_power_relation("ram_k", self.ram_k, "log_k_ram", self.log_k_ram)?;
        require_power_relation("k_chunk", self.k_chunk, "log_k_chunk", self.log_k_chunk)?;

        if self.log_k_chunk != 4 && self.log_k_chunk != 8 {
            return Err(SchemaError::new(format!(
                "log_k_chunk must be 4 or 8, got {}",
                self.log_k_chunk
            )));
        }
        if self.instruction_log_k != 128 {
            return Err(SchemaError::new(format!(
                "instruction_log_k must be 128, got {}",
                self.instruction_log_k
            )));
        }

        let instruction_d = self.instruction_log_k / self.log_k_chunk;
        let bytecode_d = self.log_k_bytecode.div_ceil(self.log_k_chunk);
        let ram_d = self.log_k_ram.div_ceil(self.log_k_chunk);
        require_eq("instruction_d", self.instruction_d, instruction_d)?;
        require_eq("bytecode_d", self.bytecode_d, bytecode_d)?;
        require_eq("ram_d", self.ram_d, ram_d)?;
        require_eq(
            "num_committed",
            self.num_committed,
            2 + instruction_d + bytecode_d + ram_d,
        )?;
        Ok(())
    }

    fn main_witness_oracles(&self) -> Vec<String> {
        let mut oracles = vec!["RdInc".to_owned(), "RamInc".to_owned()];
        oracles.extend((0..self.instruction_d).map(|index| format!("InstructionRa_{index}")));
        oracles.extend((0..self.ram_d).map(|index| format!("RamRa_{index}")));
        oracles.extend((0..self.bytecode_d).map(|index| format!("BytecodeRa_{index}")));
        oracles
    }
}

fn require_power_relation(
    value_name: &str,
    value: usize,
    log_name: &str,
    log_value: usize,
) -> Result<(), SchemaError> {
    let expected = 1usize << log_value;
    if value == expected {
        Ok(())
    } else {
        Err(SchemaError::new(format!(
            "{value_name} must equal 2^{log_name}; got {value}, expected {expected}"
        )))
    }
}

fn require_eq(name: &str, actual: usize, expected: usize) -> Result<(), SchemaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SchemaError::new(format!(
            "{name} must be {expected}, got {actual}"
        )))
    }
}
