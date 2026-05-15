use melior::ir::OperationRef;

use crate::schema::{
    int_attr as mlir_int_attr, require_attrs, symbol_attr as mlir_symbol_attr, SchemaError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltProtocolParams {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
    pub xlen: usize,
    pub log_t: usize,
    pub trace_length: usize,
    pub log_k_bytecode: usize,
    pub bytecode_k: usize,
    pub log_k_ram: usize,
    pub ram_k: usize,
    pub log_k_chunk: usize,
    pub k_chunk: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    pub instruction_log_k: usize,
    pub register_log_k: usize,
    pub lookup_table_count: usize,
    pub instruction_d: usize,
    pub instruction_ra_virtual_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,
    pub num_committed: usize,
    pub num_r1cs_constraints: usize,
    pub num_r1cs_inputs: usize,
    pub num_vars_padded: usize,
}

impl JoltProtocolParams {
    pub fn new(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Self {
        let log_k_chunk = if log_t < 25 { 4 } else { 8 };
        let instruction_log_k = 128;
        let lookups_ra_virtual_log_k_chunk = if log_t < 25 {
            instruction_log_k / 8
        } else {
            instruction_log_k / 4
        };
        let instruction_d = instruction_log_k / log_k_chunk;
        let instruction_ra_virtual_d = instruction_log_k / lookups_ra_virtual_log_k_chunk;
        let bytecode_d = log_k_bytecode.div_ceil(log_k_chunk);
        let ram_d = log_k_ram.div_ceil(log_k_chunk);
        Self {
            field: "bn254_fr",
            pcs: "dory",
            transcript: "blake2b_transcript",
            xlen: 64,
            log_t,
            trace_length: 1usize << log_t,
            log_k_bytecode,
            bytecode_k: 1usize << log_k_bytecode,
            log_k_ram,
            ram_k: 1usize << log_k_ram,
            log_k_chunk,
            k_chunk: 1usize << log_k_chunk,
            lookups_ra_virtual_log_k_chunk,
            instruction_log_k,
            register_log_k: 7,
            lookup_table_count: 41,
            instruction_d,
            instruction_ra_virtual_d,
            bytecode_d,
            ram_d,
            num_committed: 2 + instruction_d + bytecode_d + ram_d,
            num_r1cs_constraints: 19,
            num_r1cs_inputs: 35,
            num_vars_padded: 64,
        }
    }

    pub fn fixture() -> Self {
        // Max shape supported by current jolt-kernels Stage 6 RA-virtual
        // sumcheck: DENSE_STAGE6_MAX_DEGREE=5 caps bytecode_d and ram_d at 4
        // under log_k_chunk=4, so log_k_bytecode <= 16 and log_k_ram <= 16.
        // log_t=18 gives a 2^18 universal trace ceiling.
        Self::new(18, 14, 14)
    }

    pub fn attrs(&self) -> Vec<(String, String)> {
        vec![
            symbol_attr("field", self.field),
            symbol_attr("pcs", self.pcs),
            symbol_attr("transcript", self.transcript),
            int_attr("xlen", self.xlen),
            int_attr("log_t", self.log_t),
            int_attr("trace_length", self.trace_length),
            int_attr("log_k_bytecode", self.log_k_bytecode),
            int_attr("bytecode_k", self.bytecode_k),
            int_attr("log_k_ram", self.log_k_ram),
            int_attr("ram_k", self.ram_k),
            int_attr("log_k_chunk", self.log_k_chunk),
            int_attr("k_chunk", self.k_chunk),
            int_attr(
                "lookups_ra_virtual_log_k_chunk",
                self.lookups_ra_virtual_log_k_chunk,
            ),
            int_attr("instruction_log_k", self.instruction_log_k),
            int_attr("register_log_k", self.register_log_k),
            int_attr("lookup_table_count", self.lookup_table_count),
            int_attr("instruction_d", self.instruction_d),
            int_attr("instruction_ra_virtual_d", self.instruction_ra_virtual_d),
            int_attr("bytecode_d", self.bytecode_d),
            int_attr("ram_d", self.ram_d),
            int_attr("num_committed", self.num_committed),
            int_attr("num_r1cs_constraints", self.num_r1cs_constraints),
            int_attr("num_r1cs_inputs", self.num_r1cs_inputs),
            int_attr("num_vars_padded", self.num_vars_padded),
        ]
    }
}

fn symbol_attr(name: &str, value: &str) -> (String, String) {
    (name.to_owned(), format!("@{value}"))
}

fn int_attr(name: &str, value: usize) -> (String, String) {
    (name.to_owned(), format!("{value} : i64"))
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedJoltProtocolParams {
    pub(crate) field: String,
    pub(crate) pcs: String,
    pub(crate) transcript: String,
    pub(crate) log_t: usize,
    pub(crate) trace_length: usize,
    pub(crate) log_k_bytecode: usize,
    pub(crate) bytecode_k: usize,
    pub(crate) log_k_ram: usize,
    pub(crate) ram_k: usize,
    pub(crate) log_k_chunk: usize,
    pub(crate) k_chunk: usize,
    pub(crate) lookups_ra_virtual_log_k_chunk: usize,
    pub(crate) instruction_log_k: usize,
    pub(crate) instruction_d: usize,
    pub(crate) instruction_ra_virtual_d: usize,
    pub(crate) bytecode_d: usize,
    pub(crate) ram_d: usize,
    pub(crate) num_committed: usize,
}

impl ParsedJoltProtocolParams {
    pub(crate) fn from_op(operation: OperationRef<'_, '_>) -> Result<Self, SchemaError> {
        require_jolt_params_attrs(operation)?;
        Ok(Self {
            field: mlir_symbol_attr(operation, "field")?,
            pcs: mlir_symbol_attr(operation, "pcs")?,
            transcript: mlir_symbol_attr(operation, "transcript")?,
            log_t: mlir_int_attr(operation, "log_t")?,
            trace_length: mlir_int_attr(operation, "trace_length")?,
            log_k_bytecode: mlir_int_attr(operation, "log_k_bytecode")?,
            bytecode_k: mlir_int_attr(operation, "bytecode_k")?,
            log_k_ram: mlir_int_attr(operation, "log_k_ram")?,
            ram_k: mlir_int_attr(operation, "ram_k")?,
            log_k_chunk: mlir_int_attr(operation, "log_k_chunk")?,
            k_chunk: mlir_int_attr(operation, "k_chunk")?,
            lookups_ra_virtual_log_k_chunk: mlir_int_attr(
                operation,
                "lookups_ra_virtual_log_k_chunk",
            )?,
            instruction_log_k: mlir_int_attr(operation, "instruction_log_k")?,
            instruction_d: mlir_int_attr(operation, "instruction_d")?,
            instruction_ra_virtual_d: mlir_int_attr(operation, "instruction_ra_virtual_d")?,
            bytecode_d: mlir_int_attr(operation, "bytecode_d")?,
            ram_d: mlir_int_attr(operation, "ram_d")?,
            num_committed: mlir_int_attr(operation, "num_committed")?,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), SchemaError> {
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
        if self.lookups_ra_virtual_log_k_chunk < self.log_k_chunk {
            return Err(SchemaError::new(format!(
                "lookups_ra_virtual_log_k_chunk must be >= log_k_chunk; got {} < {}",
                self.lookups_ra_virtual_log_k_chunk, self.log_k_chunk
            )));
        }
        if !self
            .lookups_ra_virtual_log_k_chunk
            .is_multiple_of(self.log_k_chunk)
        {
            return Err(SchemaError::new(format!(
                "lookups_ra_virtual_log_k_chunk must be a multiple of log_k_chunk; got {} and {}",
                self.lookups_ra_virtual_log_k_chunk, self.log_k_chunk
            )));
        }
        if !self
            .instruction_log_k
            .is_multiple_of(self.lookups_ra_virtual_log_k_chunk)
        {
            return Err(SchemaError::new(format!(
                "instruction_log_k must be divisible by lookups_ra_virtual_log_k_chunk; got {} and {}",
                self.instruction_log_k, self.lookups_ra_virtual_log_k_chunk
            )));
        }

        let instruction_d = self.instruction_log_k / self.log_k_chunk;
        let instruction_ra_virtual_d = self.instruction_log_k / self.lookups_ra_virtual_log_k_chunk;
        let bytecode_d = self.log_k_bytecode.div_ceil(self.log_k_chunk);
        let ram_d = self.log_k_ram.div_ceil(self.log_k_chunk);
        require_eq("instruction_d", self.instruction_d, instruction_d)?;
        require_eq(
            "instruction_ra_virtual_d",
            self.instruction_ra_virtual_d,
            instruction_ra_virtual_d,
        )?;
        require_eq("bytecode_d", self.bytecode_d, bytecode_d)?;
        require_eq("ram_d", self.ram_d, ram_d)?;
        require_eq(
            "num_committed",
            self.num_committed,
            2 + instruction_d + bytecode_d + ram_d,
        )?;
        Ok(())
    }

    pub(crate) fn main_witness_oracles(&self) -> Vec<String> {
        let mut oracles = vec!["RdInc".to_owned(), "RamInc".to_owned()];
        oracles.extend((0..self.instruction_d).map(|index| format!("InstructionRa_{index}")));
        oracles.extend((0..self.ram_d).map(|index| format!("RamRa_{index}")));
        oracles.extend((0..self.bytecode_d).map(|index| format!("BytecodeRa_{index}")));
        oracles
    }
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
            "lookups_ra_virtual_log_k_chunk",
            "instruction_log_k",
            "register_log_k",
            "lookup_table_count",
            "instruction_d",
            "instruction_ra_virtual_d",
            "bytecode_d",
            "ram_d",
            "num_committed",
            "num_r1cs_constraints",
            "num_r1cs_inputs",
            "num_vars_padded",
        ],
    )
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
