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
    pub instruction_log_k: usize,
    pub register_log_k: usize,
    pub lookup_table_count: usize,
    pub instruction_d: usize,
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
        let instruction_d = instruction_log_k / log_k_chunk;
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
            instruction_log_k,
            register_log_k: 7,
            lookup_table_count: 41,
            instruction_d,
            bytecode_d,
            ram_d,
            num_committed: 2 + instruction_d + bytecode_d + ram_d,
            num_r1cs_constraints: 19,
            num_r1cs_inputs: 35,
            num_vars_padded: 64,
        }
    }

    pub fn fixture() -> Self {
        Self::new(16, 10, 16)
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
            int_attr("instruction_log_k", self.instruction_log_k),
            int_attr("register_log_k", self.register_log_k),
            int_attr("lookup_table_count", self.lookup_table_count),
            int_attr("instruction_d", self.instruction_d),
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
