#[derive(Debug, PartialEq)]
pub struct RVTraceRow {
    pub instruction: Instruction,
    pub register_state: RegisterState,
    pub memory_state: Option<MemoryState>,
}

#[derive(Debug, PartialEq)]
pub struct Instruction {
    pub address: u64,
    pub opcode: &'static str,
    pub rs1: Option<u64>,
    pub rs2: Option<u64>,
    pub rd: Option<u64>,
    pub imm: Option<i32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegisterState {
    pub rs1_val: Option<u64>,
    pub rs2_val: Option<u64>,
    pub rd_pre_val: Option<u64>,
    pub rd_post_val: Option<u64>,
}

#[derive(Debug, PartialEq)]
pub enum MemoryState {
    Read {
        address: u64,
        value: u64,
    },
    Write {
        address: u64,
        pre_value: u64,
        post_value: u64,
    },
}

pub mod constants;