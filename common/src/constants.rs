pub const XLEN: usize = 32;
pub const REGISTER_COUNT: u64 = 32;
pub const REGISTER_START_ADDRESS: usize = 0;
pub const RAM_START_ADDRESS: u64 = 0x80000000;
pub const BYTES_PER_INSTRUCTION: usize = 4;
pub const MEMORY_OPS_PER_INSTRUCTION: usize = 7;
pub const NUM_R1CS_POLYS: usize = 1;
pub const INPUT_START_ADDRESS: u64 = 0x20000000;
pub const INPUT_END_ADDRESS: u64 = 0x20000FFF;
pub const OUTPUT_START_ADDRESS: u64 = 0x20001000;
pub const OUTPUT_END_ADDRESS: u64 = 0x20001FFF;
pub const PANIC_ADDRESS: u64 = 0x20002000;
