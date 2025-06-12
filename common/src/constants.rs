pub const XLEN: usize = 32;
const RISCV_REGISTER_COUNT: u64 = 32;
const VIRTUAL_REGISTER_COUNT: u64 = 32; //  see Section 6.1 of Jolt paper
pub const REGISTER_COUNT: u64 = RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT;
pub const BYTES_PER_INSTRUCTION: usize = 4;
/// 3 registers (rd, rs1, rs2) + 1 RAM
pub const MEMORY_OPS_PER_INSTRUCTION: usize = 4;

pub const RAM_START_ADDRESS: u64 = 0x80000000;
pub const DEFAULT_MEMORY_SIZE: u64 = 10 * 1024 * 1024;
pub const DEFAULT_STACK_SIZE: u64 = 4096;
pub const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_BYTECODE_SIZE: u64 = 1 << 20;
pub const DEFAULT_MAX_TRACE_LENGTH: u64 = 1 << 24;

pub const fn virtual_register_index(index: u64) -> u64 {
    index + VIRTUAL_REGISTER_COUNT
}

// Layout of the witness (where || denotes concatenation):
//     registers || virtual registers || inputs || outputs || panic || termination || padding || RAM
// Layout of VM memory:
//     peripheral devices || inputs || outputs || panic || termination || padding || RAM
// Notably, we want to be able to map the VM memory address space to witness indices
// using a constant shift, namely (RAM_WITNESS_OFFSET + RAM_START_ADDRESS)
