pub const XLEN: usize = 64;
const RISCV_REGISTER_COUNT: u64 = 32;
const VIRTUAL_REGISTER_COUNT: u64 = 96; //  see Section 6.1 of Jolt paper
pub const REGISTER_COUNT: u64 = RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT; // must be a power of 2
pub const BYTES_PER_INSTRUCTION: usize = 4;
pub const ALIGNMENT_FACTOR_BYTECODE: usize = 2;

pub const RAM_START_ADDRESS: u64 = 0x80000000;

// big enough to run Linux and xv6
pub const EMULATOR_MEMORY_CAPACITY: u64 = 1024 * 1024 * 128;

pub const DEFAULT_MEMORY_SIZE: u64 = 32 * 1024 * 1024;

pub const DEFAULT_STACK_SIZE: u64 = 4096;
pub const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_TRACE_LENGTH: u64 = 1 << 24;

pub const fn virtual_register_index(index: u64) -> u64 {
    index + RISCV_REGISTER_COUNT
}

// Layout of the witness (where || denotes concatenation):
//     inputs || outputs || panic || termination || padding || RAM
// Layout of VM memory:
//     peripheral devices || inputs || outputs || panic || termination || padding || RAM
// Notably, we want to be able to map the VM memory address space to witness indices
// using a constant shift, namely (RAM_WITNESS_OFFSET + RAM_START_ADDRESS)
