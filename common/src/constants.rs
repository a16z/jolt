pub const XLEN: usize = 64;
pub const RISCV_REGISTER_COUNT: u8 = 32;
pub const VIRTUAL_REGISTER_COUNT: u8 = 96; //  see Section 6.1 of Jolt paper
pub const VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT: u8 = 7; // Reserved virtual registers for virtual instructions
pub const REGISTER_COUNT: u8 = RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT; // must be a power of 2
pub const BYTES_PER_INSTRUCTION: usize = 4;
pub const ALIGNMENT_FACTOR_BYTECODE: usize = 2;

/// Threshold for trace length (log scale) at which we switch between different
/// one-hot chunking parameters. Below this threshold (i.e., for smaller traces),
/// we use smaller chunk sizes for better performance (reduced commitment & PCS opening costs).
/// This value was empirically determined.
pub const ONEHOT_CHUNK_THRESHOLD_LOG_T: usize = 25;

/// Threshold for trace length (log scale) at which we switch the number of
/// instruction sumcheck phases from 16 to 8. Below this threshold, we use
/// more phases (16) for smaller sumcheck instances in each phase (8 instead of 16 variables).
/// This value was empirically determined.
pub const INSTRUCTION_PHASES_THRESHOLD_LOG_T: usize = 24;

pub const RAM_START_ADDRESS: u64 = 0x80000000;

// big enough to run Linux and xv6
pub const EMULATOR_MEMORY_CAPACITY: u64 = 1024 * 1024 * 128;

pub const DEFAULT_MEMORY_SIZE: u64 = EMULATOR_MEMORY_CAPACITY;

pub const DEFAULT_STACK_SIZE: u64 = 4096;
// 64 byte stack canary. 4 word protection for 32-bit and 2 word for 64-bit
pub const STACK_CANARY_SIZE: u64 = 128;
pub const DEFAULT_MAX_TRUSTED_ADVICE_SIZE: u64 = 4096;
pub const DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE: u64 = 4096;
pub const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_TRACE_LENGTH: u64 = 1 << 24;

// Layout of the witness (where || denotes concatenation):
//     advice || inputs || outputs || panic || termination || padding || RAM
// Layout of VM memory:
//     peripheral devices || advice || inputs || outputs || panic || termination || padding || RAM
// Notably, we want to be able to map the VM memory address space to witness indices
// using a constant shift, namely (RAM_WITNESS_OFFSET + RAM_START_ADDRESS)
