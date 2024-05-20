pub const XLEN: usize = 32;
const RISCV_REGISTER_COUNT: u64 = 32;
const VIRTUAL_REGISTER_COUNT: u64 = 32; //  see Section 6.1 of Jolt paper
pub const REGISTER_COUNT: u64 = RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT;
pub const BYTES_PER_INSTRUCTION: usize = 4;
pub const REG_OPS_PER_INSTRUCTION: usize = 3;
pub const RAM_OPS_PER_INSTRUCTION: usize = 4;
pub const MEMORY_OPS_PER_INSTRUCTION: usize = REG_OPS_PER_INSTRUCTION + RAM_OPS_PER_INSTRUCTION;

pub const RAM_START_ADDRESS: u64 = 0x80000000;
pub const DEFAULT_MEMORY_SIZE: u64 = 10 * 1024 * 1024;
pub const DEFAULT_STACK_SIZE: u64 = 4096;
pub const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;
pub const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;

pub const fn memory_address_to_witness_index(address: u64, ram_witness_offset: u64) -> usize {
    (address + ram_witness_offset - RAM_START_ADDRESS) as usize
}
pub const fn witness_index_to_memory_address(index: usize, ram_witness_offset: u64) -> u64 {
    index as u64 + RAM_START_ADDRESS - ram_witness_offset
}
pub const fn virtual_register_index(index: u64) -> u64 {
    index + VIRTUAL_REGISTER_COUNT
}

// Layout of the witness (where || denotes concatenation):
//     registers || inputs || outputs || panic || padding || RAM
// Layout of VM memory:
//     peripheral devices || inputs || outputs || panic || padding || RAM
// Notably, we want to be able to map the VM memory address space to witness indices
// using a constant shift, namely (RAM_WITNESS_OFFSET + RAM_START_ADDRESS)
