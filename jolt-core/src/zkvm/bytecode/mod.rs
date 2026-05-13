use tracer::instruction::Cycle;

pub mod read_raf_checking;

pub use jolt_program::preprocess::{BytecodePCMapper, BytecodePreprocessing, PreprocessingError};

pub fn get_pc_for_cycle(bytecode: &BytecodePreprocessing, cycle: &Cycle) -> usize {
    if matches!(cycle, Cycle::NoOp) {
        return 0;
    }
    let instruction = cycle.instruction().jolt_instruction_row();
    match bytecode.get_pc(&instruction) {
        Some(pc) => pc,
        None => panic!(
            "bytecode preprocessing is missing PC mapping for instruction at address {:#x} with virtual_sequence_remaining {:?}",
            instruction.address, instruction.virtual_sequence_remaining
        ),
    }
}

pub fn entry_bytecode_index(bytecode: &BytecodePreprocessing) -> usize {
    match bytecode.entry_bytecode_index() {
        Some(pc) => pc,
        None => panic!(
            "bytecode preprocessing is missing entry mapping for address {:#x}",
            bytecode.entry_address
        ),
    }
}
