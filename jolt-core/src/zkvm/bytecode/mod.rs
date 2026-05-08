use tracer::instruction::Cycle;

pub mod read_raf_checking;

pub use jolt_program::preprocess::{BytecodePCMapper, BytecodePreprocessing, PreprocessingError};

pub fn get_pc_for_cycle(bytecode: &BytecodePreprocessing, cycle: &Cycle) -> usize {
    if matches!(cycle, Cycle::NoOp) {
        return 0;
    }
    let instruction = cycle.instruction().normalize();
    bytecode.get_pc(&instruction).unwrap_or(0)
}

pub fn entry_bytecode_index(bytecode: &BytecodePreprocessing) -> usize {
    bytecode.entry_bytecode_index().unwrap_or(0)
}
