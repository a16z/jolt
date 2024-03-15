use super::{
    instruction::JoltInstruction,
    vm::{bytecode::BytecodeRow, read_write_memory::MemoryOp},
};

pub trait JoltProvableTrace {
    type JoltInstructionEnum: JoltInstruction;
    fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum>;
    fn to_ram_ops(&self) -> Vec<MemoryOp>;
}

pub mod rv;
