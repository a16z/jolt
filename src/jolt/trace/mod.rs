use super::{instruction::JoltInstruction, vm::pc::ELFRow};


// TODO(sragss): Move to memory checking.
pub enum MemoryOp {
    Read(u64, u64),       // (address, value)
    Write(u64, u64, u64), // (address, old_value, new_value)
}

trait JoltProvableTrace {
    type JoltInstructionEnum: JoltInstruction;
    fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum>;
    fn to_ram_ops(&self) -> Vec<MemoryOp>;
    fn to_pc_trace(&self) -> ELFRow;
}

pub mod rv;