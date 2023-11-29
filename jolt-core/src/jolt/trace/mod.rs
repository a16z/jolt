use ark_ff::PrimeField;

use super::{instruction::JoltInstruction, vm::pc::ELFRow};


#[derive(Debug, PartialEq)]
pub enum MemoryOp {
    Read(u64, u64),       // (address, value)
    Write(u64, u64, u64), // (address, old_value, new_value)
}

impl MemoryOp {
    fn no_op() -> Self {
        Self::Read(0, 0)
    }
}

trait JoltProvableTrace {
    type JoltInstructionEnum: JoltInstruction;
    fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum>;
    fn to_ram_ops(&self) -> Vec<MemoryOp>;
    fn to_pc_trace(&self) -> ELFRow;
    fn to_circuit_flags<F: PrimeField>(&self) -> Vec<F>;
}

pub mod rv;