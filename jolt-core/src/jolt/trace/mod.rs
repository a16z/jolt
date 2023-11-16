use ark_ff::PrimeField;

use super::{instruction::JoltInstruction, vm::pc::ELFRow};


// TODO(sragss): Move to memory checking.
pub enum MemoryOp {
    Read(u64, u64),       // (address, value)
    Write(u64, u64, u64), // (address, old_value, new_value)
}

impl MemoryOp {
    fn new_read(address: u64, value: u64) -> Self {
        Self::Read(address, value)
    }

    fn new_write(address: u64, old_value: u64, new_value: u64) -> Self {
        Self::Write(address, old_value, new_value)
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