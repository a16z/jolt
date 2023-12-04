use ark_ff::PrimeField;

use super::{instruction::JoltInstruction, vm::{pc::ELFRow, read_write_memory::MemoryOp}};


trait JoltProvableTrace {
    type JoltInstructionEnum: JoltInstruction;
    fn to_jolt_instructions(&self) -> Vec<Self::JoltInstructionEnum>;
    fn to_ram_ops(&self) -> Vec<MemoryOp>;
    fn to_pc_trace(&self) -> ELFRow;
    fn to_circuit_flags<F: PrimeField>(&self) -> Vec<F>;
}

pub mod rv;