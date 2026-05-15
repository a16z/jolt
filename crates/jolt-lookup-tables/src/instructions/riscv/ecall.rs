use crate::traits::LookupQuery;
use jolt_riscv::instructions::Ecall;
use jolt_riscv::JoltCycle;

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Ecall<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
