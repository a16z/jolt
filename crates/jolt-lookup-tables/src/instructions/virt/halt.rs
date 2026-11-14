use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::{Panic, Success};
use jolt_riscv::JoltCycle;

impl_lookup_table!(Success, None);
impl_lookup_table!(Panic, None);

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Success<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Panic<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
