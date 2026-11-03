use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::Ld;
use jolt_riscv::JoltCycle;

impl_lookup_table!(Ld, None);

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Ld<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, self.0.rd_vals().map_or(0, |(_, post)| post) as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
