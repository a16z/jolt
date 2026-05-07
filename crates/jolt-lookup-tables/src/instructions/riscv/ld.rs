use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Ld;
use jolt_trace::JoltCycle;

impl_lookup_table!(Ld, None);

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Ld<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
