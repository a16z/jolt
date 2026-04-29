use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualHostIO;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualHostIO, None);

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualHostIO<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
