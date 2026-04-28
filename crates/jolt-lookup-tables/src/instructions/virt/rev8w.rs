use crate::tables::virtual_rev8w::rev8w;
use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualRev8W;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualRev8W, Some(VirtualRev8W));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualRev8W<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        (0, self.0.rs1_val().unwrap_or(0).into())
    }

    fn to_lookup_index(&self) -> u128 {
        self.0.rs1_val().unwrap_or(0).into()
    }

    fn to_lookup_output(&self) -> u64 {
        rev8w(self.0.rs1_val().unwrap_or(0))
    }
}
