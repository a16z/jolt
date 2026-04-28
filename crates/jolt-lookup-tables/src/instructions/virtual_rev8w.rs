use crate::tables::virtual_rev8w::rev8w;
use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualRev8W;
use tracer::instruction::{virtual_rev8w::VirtualRev8W as TracerVirtualRev8W, RISCVCycle};

impl_lookup_table!(VirtualRev8W, Some(VirtualRev8W));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualRev8W> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        (0, self.register_state.rs1.into())
    }

    fn to_lookup_index(&self) -> u128 {
        self.register_state.rs1.into()
    }

    fn to_lookup_output(&self) -> u64 {
        rev8w(self.register_state.rs1)
    }
}
