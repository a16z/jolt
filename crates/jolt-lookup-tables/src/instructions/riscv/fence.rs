use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Fence;
use tracer::instruction::{fence::FENCE, RISCVCycle};

impl_lookup_table!(Fence, None);

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<FENCE> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
