use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualHostIO;
use tracer::instruction::{virtual_host_io::VirtualHostIO as TracerVirtualHostIO, RISCVCycle};

impl_lookup_table!(VirtualHostIO, None);

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualHostIO> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
