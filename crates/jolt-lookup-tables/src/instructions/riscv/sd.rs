use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Sd;
use tracer::instruction::{sd::SD, RISCVCycle};

impl_lookup_table!(Sd, None);

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<SD> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
