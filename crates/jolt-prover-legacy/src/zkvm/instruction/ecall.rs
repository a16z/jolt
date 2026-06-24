use tracer::instruction::{ecall::ECALL, RISCVCycle};

use super::LookupQuery;

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ECALL> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
