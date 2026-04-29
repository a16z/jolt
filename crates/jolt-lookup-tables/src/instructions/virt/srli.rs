use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSrli;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(VirtualSrli, Some(VirtualSRL));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrli<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), self.0.instruction().imm())
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (imm as u64).trailing_zeros();
        (rs1 & mask) >> shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualsrli() {
        materialize_entry_test::<
            VirtualSrli<RISCVCycle<tracer::instruction::virtual_srli::VirtualSRLI>>,
            RISCVCycle<tracer::instruction::virtual_srli::VirtualSRLI>,
        >();
    }
}
