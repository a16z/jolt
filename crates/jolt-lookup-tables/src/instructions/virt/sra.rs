use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSra;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualSra, Some(VirtualSRA));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSra<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (rs2 as u64).trailing_zeros();
        let signed = ((rs1 as i64) << (64 - XLEN)) >> (64 - XLEN);
        ((signed >> shift) as u64) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualsra() {
        materialize_entry_test::<
            VirtualSra<RISCVCycle<tracer::instruction::virtual_sra::VirtualSRA>>,
            RISCVCycle<tracer::instruction::virtual_sra::VirtualSRA>,
        >();
    }
}
