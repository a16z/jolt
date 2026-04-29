use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::BgeU;
use jolt_trace::JoltCycle;

impl_lookup_table!(BgeU, Some(UnsignedGreaterThanEqual));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for BgeU<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (x >= y as u64).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_bgeu() {
        materialize_entry_test::<
            BgeU<RISCVCycle<tracer::instruction::bgeu::BGEU>>,
            RISCVCycle<tracer::instruction::bgeu::BGEU>,
        >();
    }
}
