use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertValidDiv0;
use jolt_trace::JoltCycle;

impl_lookup_table!(AssertValidDiv0, Some(ValidDiv0));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for AssertValidDiv0<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (divisor, quotient) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let max_val = (1u128 << XLEN).wrapping_sub(1) as u64;
        if divisor == 0 {
            (quotient as u64 == max_val).into()
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualassertvaliddiv0() {
        materialize_entry_test::<
            AssertValidDiv0<RISCVCycle<tracer::instruction::virtual_assert_valid_div0::VirtualAssertValidDiv0>>,
            RISCVCycle<tracer::instruction::virtual_assert_valid_div0::VirtualAssertValidDiv0>,
        >();
    }
}
