use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Sub;
use jolt_trace::JoltCycle;

impl_lookup_table!(Sub, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Sub<C> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let x = x as u128;
        let y = (1u128 << XLEN) - y as u128;
        (0, x + y)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x.wrapping_sub(y as u64) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_sub() {
        materialize_entry_test::<
            Sub<RISCVCycle<tracer::instruction::sub::SUB>>,
            RISCVCycle<tracer::instruction::sub::SUB>,
        >();
    }
}
