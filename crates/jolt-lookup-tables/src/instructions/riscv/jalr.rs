use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Jalr;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(Jalr, Some(RangeCheckAligned));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Jalr<C> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (x as i128 + y) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            self.0.instruction().imm() & mask as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x.wrapping_add(y as u64) & mask & !1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_jalr() {
        materialize_entry_test::<
            Jalr<RISCVCycle<tracer::instruction::jalr::JALR>>,
            RISCVCycle<tracer::instruction::jalr::JALR>,
        >();
    }
}
