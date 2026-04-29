use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::MulI;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(MulI, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for MulI<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            self.0.instruction().imm() & mask as i128,
        )
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 * y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let shift = 64 - XLEN;
        let signed_x = ((x as i64) << shift) >> shift;
        let signed_y = ((y as i64) << shift) >> shift;
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        signed_x.wrapping_mul(signed_y) as u64 & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualmuli() {
        materialize_entry_test::<
            MulI<RISCVCycle<tracer::instruction::virtual_muli::VirtualMULI>>,
            RISCVCycle<tracer::instruction::virtual_muli::VirtualMULI>,
        >();
    }
}
