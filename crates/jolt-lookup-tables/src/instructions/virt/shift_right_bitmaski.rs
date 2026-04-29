use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualShiftRightBitmaski;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(VirtualShiftRightBitmaski, Some(ShiftRightBitmask));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualShiftRightBitmaski<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (0, self.0.instruction().imm() & mask as i128)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        let shift = (y & ((XLEN as u128) - 1)) as usize;
        (((1u128 << (XLEN - shift)) - 1) << shift) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualshiftrightbitmaski() {
        materialize_entry_test::<
            VirtualShiftRightBitmaski<RISCVCycle<tracer::instruction::virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI>>,
            RISCVCycle<tracer::instruction::virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI>,
        >();
    }
}
