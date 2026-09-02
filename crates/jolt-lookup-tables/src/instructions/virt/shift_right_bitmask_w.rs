use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::VirtualShiftRightBitmaskW;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualShiftRightBitmaskW, Some(ShiftRightBitmaskW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualShiftRightBitmaskW<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (self.0.rs1_val().unwrap_or(0) & mask, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let half = XLEN / 2;
        let shift = LookupQuery::<XLEN>::to_lookup_index(self) as usize % half;
        ((1u128 << half) - (1u128 << shift)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    #[test]
    fn materialize_entry_virtualshiftrightbitmaskw() {
        materialize_entry_test!(
            VirtualShiftRightBitmaskW,
            tracer::instruction::virtual_shift_right_bitmask_w::VirtualShiftRightBitmaskW
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualshiftrightbitmaskw() {
        instruction_inputs_match_constraint_test!(
            VirtualShiftRightBitmaskW,
            tracer::instruction::virtual_shift_right_bitmask_w::VirtualShiftRightBitmaskW
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualshiftrightbitmaskw() {
        lookup_output_matches_trace_test!(
            VirtualShiftRightBitmaskW,
            tracer::instruction::virtual_shift_right_bitmask_w::VirtualShiftRightBitmaskW
        );
    }
}
