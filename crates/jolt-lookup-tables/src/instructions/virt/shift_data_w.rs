use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::ShiftDataW;
use jolt_riscv::JoltCycle;

impl_lookup_table!(ShiftDataW, Some(ShiftDataW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for ShiftDataW<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let eighth = XLEN / 8;
        let lane_mask = ((1u128 << (4 * eighth)) - 1) as u64;
        let lane = rs1 & lane_mask;
        let offset = rs2 as u64 & 4;
        let xlen_mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (lane << (eighth as u32 * offset as u32)) & xlen_mask
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
    fn materialize_entry_shiftdataw() {
        materialize_entry_test!(
            ShiftDataW,
            tracer::instruction::virtual_shift_data_w::VirtualShiftDataW
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_shiftdataw() {
        instruction_inputs_match_constraint_test!(
            ShiftDataW,
            tracer::instruction::virtual_shift_data_w::VirtualShiftDataW
        );
    }

    #[test]
    fn lookup_output_matches_trace_shiftdataw() {
        lookup_output_matches_trace_test!(
            ShiftDataW,
            tracer::instruction::virtual_shift_data_w::VirtualShiftDataW
        );
    }
}
