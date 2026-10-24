use crate::tables::window_mask_d::WindowMaskDTable;
use crate::traits::impl_lookup_table;
use crate::traits::{LookupQuery, LookupTable};
use jolt_riscv::instructions::WindowMaskD;
use jolt_riscv::{JoltCycle, JoltInstructionRow};

impl_lookup_table!(WindowMaskD, Some(WindowMaskD));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for WindowMaskD<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            Into::<JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm
                & mask as i128,
        )
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (x as i128 + y) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        WindowMaskDTable::<XLEN>.materialize_entry(LookupQuery::<XLEN>::to_lookup_index(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, lookup_output_matches_trace_test};
    use tracer::instruction::virtual_window_mask_d::VirtualWindowMaskD;

    #[test]
    fn instruction_inputs_match_constraint_windowmaskd() {
        instruction_inputs_match_constraint_test!(WindowMaskD, VirtualWindowMaskD);
    }

    #[test]
    fn lookup_output_matches_trace_windowmaskd() {
        lookup_output_matches_trace_test!(WindowMaskD, VirtualWindowMaskD);
    }
}
