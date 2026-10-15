use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::WindowMaskW;
use jolt_riscv::{JoltCycle, JoltInstructionRow};

impl_lookup_table!(WindowMaskW, Some(WindowMaskW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for WindowMaskW<C> {
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
        let index = LookupQuery::<XLEN>::to_lookup_index(self);
        let half = XLEN / 2;
        let mask = ((1u128 << half) - 1) as u64;
        mask << (half as u32 * ((index >> 2) & 1) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_window_mask_w::VirtualWindowMaskW;

    #[test]
    fn materialize_entry_windowmaskw() {
        materialize_entry_test!(WindowMaskW, VirtualWindowMaskW);
    }

    #[test]
    fn instruction_inputs_match_constraint_windowmaskw() {
        instruction_inputs_match_constraint_test!(WindowMaskW, VirtualWindowMaskW);
    }

    #[test]
    fn lookup_output_matches_trace_windowmaskw() {
        lookup_output_matches_trace_test!(WindowMaskW, VirtualWindowMaskW);
    }
}
