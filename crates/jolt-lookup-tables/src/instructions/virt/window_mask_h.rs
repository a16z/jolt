use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::WindowMaskH;
use jolt_riscv::{JoltCycle, JoltInstructionRow};

impl_lookup_table!(WindowMaskH, Some(WindowMaskH));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for WindowMaskH<C> {
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
        let eighth = XLEN / 8;
        let mask = ((1u128 << (2 * eighth)) - 1) as u64;
        mask << (eighth as u32 * (index & 6) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_window_mask_h::VirtualWindowMaskH;

    #[test]
    fn materialize_entry_windowmaskh() {
        materialize_entry_test!(WindowMaskH, VirtualWindowMaskH);
    }

    #[test]
    fn instruction_inputs_match_constraint_windowmaskh() {
        instruction_inputs_match_constraint_test!(WindowMaskH, VirtualWindowMaskH);
    }

    #[test]
    fn lookup_output_matches_trace_windowmaskh() {
        lookup_output_matches_trace_test!(WindowMaskH, VirtualWindowMaskH);
    }
}
