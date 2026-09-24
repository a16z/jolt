use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::WindowMaskB;
use jolt_riscv::{JoltCycle, JoltInstructionRow};

impl_lookup_table!(WindowMaskB, Some(WindowMaskB));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for WindowMaskB<C> {
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
        let mask = ((1u128 << eighth) - 1) as u64;
        mask << (eighth as u32 * (index & 7) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_window_mask_b::VirtualWindowMaskB;

    #[test]
    fn materialize_entry_windowmaskb() {
        materialize_entry_test!(WindowMaskB, VirtualWindowMaskB);
    }

    #[test]
    fn instruction_inputs_match_constraint_windowmaskb() {
        instruction_inputs_match_constraint_test!(WindowMaskB, VirtualWindowMaskB);
    }

    #[test]
    fn lookup_output_matches_trace_windowmaskb() {
        lookup_output_matches_trace_test!(WindowMaskB, VirtualWindowMaskB);
    }
}
