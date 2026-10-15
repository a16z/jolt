use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::AlignAddr;
use jolt_riscv::{JoltCycle, JoltInstructionRow};

impl_lookup_table!(AlignAddr, Some(AlignAddr));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for AlignAddr<C> {
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
            Into::<JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm
                & mask as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x.wrapping_add(y as u64) & mask & !7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_align_addr::VirtualAlignAddr;

    #[test]
    fn materialize_entry_alignaddr() {
        materialize_entry_test!(AlignAddr, VirtualAlignAddr);
    }

    #[test]
    fn instruction_inputs_match_constraint_alignaddr() {
        instruction_inputs_match_constraint_test!(AlignAddr, VirtualAlignAddr);
    }

    #[test]
    fn lookup_output_matches_trace_alignaddr() {
        lookup_output_matches_trace_test!(AlignAddr, VirtualAlignAddr);
    }
}
