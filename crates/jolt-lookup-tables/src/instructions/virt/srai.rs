use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::VirtualSrai;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualSrai, Some(VirtualSRA));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrai<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            Into::<jolt_riscv::JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (imm as u64).trailing_zeros();
        let signed = ((rs1 as i64) << (64 - XLEN)) >> (64 - XLEN);
        ((signed >> shift) as u64) & mask
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
    fn materialize_entry_virtualsrai() {
        materialize_entry_test!(VirtualSrai, tracer::instruction::virtual_srai::VirtualSRAI);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsrai() {
        instruction_inputs_match_constraint_test!(
            VirtualSrai,
            tracer::instruction::virtual_srai::VirtualSRAI
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualsrai() {
        lookup_output_matches_trace_test!(
            VirtualSrai,
            tracer::instruction::virtual_srai::VirtualSRAI
        );
    }
}
