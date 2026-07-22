use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::SltI;
use jolt_riscv::JoltCycle;

impl_lookup_table!(SltI, Some(SignedLessThan));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for SltI<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            Into::<jolt_riscv::JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm
                & mask as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let sx = ((x as i64) << (64 - XLEN)) >> (64 - XLEN);
        let sy = ((y as i64) << (64 - XLEN)) >> (64 - XLEN);
        (sx < sy) as u64
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
    fn materialize_entry_slti() {
        materialize_entry_test!(SltI, jolt_tracer::instruction::slti::SLTI);
    }

    #[test]
    fn instruction_inputs_match_constraint_slti() {
        instruction_inputs_match_constraint_test!(SltI, jolt_tracer::instruction::slti::SLTI);
    }

    #[test]
    fn lookup_output_matches_trace_slti() {
        lookup_output_matches_trace_test!(SltI, jolt_tracer::instruction::slti::SLTI);
    }
}
