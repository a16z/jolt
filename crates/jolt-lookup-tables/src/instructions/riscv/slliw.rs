use super::sign_extend_half_word;
use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::{instructions::SllIW, JoltCycle};

impl_lookup_table!(SllIW, Some(SignExtendHalfWord));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for SllIW<C> {
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

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 * y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        sign_extend_half_word::<XLEN>(x.wrapping_mul(y as u64))
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
    fn materialize_entry_slliw() {
        materialize_entry_test!(SllIW, tracer::instruction::slliw::SLLIW);
    }

    #[test]
    fn instruction_inputs_match_constraint_slliw() {
        instruction_inputs_match_constraint_test!(SllIW, tracer::instruction::slliw::SLLIW);
    }

    #[test]
    fn lookup_output_matches_trace_slliw() {
        lookup_output_matches_trace_test!(SllIW, tracer::instruction::slliw::SLLIW);
    }
}
