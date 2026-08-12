use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::{instructions::AddW, JoltCycle};

impl_lookup_table!(AddW, Some(SignExtendWord));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for AddW<C> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        x.wrapping_add(y as u64) as u32 as i32 as i64 as u64
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
    fn materialize_entry_addw() {
        materialize_entry_test!(AddW, tracer::instruction::addw::ADDW);
    }

    #[test]
    fn instruction_inputs_match_constraint_addw() {
        instruction_inputs_match_constraint_test!(AddW, tracer::instruction::addw::ADDW);
    }

    #[test]
    fn lookup_output_matches_trace_addw() {
        lookup_output_matches_trace_test!(AddW, tracer::instruction::addw::ADDW);
    }
}
