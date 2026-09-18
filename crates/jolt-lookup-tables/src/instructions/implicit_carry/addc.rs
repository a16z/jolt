use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::AddC;
use jolt_riscv::JoltCycle;

impl_lookup_table!(AddC, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for AddC<C> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        // rs1 + rs2 + carry < 3 * 2^64, well inside the 128-bit lookup domain.
        (0, x as u128 + y as u64 as u128 + self.0.carry() as u128)
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
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x.wrapping_add(y as u64).wrapping_add(self.0.carry()) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::addc::ADDC;

    #[test]
    fn materialize_entry_addc() {
        materialize_entry_test!(AddC, ADDC);
    }

    #[test]
    fn instruction_inputs_match_constraint_addc() {
        instruction_inputs_match_constraint_test!(AddC, ADDC);
    }

    #[test]
    fn lookup_output_matches_trace_addc() {
        lookup_output_matches_trace_test!(AddC, ADDC);
    }
}
