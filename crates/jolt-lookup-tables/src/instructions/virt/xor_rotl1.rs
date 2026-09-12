use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::VirtualXorRotL1;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualXorRotL1, Some(VirtualXORROTL1));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotL1<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (x ^ (y as u64) << 1 ^ (y as u64) >> (XLEN - 1)) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_xor_rotl1::VirtualXORROTL1;

    #[test]
    fn materialize_entry_virtualxorrotl1() {
        materialize_entry_test!(VirtualXorRotL1, VirtualXORROTL1);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotl1() {
        instruction_inputs_match_constraint_test!(VirtualXorRotL1, VirtualXORROTL1);
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotl1() {
        lookup_output_matches_trace_test!(VirtualXorRotL1, VirtualXORROTL1);
    }
}
