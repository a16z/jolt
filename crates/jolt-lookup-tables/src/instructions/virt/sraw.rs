use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::VirtualSraw;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualSraw, Some(VirtualSRAW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSraw<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        sraw::<XLEN>(x, y as u64)
    }
}

pub(super) fn sraw<const XLEN: usize>(x: u64, y: u64) -> u64 {
    let half = XLEN / 2;
    let shift = y.trailing_zeros() as usize;
    let word = x as u128 & ((1u128 << half) - 1);
    let sign_bit = (word >> (half - 1)) & 1;
    let shifted = word >> shift;
    (shifted + sign_bit * ((1u128 << XLEN) - (1u128 << (half - shift)))) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    #[test]
    fn materialize_entry_virtualsraw() {
        materialize_entry_test!(VirtualSraw, tracer::instruction::virtual_sraw::VirtualSRAW);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsraw() {
        instruction_inputs_match_constraint_test!(
            VirtualSraw,
            tracer::instruction::virtual_sraw::VirtualSRAW
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualsraw() {
        lookup_output_matches_trace_test!(
            VirtualSraw,
            tracer::instruction::virtual_sraw::VirtualSRAW
        );
    }
}
