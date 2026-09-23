use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::VirtualSrlw;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualSrlw, Some(VirtualSRLW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrlw<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        srlw::<XLEN>(x, y as u64)
    }
}

pub(super) fn srlw<const XLEN: usize>(x: u64, y: u64) -> u64 {
    let half = XLEN / 2;
    let word_mask = (1u128 << half) - 1;
    let shift = y.trailing_zeros() as usize;
    let result = (x as u128 & word_mask) >> shift;
    let sign_bit = (result >> (half - 1)) & 1;
    (result + sign_bit * ((1u128 << XLEN) - (1u128 << half))) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    #[test]
    fn materialize_entry_virtualsrlw() {
        materialize_entry_test!(VirtualSrlw, tracer::instruction::virtual_srlw::VirtualSRLW);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsrlw() {
        instruction_inputs_match_constraint_test!(
            VirtualSrlw,
            tracer::instruction::virtual_srlw::VirtualSRLW
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualsrlw() {
        lookup_output_matches_trace_test!(
            VirtualSrlw,
            tracer::instruction::virtual_srlw::VirtualSRLW
        );
    }
}
