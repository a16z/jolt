use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::Pext;
use jolt_riscv::JoltCycle;

impl_lookup_table!(Pext, Some(Pext));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Pext<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let (x, y) = (rs1 & mask, rs2 as u64 & mask);

        let mut pext = 0u64;
        for i in (0..XLEN).rev() {
            if (y >> i) & 1 == 1 {
                pext = (pext << 1) | ((x >> i) & 1);
            }
        }
        pext
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_pext::VirtualPext;

    #[test]
    fn materialize_entry_pext() {
        materialize_entry_test!(Pext, VirtualPext);
    }

    #[test]
    fn instruction_inputs_match_constraint_pext() {
        instruction_inputs_match_constraint_test!(Pext, VirtualPext);
    }

    #[test]
    fn lookup_output_matches_trace_pext() {
        lookup_output_matches_trace_test!(Pext, VirtualPext);
    }
}
