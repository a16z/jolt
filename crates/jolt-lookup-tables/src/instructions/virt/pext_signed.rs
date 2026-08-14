use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::PextSigned;
use jolt_riscv::JoltCycle;

impl_lookup_table!(PextSigned, Some(PextSigned));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for PextSigned<C> {
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
        let mut sign = 0u64;
        let mut pc = 0u32;
        for i in (0..XLEN).rev() {
            if (y >> i) & 1 == 1 {
                let x_i = (x >> i) & 1;
                if pc == 0 {
                    sign = x_i;
                }
                pext = (pext << 1) | x_i;
                pc += 1;
            }
        }
        let ext = if sign == 1 {
            ((1u128 << XLEN) - (1u128 << pc)) as u64
        } else {
            0
        };
        pext + ext
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
    fn materialize_entry_pextsigned() {
        materialize_entry_test!(
            PextSigned,
            tracer::instruction::virtual_pext_signed::VirtualPextSigned
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_pextsigned() {
        instruction_inputs_match_constraint_test!(
            PextSigned,
            tracer::instruction::virtual_pext_signed::VirtualPextSigned
        );
    }

    #[test]
    fn lookup_output_matches_trace_pextsigned() {
        lookup_output_matches_trace_test!(
            PextSigned,
            tracer::instruction::virtual_pext_signed::VirtualPextSigned
        );
    }
}
