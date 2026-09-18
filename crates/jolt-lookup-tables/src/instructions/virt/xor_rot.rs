use crate::tables::virtual_xor_rot::rotate_right_xlen;
use crate::tables::LookupTableKind;
use crate::traits::{InstructionLookupTable, LookupQuery};
use jolt_riscv::instructions::VirtualXorRot;
use jolt_riscv::{JoltCycle, JoltInstructionRow, JoltInstructionRowData};

impl<const XLEN: usize, T: JoltInstructionRowData> InstructionLookupTable<XLEN>
    for VirtualXorRot<T>
{
    #[inline]
    fn lookup_table(&self) -> Option<LookupTableKind<XLEN>> {
        Some(LookupTableKind::xor_rot(
            self.0.jolt_instruction_row().operands.imm as u32,
        ))
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRot<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let rotation = Into::<JoltInstructionRow>::into(self.0.instruction())
            .operands
            .imm as u32;
        rotate_right_xlen::<XLEN>(rs1 ^ (rs2 as u64), rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test, XLEN,
    };
    use jolt_riscv::instructions::XOR_ROT_ROTATIONS;
    use tracer::instruction::virtual_xor_rot::VirtualXORROT;

    #[test]
    fn materialize_entry_virtualxorrot() {
        materialize_entry_test!(VirtualXorRot, VirtualXORROT);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot() {
        instruction_inputs_match_constraint_test!(VirtualXorRot, VirtualXORROT);
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot() {
        lookup_output_matches_trace_test!(VirtualXorRot, VirtualXORROT);
    }

    #[test]
    fn every_supported_rotation_selects_its_table() {
        for rotation in XOR_ROT_ROTATIONS {
            let table = LookupTableKind::<XLEN>::xor_rot(rotation);
            let index = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
            let (x, y) = crate::uninterleave_bits(index);
            assert_eq!(
                table.materialize_entry(index),
                rotate_right_xlen::<XLEN>(x ^ y, rotation),
                "rotation {rotation}"
            );
        }
    }
}
