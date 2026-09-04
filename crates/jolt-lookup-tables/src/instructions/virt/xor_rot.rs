use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::{
    VirtualXorRot16, VirtualXorRot19, VirtualXorRot2, VirtualXorRot20, VirtualXorRot21,
    VirtualXorRot23, VirtualXorRot24, VirtualXorRot25, VirtualXorRot28, VirtualXorRot3,
    VirtualXorRot32, VirtualXorRot36, VirtualXorRot37, VirtualXorRot39, VirtualXorRot43,
    VirtualXorRot44, VirtualXorRot46, VirtualXorRot49, VirtualXorRot50, VirtualXorRot54,
    VirtualXorRot56, VirtualXorRot58, VirtualXorRot61, VirtualXorRot62, VirtualXorRot63,
    VirtualXorRot8, VirtualXorRot9,
};
use jolt_riscv::JoltCycle;

macro_rules! impl_xor_rot_query {
    ($instr:ident, $table:ident, $rotation:expr) => {
        impl_lookup_table!($instr, Some($table));
        impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for $instr<C> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                (
                    self.0.rs1_val().unwrap_or(0),
                    self.0.rs2_val().unwrap_or(0) as i128,
                )
            }

            fn to_lookup_output(&self) -> u64 {
                let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
                let xor_result = (rs1 ^ (rs2 as u64)) & mask;
                let v = xor_result as u128;
                (((v >> $rotation) | (v << (XLEN - $rotation))) as u64) & mask
            }
        }
    };
}

impl_xor_rot_query!(VirtualXorRot2, VirtualXORROT2, 2);
impl_xor_rot_query!(VirtualXorRot3, VirtualXORROT3, 3);
impl_xor_rot_query!(VirtualXorRot8, VirtualXORROT8, 8);
impl_xor_rot_query!(VirtualXorRot9, VirtualXORROT9, 9);
impl_xor_rot_query!(VirtualXorRot16, VirtualXORROT16, 16);
impl_xor_rot_query!(VirtualXorRot19, VirtualXORROT19, 19);
impl_xor_rot_query!(VirtualXorRot20, VirtualXORROT20, 20);
impl_xor_rot_query!(VirtualXorRot21, VirtualXORROT21, 21);
impl_xor_rot_query!(VirtualXorRot23, VirtualXORROT23, 23);
impl_xor_rot_query!(VirtualXorRot24, VirtualXORROT24, 24);
impl_xor_rot_query!(VirtualXorRot25, VirtualXORROT25, 25);
impl_xor_rot_query!(VirtualXorRot28, VirtualXORROT28, 28);
impl_xor_rot_query!(VirtualXorRot32, VirtualXORROT32, 32);
impl_xor_rot_query!(VirtualXorRot36, VirtualXORROT36, 36);
impl_xor_rot_query!(VirtualXorRot37, VirtualXORROT37, 37);
impl_xor_rot_query!(VirtualXorRot39, VirtualXORROT39, 39);
impl_xor_rot_query!(VirtualXorRot43, VirtualXORROT43, 43);
impl_xor_rot_query!(VirtualXorRot44, VirtualXORROT44, 44);
impl_xor_rot_query!(VirtualXorRot46, VirtualXORROT46, 46);
impl_xor_rot_query!(VirtualXorRot49, VirtualXORROT49, 49);
impl_xor_rot_query!(VirtualXorRot50, VirtualXORROT50, 50);
impl_xor_rot_query!(VirtualXorRot54, VirtualXORROT54, 54);
impl_xor_rot_query!(VirtualXorRot56, VirtualXORROT56, 56);
impl_xor_rot_query!(VirtualXorRot58, VirtualXORROT58, 58);
impl_xor_rot_query!(VirtualXorRot61, VirtualXORROT61, 61);
impl_xor_rot_query!(VirtualXorRot62, VirtualXORROT62, 62);
impl_xor_rot_query!(VirtualXorRot63, VirtualXORROT63, 63);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    #[test]
    fn materialize_entry_virtualxorrot32() {
        materialize_entry_test!(
            VirtualXorRot32,
            tracer::instruction::virtual_xor_rot::VirtualXORROT32
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot32() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRot32,
            tracer::instruction::virtual_xor_rot::VirtualXORROT32
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot32() {
        lookup_output_matches_trace_test!(
            VirtualXorRot32,
            tracer::instruction::virtual_xor_rot::VirtualXORROT32
        );
    }

    #[test]
    fn materialize_entry_virtualxorrot24() {
        materialize_entry_test!(
            VirtualXorRot24,
            tracer::instruction::virtual_xor_rot::VirtualXORROT24
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot24() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRot24,
            tracer::instruction::virtual_xor_rot::VirtualXORROT24
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot24() {
        lookup_output_matches_trace_test!(
            VirtualXorRot24,
            tracer::instruction::virtual_xor_rot::VirtualXORROT24
        );
    }

    #[test]
    fn materialize_entry_virtualxorrot16() {
        materialize_entry_test!(
            VirtualXorRot16,
            tracer::instruction::virtual_xor_rot::VirtualXORROT16
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot16() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRot16,
            tracer::instruction::virtual_xor_rot::VirtualXORROT16
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot16() {
        lookup_output_matches_trace_test!(
            VirtualXorRot16,
            tracer::instruction::virtual_xor_rot::VirtualXORROT16
        );
    }

    #[test]
    fn materialize_entry_virtualxorrot63() {
        materialize_entry_test!(
            VirtualXorRot63,
            tracer::instruction::virtual_xor_rot::VirtualXORROT63
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot63() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRot63,
            tracer::instruction::virtual_xor_rot::VirtualXORROT63
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot63() {
        lookup_output_matches_trace_test!(
            VirtualXorRot63,
            tracer::instruction::virtual_xor_rot::VirtualXORROT63
        );
    }
}
