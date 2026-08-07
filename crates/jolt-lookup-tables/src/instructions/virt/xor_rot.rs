use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::{
    VirtualXorRot16, VirtualXorRot24, VirtualXorRot32, VirtualXorRot63,
};
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualXorRot32, Some(VirtualXORROT32));
impl_lookup_table!(VirtualXorRot24, Some(VirtualXORROT24));
impl_lookup_table!(VirtualXorRot16, Some(VirtualXORROT16));
impl_lookup_table!(VirtualXorRot63, Some(VirtualXORROT63));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRot32<C> {
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
        (((v >> 32) | (v << (XLEN - 32))) as u64) & mask
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRot24<C> {
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
        (((v >> 24) | (v << (XLEN - 24))) as u64) & mask
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRot16<C> {
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
        (((v >> 16) | (v << (XLEN - 16))) as u64) & mask
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRot63<C> {
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
        (((v >> 63) | (v << (XLEN - 63))) as u64) & mask
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

use jolt_riscv::instructions::{
    VirtualXorRot19, VirtualXorRot2, VirtualXorRot20, VirtualXorRot21, VirtualXorRot23,
    VirtualXorRot25, VirtualXorRot28, VirtualXorRot3, VirtualXorRot36, VirtualXorRot37,
    VirtualXorRot39, VirtualXorRot43, VirtualXorRot44, VirtualXorRot46, VirtualXorRot49,
    VirtualXorRot50, VirtualXorRot54, VirtualXorRot56, VirtualXorRot58, VirtualXorRot61,
    VirtualXorRot62, VirtualXorRot8, VirtualXorRot9,
};

/// `LookupQuery` for the Keccak rho-fusion XOR-ROT family: identical to the
/// Blake variants above, generic over the rotation amount.
macro_rules! impl_keccak_xor_rot_query {
    ($instr:ident, $rotation:expr) => {
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

impl_lookup_table!(VirtualXorRot2, Some(VirtualXORROT2));
impl_lookup_table!(VirtualXorRot3, Some(VirtualXORROT3));
impl_lookup_table!(VirtualXorRot8, Some(VirtualXORROT8));
impl_lookup_table!(VirtualXorRot9, Some(VirtualXORROT9));
impl_lookup_table!(VirtualXorRot19, Some(VirtualXORROT19));
impl_lookup_table!(VirtualXorRot20, Some(VirtualXORROT20));
impl_lookup_table!(VirtualXorRot21, Some(VirtualXORROT21));
impl_lookup_table!(VirtualXorRot23, Some(VirtualXORROT23));
impl_lookup_table!(VirtualXorRot25, Some(VirtualXORROT25));
impl_lookup_table!(VirtualXorRot28, Some(VirtualXORROT28));
impl_lookup_table!(VirtualXorRot36, Some(VirtualXORROT36));
impl_lookup_table!(VirtualXorRot37, Some(VirtualXORROT37));
impl_lookup_table!(VirtualXorRot39, Some(VirtualXORROT39));
impl_lookup_table!(VirtualXorRot43, Some(VirtualXORROT43));
impl_lookup_table!(VirtualXorRot44, Some(VirtualXORROT44));
impl_lookup_table!(VirtualXorRot46, Some(VirtualXORROT46));
impl_lookup_table!(VirtualXorRot49, Some(VirtualXORROT49));
impl_lookup_table!(VirtualXorRot50, Some(VirtualXORROT50));
impl_lookup_table!(VirtualXorRot54, Some(VirtualXORROT54));
impl_lookup_table!(VirtualXorRot56, Some(VirtualXORROT56));
impl_lookup_table!(VirtualXorRot58, Some(VirtualXORROT58));
impl_lookup_table!(VirtualXorRot61, Some(VirtualXORROT61));
impl_lookup_table!(VirtualXorRot62, Some(VirtualXORROT62));

impl_keccak_xor_rot_query!(VirtualXorRot2, 2);
impl_keccak_xor_rot_query!(VirtualXorRot3, 3);
impl_keccak_xor_rot_query!(VirtualXorRot8, 8);
impl_keccak_xor_rot_query!(VirtualXorRot9, 9);
impl_keccak_xor_rot_query!(VirtualXorRot19, 19);
impl_keccak_xor_rot_query!(VirtualXorRot20, 20);
impl_keccak_xor_rot_query!(VirtualXorRot21, 21);
impl_keccak_xor_rot_query!(VirtualXorRot23, 23);
impl_keccak_xor_rot_query!(VirtualXorRot25, 25);
impl_keccak_xor_rot_query!(VirtualXorRot28, 28);
impl_keccak_xor_rot_query!(VirtualXorRot36, 36);
impl_keccak_xor_rot_query!(VirtualXorRot37, 37);
impl_keccak_xor_rot_query!(VirtualXorRot39, 39);
impl_keccak_xor_rot_query!(VirtualXorRot43, 43);
impl_keccak_xor_rot_query!(VirtualXorRot44, 44);
impl_keccak_xor_rot_query!(VirtualXorRot46, 46);
impl_keccak_xor_rot_query!(VirtualXorRot49, 49);
impl_keccak_xor_rot_query!(VirtualXorRot50, 50);
impl_keccak_xor_rot_query!(VirtualXorRot54, 54);
impl_keccak_xor_rot_query!(VirtualXorRot56, 56);
impl_keccak_xor_rot_query!(VirtualXorRot58, 58);
impl_keccak_xor_rot_query!(VirtualXorRot61, 61);
impl_keccak_xor_rot_query!(VirtualXorRot62, 62);
