use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::{
    VirtualXorRot10, VirtualXorRot14, VirtualXorRot18, VirtualXorRot21, VirtualXorRot22,
    VirtualXorRot23, VirtualXorRot26, VirtualXorRot27, VirtualXorRot3, VirtualXorRot33,
    VirtualXorRot4, VirtualXorRot43, VirtualXorRot44, VirtualXorRot45, VirtualXorRot47,
    VirtualXorRot49, VirtualXorRot52, VirtualXorRot6, VirtualXorRot60, VirtualXorRot61,
    VirtualXorRot9,
};
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

macro_rules! impl_keccak_xor_rot {
    ($name:ident, $table:ident, $rotation:literal) => {
        impl_lookup_table!($name, Some($table));
        impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for $name<C> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                (
                    self.0.rs1_val().unwrap_or(0),
                    self.0.rs2_val().unwrap_or(0) as i128,
                )
            }
            fn to_lookup_output(&self) -> u64 {
                let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
                let value = ((x ^ y as u64) & mask) as u128;
                let rotation = $rotation % XLEN;
                (((value >> rotation) | (value << (XLEN - rotation))) as u64) & mask
            }
        }
    };
}
impl_keccak_xor_rot!(VirtualXorRot3, VirtualXORROT3, 3);
impl_keccak_xor_rot!(VirtualXorRot4, VirtualXORROT4, 4);
impl_keccak_xor_rot!(VirtualXorRot6, VirtualXORROT6, 6);
impl_keccak_xor_rot!(VirtualXorRot9, VirtualXORROT9, 9);
impl_keccak_xor_rot!(VirtualXorRot10, VirtualXORROT10, 10);
impl_keccak_xor_rot!(VirtualXorRot14, VirtualXORROT14, 14);
impl_keccak_xor_rot!(VirtualXorRot18, VirtualXORROT18, 18);
impl_keccak_xor_rot!(VirtualXorRot21, VirtualXORROT21, 21);
impl_keccak_xor_rot!(VirtualXorRot22, VirtualXORROT22, 22);
impl_keccak_xor_rot!(VirtualXorRot23, VirtualXORROT23, 23);
impl_keccak_xor_rot!(VirtualXorRot26, VirtualXORROT26, 26);
impl_keccak_xor_rot!(VirtualXorRot27, VirtualXORROT27, 27);
impl_keccak_xor_rot!(VirtualXorRot33, VirtualXORROT33, 33);
impl_keccak_xor_rot!(VirtualXorRot43, VirtualXORROT43, 43);
impl_keccak_xor_rot!(VirtualXorRot44, VirtualXORROT44, 44);
impl_keccak_xor_rot!(VirtualXorRot45, VirtualXORROT45, 45);
impl_keccak_xor_rot!(VirtualXorRot47, VirtualXORROT47, 47);
impl_keccak_xor_rot!(VirtualXorRot49, VirtualXORROT49, 49);
impl_keccak_xor_rot!(VirtualXorRot52, VirtualXORROT52, 52);
impl_keccak_xor_rot!(VirtualXorRot60, VirtualXORROT60, 60);
impl_keccak_xor_rot!(VirtualXorRot61, VirtualXORROT61, 61);

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
