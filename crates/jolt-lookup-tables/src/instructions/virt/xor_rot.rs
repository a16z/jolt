use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::{
    VirtualXorRot16, VirtualXorRot24, VirtualXorRot32, VirtualXorRot63,
};
use jolt_trace::JoltCycle;

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
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualxorrot32() {
        materialize_entry_test!(VirtualXorRot32, tracer::instruction::virtual_xor_rot::VirtualXORROT32);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot32() {
        instruction_inputs_match_constraint_test!(VirtualXorRot32, tracer::instruction::virtual_xor_rot::VirtualXORROT32);
    }

    #[test]
    fn materialize_entry_virtualxorrot24() {
        materialize_entry_test!(VirtualXorRot24, tracer::instruction::virtual_xor_rot::VirtualXORROT24);
    }

    #[test]
    fn materialize_entry_virtualxorrot16() {
        materialize_entry_test!(VirtualXorRot16, tracer::instruction::virtual_xor_rot::VirtualXORROT16);
    }

    #[test]
    fn materialize_entry_virtualxorrot63() {
        materialize_entry_test!(VirtualXorRot63, tracer::instruction::virtual_xor_rot::VirtualXORROT63);
    }
}
