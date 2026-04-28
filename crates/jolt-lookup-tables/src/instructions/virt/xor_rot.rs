use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::{
    VirtualXorRot16, VirtualXorRot24, VirtualXorRot32, VirtualXorRot63,
};
use tracer::instruction::{
    virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63},
    RISCVCycle,
};

impl_lookup_table!(VirtualXorRot32, Some(VirtualXORROT32));
impl_lookup_table!(VirtualXorRot24, Some(VirtualXORROT24));
impl_lookup_table!(VirtualXorRot16, Some(VirtualXORROT16));
impl_lookup_table!(VirtualXorRot63, Some(VirtualXORROT63));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROT32> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 32) | (v << (XLEN - 32))) as u64) & mask
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROT24> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 24) | (v << (XLEN - 24))) as u64) & mask
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROT16> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 16) | (v << (XLEN - 16))) as u64) & mask
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROT63> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 63) | (v << (XLEN - 63))) as u64) & mask
    }
}
