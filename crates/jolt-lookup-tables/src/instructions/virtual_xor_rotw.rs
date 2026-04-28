use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::{
    VirtualXorRotW12, VirtualXorRotW16, VirtualXorRotW7, VirtualXorRotW8,
};
use tracer::instruction::{
    virtual_xor_rotw::{VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8},
    RISCVCycle,
};

impl_lookup_table!(VirtualXorRotW16, Some(VirtualXORROTW16));
impl_lookup_table!(VirtualXorRotW12, Some(VirtualXORROTW12));
impl_lookup_table!(VirtualXorRotW8, Some(VirtualXORROTW8));
impl_lookup_table!(VirtualXorRotW7, Some(VirtualXORROTW7));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROTW16> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let half = XLEN / 2;
        let mask = (1u128 << half).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 16) | (v << (half - 16))) as u64) & mask
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROTW12> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let half = XLEN / 2;
        let mask = (1u128 << half).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 12) | (v << (half - 12))) as u64) & mask
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROTW8> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let half = XLEN / 2;
        let mask = (1u128 << half).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 8) | (v << (half - 8))) as u64) & mask
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROTW7> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let half = XLEN / 2;
        let mask = (1u128 << half).wrapping_sub(1) as u64;
        let xor_result = (rs1 ^ (rs2 as u64)) & mask;
        let v = xor_result as u128;
        (((v >> 7) | (v << (half - 7))) as u64) & mask
    }
}
