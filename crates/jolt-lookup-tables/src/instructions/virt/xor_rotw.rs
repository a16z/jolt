use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::{
    VirtualXorRotW12, VirtualXorRotW16, VirtualXorRotW7, VirtualXorRotW8,
};
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualXorRotW16, Some(VirtualXORROTW16));
impl_lookup_table!(VirtualXorRotW12, Some(VirtualXORROTW12));
impl_lookup_table!(VirtualXorRotW8, Some(VirtualXORROTW8));
impl_lookup_table!(VirtualXorRotW7, Some(VirtualXORROTW7));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW16<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
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

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW12<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
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

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW8<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
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

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW7<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::test::materialize_entry_test;
    use tracer::instruction::RISCVCycle;

    #[test]
    fn materialize_entry_virtualxorrotw16() {
        materialize_entry_test::<
            VirtualXorRotW16<RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW16>>,
            RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW16>,
        >();
    }

    #[test]
    fn materialize_entry_virtualxorrotw12() {
        materialize_entry_test::<
            VirtualXorRotW12<RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW12>>,
            RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW12>,
        >();
    }

    #[test]
    fn materialize_entry_virtualxorrotw8() {
        materialize_entry_test::<
            VirtualXorRotW8<RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW8>>,
            RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW8>,
        >();
    }

    #[test]
    fn materialize_entry_virtualxorrotw7() {
        materialize_entry_test::<
            VirtualXorRotW7<RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW7>>,
            RISCVCycle<tracer::instruction::virtual_xor_rotw::VirtualXORROTW7>,
        >();
    }
}
