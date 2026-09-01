use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::{
    VirtualXorRotW12, VirtualXorRotW16, VirtualXorRotW19, VirtualXorRotW22, VirtualXorRotW6,
    VirtualXorRotW7, VirtualXorRotW8,
};
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualXorRotW16, Some(VirtualXORROTW16));
impl_lookup_table!(VirtualXorRotW12, Some(VirtualXORROTW12));
impl_lookup_table!(VirtualXorRotW8, Some(VirtualXORROTW8));
impl_lookup_table!(VirtualXorRotW7, Some(VirtualXORROTW7));
impl_lookup_table!(VirtualXorRotW22, Some(VirtualXORROTW22));
impl_lookup_table!(VirtualXorRotW19, Some(VirtualXORROTW19));
impl_lookup_table!(VirtualXorRotW6, Some(VirtualXORROTW6));

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

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW22<C> {
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
        (((v >> 22) | (v << (half - 22))) as u64) & mask
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW19<C> {
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
        (((v >> 19) | (v << (half - 19))) as u64) & mask
    }
}

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualXorRotW6<C> {
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
        (((v >> 6) | (v << (half - 6))) as u64) & mask
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
    fn materialize_entry_virtualxorrotw16() {
        materialize_entry_test!(
            VirtualXorRotW16,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW16
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw16() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW16,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW16
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw16() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW16,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW16
        );
    }

    #[test]
    fn materialize_entry_virtualxorrotw12() {
        materialize_entry_test!(
            VirtualXorRotW12,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW12
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw12() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW12,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW12
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw12() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW12,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW12
        );
    }

    #[test]
    fn materialize_entry_virtualxorrotw8() {
        materialize_entry_test!(
            VirtualXorRotW8,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW8
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw8() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW8,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW8
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw8() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW8,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW8
        );
    }

    #[test]
    fn materialize_entry_virtualxorrotw7() {
        materialize_entry_test!(
            VirtualXorRotW7,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW7
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw7() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW7,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW7
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw7() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW7,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW7
        );
    }

    #[test]
    fn materialize_entry_virtualxorrotw22() {
        materialize_entry_test!(
            VirtualXorRotW22,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW22
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw22() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW22,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW22
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw22() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW22,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW22
        );
    }

    #[test]
    fn materialize_entry_virtualxorrotw19() {
        materialize_entry_test!(
            VirtualXorRotW19,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW19
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw19() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW19,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW19
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw19() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW19,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW19
        );
    }

    #[test]
    fn materialize_entry_virtualxorrotw6() {
        materialize_entry_test!(
            VirtualXorRotW6,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW6
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrotw6() {
        instruction_inputs_match_constraint_test!(
            VirtualXorRotW6,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW6
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrotw6() {
        lookup_output_matches_trace_test!(
            VirtualXorRotW6,
            tracer::instruction::virtual_xor_rotw::VirtualXORROTW6
        );
    }
}
