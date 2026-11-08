use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::{
    VirtualXorRot16, VirtualXorRot19, VirtualXorRot2, VirtualXorRot20, VirtualXorRot24,
    VirtualXorRot3, VirtualXorRot32, VirtualXorRot61, VirtualXorRot63, VirtualXorRot8,
    VirtualXorRot9,
};
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualXorRot32, Some(VirtualXORROT32));
impl_lookup_table!(VirtualXorRot24, Some(VirtualXORROT24));
impl_lookup_table!(VirtualXorRot16, Some(VirtualXORROT16));
impl_lookup_table!(VirtualXorRot63, Some(VirtualXORROT63));
impl_lookup_table!(VirtualXorRot2, Some(VirtualXORROT2));
impl_lookup_table!(VirtualXorRot61, Some(VirtualXORROT61));
impl_lookup_table!(VirtualXorRot3, Some(VirtualXORROT3));
impl_lookup_table!(VirtualXorRot8, Some(VirtualXORROT8));
impl_lookup_table!(VirtualXorRot9, Some(VirtualXORROT9));
impl_lookup_table!(VirtualXorRot19, Some(VirtualXORROT19));
impl_lookup_table!(VirtualXorRot20, Some(VirtualXORROT20));

macro_rules! impl_xor_rot_query {
    ($instruction:ident, $rotation:expr) => {
        impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for $instruction<C> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                (
                    self.0.rs1_val().unwrap_or(0),
                    self.0.rs2_val().unwrap_or(0) as i128,
                )
            }

            fn to_lookup_output(&self) -> u64 {
                let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
                let v = ((rs1 ^ (rs2 as u64)) & mask) as u128;
                let rotation = $rotation % XLEN;
                (((v >> rotation) | (v << (XLEN - rotation))) as u64) & mask
            }
        }
    };
}

impl_xor_rot_query!(VirtualXorRot32, 32);
impl_xor_rot_query!(VirtualXorRot24, 24);
impl_xor_rot_query!(VirtualXorRot16, 16);
impl_xor_rot_query!(VirtualXorRot63, 63);
impl_xor_rot_query!(VirtualXorRot2, 2);
impl_xor_rot_query!(VirtualXorRot61, 61);
impl_xor_rot_query!(VirtualXorRot3, 3);
impl_xor_rot_query!(VirtualXorRot8, 8);
impl_xor_rot_query!(VirtualXorRot9, 9);
impl_xor_rot_query!(VirtualXorRot19, 19);
impl_xor_rot_query!(VirtualXorRot20, 20);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_xor_rot::{VirtualXORROT2, VirtualXORROT61};

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

    #[test]
    fn materialize_entry_virtualxorrot2() {
        materialize_entry_test!(VirtualXorRot2, VirtualXORROT2);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot2() {
        instruction_inputs_match_constraint_test!(VirtualXorRot2, VirtualXORROT2);
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot2() {
        lookup_output_matches_trace_test!(VirtualXorRot2, VirtualXORROT2);
    }

    #[test]
    fn materialize_entry_virtualxorrot61() {
        materialize_entry_test!(VirtualXorRot61, VirtualXORROT61);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualxorrot61() {
        instruction_inputs_match_constraint_test!(VirtualXorRot61, VirtualXORROT61);
    }

    #[test]
    fn lookup_output_matches_trace_virtualxorrot61() {
        lookup_output_matches_trace_test!(VirtualXorRot61, VirtualXORROT61);
    }
    #[test]
    fn xorrot3_lookup_and_trace() {
        use tracer::instruction::virtual_xor_rot::VirtualXORROT3;
        materialize_entry_test!(VirtualXorRot3, VirtualXORROT3);
        instruction_inputs_match_constraint_test!(VirtualXorRot3, VirtualXORROT3);
        lookup_output_matches_trace_test!(VirtualXorRot3, VirtualXORROT3);
    }

    #[test]
    fn xorrot8_lookup_and_trace() {
        use tracer::instruction::virtual_xor_rot::VirtualXORROT8;
        materialize_entry_test!(VirtualXorRot8, VirtualXORROT8);
        instruction_inputs_match_constraint_test!(VirtualXorRot8, VirtualXORROT8);
        lookup_output_matches_trace_test!(VirtualXorRot8, VirtualXORROT8);
    }

    #[test]
    fn xorrot9_lookup_and_trace() {
        use tracer::instruction::virtual_xor_rot::VirtualXORROT9;
        materialize_entry_test!(VirtualXorRot9, VirtualXORROT9);
        instruction_inputs_match_constraint_test!(VirtualXorRot9, VirtualXORROT9);
        lookup_output_matches_trace_test!(VirtualXorRot9, VirtualXORROT9);
    }

    #[test]
    fn xorrot19_lookup_and_trace() {
        use tracer::instruction::virtual_xor_rot::VirtualXORROT19;
        materialize_entry_test!(VirtualXorRot19, VirtualXORROT19);
        instruction_inputs_match_constraint_test!(VirtualXorRot19, VirtualXORROT19);
        lookup_output_matches_trace_test!(VirtualXorRot19, VirtualXORROT19);
    }

    #[test]
    fn xorrot20_lookup_and_trace() {
        use tracer::instruction::virtual_xor_rot::VirtualXORROT20;
        materialize_entry_test!(VirtualXorRot20, VirtualXORROT20);
        instruction_inputs_match_constraint_test!(VirtualXorRot20, VirtualXORROT20);
        lookup_output_matches_trace_test!(VirtualXorRot20, VirtualXORROT20);
    }
}
