use crate::tables::virtual_xor_rot::rotate_right_xlen;
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
use paste::paste;

/// Every XOR-then-rotate-right instruction, keyed by its rotation: the
/// instruction type, its lookup table variant, and its lookup query.
macro_rules! impl_xor_rot_query {
    ($($rotation:literal),+ $(,)?) => {
        paste! {
            $(
                impl_lookup_table!([<VirtualXorRot $rotation>], Some([<VirtualXORROT $rotation>]));
                impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for [<VirtualXorRot $rotation>]<C> {
                    fn to_instruction_inputs(&self) -> (u64, i128) {
                        (
                            self.0.rs1_val().unwrap_or(0),
                            self.0.rs2_val().unwrap_or(0) as i128,
                        )
                    }

                    fn to_lookup_output(&self) -> u64 {
                        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                        rotate_right_xlen::<XLEN>(rs1 ^ (rs2 as u64), $rotation)
                    }
                }
            )+
        }
    };
}

impl_xor_rot_query!(
    2, 3, 8, 9, 16, 19, 20, 21, 23, 24, 25, 28, 32, 36, 37, 39, 43, 44, 46, 49, 50, 54, 56, 58, 61,
    62, 63,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    macro_rules! xor_rot_query_tests {
        ($($rotation:literal),+ $(,)?) => {
            paste! {
                $(
                    #[test]
                    fn [<materialize_entry_virtualxorrot $rotation>]() {
                        materialize_entry_test!(
                            [<VirtualXorRot $rotation>],
                            tracer::instruction::virtual_xor_rot::[<VirtualXORROT $rotation>]
                        );
                    }

                    #[test]
                    fn [<instruction_inputs_match_constraint_virtualxorrot $rotation>]() {
                        instruction_inputs_match_constraint_test!(
                            [<VirtualXorRot $rotation>],
                            tracer::instruction::virtual_xor_rot::[<VirtualXORROT $rotation>]
                        );
                    }

                    #[test]
                    fn [<lookup_output_matches_trace_virtualxorrot $rotation>]() {
                        lookup_output_matches_trace_test!(
                            [<VirtualXorRot $rotation>],
                            tracer::instruction::virtual_xor_rot::[<VirtualXORROT $rotation>]
                        );
                    }
                )+
            }
        };
    }

    xor_rot_query_tests!(
        2, 3, 8, 9, 16, 19, 20, 21, 23, 24, 25, 28, 32, 36, 37, 39, 43, 44, 46, 49, 50, 54, 56, 58,
        61, 62, 63,
    );
}
