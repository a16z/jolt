use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::virtual_xor_rot::{
    VirtualXORROT19, VirtualXORROT2, VirtualXORROT20, VirtualXORROT21, VirtualXORROT23,
    VirtualXORROT25, VirtualXORROT28, VirtualXORROT3, VirtualXORROT36, VirtualXORROT37,
    VirtualXORROT39, VirtualXORROT43, VirtualXORROT44, VirtualXORROT46, VirtualXORROT49,
    VirtualXORROT50, VirtualXORROT54, VirtualXORROT56, VirtualXORROT58, VirtualXORROT61,
    VirtualXORROT62, VirtualXORROT8, VirtualXORROT9,
};
use tracer::instruction::{
    virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63},
    RISCVCycle,
};

use crate::zkvm::lookup_table::{virtual_xor_rot::VirtualXORROTTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

// Macro to implement traits for each specific rotation value
macro_rules! impl_virtual_xor_rot {
    ($type:ty, $rotation:expr) => {
        impl<const XLEN: usize> InstructionLookup<XLEN> for $type {
            fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
                Some(VirtualXORROTTable::<XLEN, $rotation>.into())
            }
        }

        impl Flags for $type {
            fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
                let mut flags = [false; NUM_CIRCUIT_FLAGS];
                flags[CircuitFlags::WriteLookupOutputToRD] = true;
                flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
                flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
                    self.virtual_sequence_remaining.unwrap_or(0) != 0;
                flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
                flags[CircuitFlags::IsCompressed] = self.is_compressed;
                flags
            }

            fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
                let mut flags = [false; NUM_INSTRUCTION_FLAGS];
                flags[InstructionFlags::LeftOperandIsRs1Value] = true;
                flags[InstructionFlags::RightOperandIsRs2Value] = true;
                flags
            }
        }

        impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<$type> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                match XLEN {
                    #[cfg(test)]
                    8 => (
                        self.register_state.rs1 as u8 as u64,
                        self.register_state.rs2 as u8 as i128,
                    ),
                    64 => (self.register_state.rs1, self.register_state.rs2 as i128),
                    _ => panic!("{XLEN}-bit word size is unsupported"),
                }
            }

            fn to_lookup_output(&self) -> u64 {
                let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                match XLEN {
                    #[cfg(test)]
                    8 => {
                        let xor_result = (x as u8) ^ (y as u8);
                        xor_result.rotate_right($rotation).into()
                    }
                    64 => {
                        let xor_result = x ^ (y as u64);
                        xor_result.rotate_right($rotation)
                    }
                    _ => panic!("{XLEN}-bit word size is unsupported"),
                }
            }
        }
    };
}

// Implement for each specific rotation value
impl_virtual_xor_rot!(VirtualXORROT32, 32);
impl_virtual_xor_rot!(VirtualXORROT24, 24);
impl_virtual_xor_rot!(VirtualXORROT16, 16);
impl_virtual_xor_rot!(VirtualXORROT63, 63);
impl_virtual_xor_rot!(VirtualXORROT2, 2);
impl_virtual_xor_rot!(VirtualXORROT3, 3);
impl_virtual_xor_rot!(VirtualXORROT8, 8);
impl_virtual_xor_rot!(VirtualXORROT9, 9);
impl_virtual_xor_rot!(VirtualXORROT19, 19);
impl_virtual_xor_rot!(VirtualXORROT20, 20);
impl_virtual_xor_rot!(VirtualXORROT21, 21);
impl_virtual_xor_rot!(VirtualXORROT23, 23);
impl_virtual_xor_rot!(VirtualXORROT25, 25);
impl_virtual_xor_rot!(VirtualXORROT28, 28);
impl_virtual_xor_rot!(VirtualXORROT36, 36);
impl_virtual_xor_rot!(VirtualXORROT37, 37);
impl_virtual_xor_rot!(VirtualXORROT39, 39);
impl_virtual_xor_rot!(VirtualXORROT43, 43);
impl_virtual_xor_rot!(VirtualXORROT44, 44);
impl_virtual_xor_rot!(VirtualXORROT46, 46);
impl_virtual_xor_rot!(VirtualXORROT49, 49);
impl_virtual_xor_rot!(VirtualXORROT50, 50);
impl_virtual_xor_rot!(VirtualXORROT54, 54);
impl_virtual_xor_rot!(VirtualXORROT56, 56);
impl_virtual_xor_rot!(VirtualXORROT58, 58);
impl_virtual_xor_rot!(VirtualXORROT61, 61);
impl_virtual_xor_rot!(VirtualXORROT62, 62);

#[cfg(test)]
mod test {
    use super::*;
    use crate::zkvm::instruction::test::{
        lookup_output_matches_trace_test, materialize_entry_test,
    };
    use ark_bn254::Fr;

    macro_rules! xor_rot_query_tests {
        ($($type:ident => ($materialize:ident, $trace:ident)),+ $(,)?) => {
            $(
                #[test]
                fn $materialize() {
                    materialize_entry_test::<Fr, $type>();
                }

                #[test]
                fn $trace() {
                    lookup_output_matches_trace_test::<$type>();
                }
            )+
        };
    }

    xor_rot_query_tests!(
        VirtualXORROT2 => (materialize_entry_2, lookup_output_matches_trace_2),
        VirtualXORROT3 => (materialize_entry_3, lookup_output_matches_trace_3),
        VirtualXORROT8 => (materialize_entry_8, lookup_output_matches_trace_8),
        VirtualXORROT9 => (materialize_entry_9, lookup_output_matches_trace_9),
        VirtualXORROT16 => (materialize_entry_16, lookup_output_matches_trace_16),
        VirtualXORROT19 => (materialize_entry_19, lookup_output_matches_trace_19),
        VirtualXORROT20 => (materialize_entry_20, lookup_output_matches_trace_20),
        VirtualXORROT21 => (materialize_entry_21, lookup_output_matches_trace_21),
        VirtualXORROT23 => (materialize_entry_23, lookup_output_matches_trace_23),
        VirtualXORROT24 => (materialize_entry_24, lookup_output_matches_trace_24),
        VirtualXORROT25 => (materialize_entry_25, lookup_output_matches_trace_25),
        VirtualXORROT28 => (materialize_entry_28, lookup_output_matches_trace_28),
        VirtualXORROT32 => (materialize_entry_32, lookup_output_matches_trace_32),
        VirtualXORROT36 => (materialize_entry_36, lookup_output_matches_trace_36),
        VirtualXORROT37 => (materialize_entry_37, lookup_output_matches_trace_37),
        VirtualXORROT39 => (materialize_entry_39, lookup_output_matches_trace_39),
        VirtualXORROT43 => (materialize_entry_43, lookup_output_matches_trace_43),
        VirtualXORROT44 => (materialize_entry_44, lookup_output_matches_trace_44),
        VirtualXORROT46 => (materialize_entry_46, lookup_output_matches_trace_46),
        VirtualXORROT49 => (materialize_entry_49, lookup_output_matches_trace_49),
        VirtualXORROT50 => (materialize_entry_50, lookup_output_matches_trace_50),
        VirtualXORROT54 => (materialize_entry_54, lookup_output_matches_trace_54),
        VirtualXORROT56 => (materialize_entry_56, lookup_output_matches_trace_56),
        VirtualXORROT58 => (materialize_entry_58, lookup_output_matches_trace_58),
        VirtualXORROT61 => (materialize_entry_61, lookup_output_matches_trace_61),
        VirtualXORROT62 => (materialize_entry_62, lookup_output_matches_trace_62),
        VirtualXORROT63 => (materialize_entry_63, lookup_output_matches_trace_63),
    );
}
