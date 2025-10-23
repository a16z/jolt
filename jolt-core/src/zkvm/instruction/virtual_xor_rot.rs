use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
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
                flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
                flags[CircuitFlags::VirtualInstruction as usize] =
                    self.virtual_sequence_remaining.is_some();
                flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
                    self.virtual_sequence_remaining.unwrap_or(0) != 0;
                flags[CircuitFlags::IsFirstInSequence as usize] = self.is_first_in_sequence;
                flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
                flags
            }

            fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
                let mut flags = [false; NUM_INSTRUCTION_FLAGS];
                flags[InstructionFlags::LeftOperandIsRs1Value as usize] = true;
                flags[InstructionFlags::RightOperandIsRs2Value as usize] = true;
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::zkvm::instruction::test::{
        lookup_output_matches_trace_test, materialize_entry_test,
    };
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry_32() {
        materialize_entry_test::<Fr, VirtualXORROT32>();
    }

    #[test]
    fn materialize_entry_24() {
        materialize_entry_test::<Fr, VirtualXORROT24>();
    }

    #[test]
    fn materialize_entry_16() {
        materialize_entry_test::<Fr, VirtualXORROT16>();
    }

    #[test]
    fn materialize_entry_63() {
        materialize_entry_test::<Fr, VirtualXORROT63>();
    }

    #[test]
    fn lookup_output_matches_trace_32() {
        lookup_output_matches_trace_test::<VirtualXORROT32>();
    }

    #[test]
    fn lookup_output_matches_trace_24() {
        lookup_output_matches_trace_test::<VirtualXORROT24>();
    }

    #[test]
    fn lookup_output_matches_trace_16() {
        lookup_output_matches_trace_test::<VirtualXORROT16>();
    }

    #[test]
    fn lookup_output_matches_trace_63() {
        lookup_output_matches_trace_test::<VirtualXORROT63>();
    }
}
