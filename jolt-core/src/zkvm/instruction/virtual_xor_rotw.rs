use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{
    virtual_xor_rotw::{VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8},
    RISCVCycle,
};

use crate::zkvm::lookup_table::{virtual_xor_rotw::VirtualXORROTWTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

// Macro to implement traits for each specific rotation value
macro_rules! impl_virtual_xor_rotw {
    ($type:ty, $rotation:expr) => {
        impl<const XLEN: usize> InstructionLookup<XLEN> for $type {
            fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
                Some(VirtualXORROTWTable::<XLEN, $rotation>.into())
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
                flags[InstructionFlags::IsRdNotZero] = self.operands.rd != 0;
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
                        (((xor_result & 0x0F) >> ($rotation % 4))
                            | (((xor_result & 0x0F) << (4 - ($rotation % 4))) & 0x0F))
                            as u64
                    }
                    64 => {
                        let x_32 = x as u32;
                        let y_32 = y as u32;
                        let xor_result = x_32 ^ y_32;
                        xor_result.rotate_right($rotation) as u64
                    }
                    _ => panic!("{XLEN}-bit word size is unsupported"),
                }
            }
        }
    };
}

impl_virtual_xor_rotw!(VirtualXORROTW16, 16);
impl_virtual_xor_rotw!(VirtualXORROTW12, 12);
impl_virtual_xor_rotw!(VirtualXORROTW8, 8);
impl_virtual_xor_rotw!(VirtualXORROTW7, 7);

#[cfg(test)]
mod test {
    use super::*;
    use crate::zkvm::instruction::test::materialize_entry_test;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry_16() {
        materialize_entry_test::<Fr, VirtualXORROTW16>();
    }

    #[test]
    fn materialize_entry_12() {
        materialize_entry_test::<Fr, VirtualXORROTW12>();
    }

    #[test]
    fn materialize_entry_8() {
        materialize_entry_test::<Fr, VirtualXORROTW8>();
    }

    #[test]
    fn materialize_entry_7() {
        materialize_entry_test::<Fr, VirtualXORROTW7>();
    }
}
