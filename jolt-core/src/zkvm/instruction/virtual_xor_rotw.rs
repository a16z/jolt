use tracer::instruction::{
    virtual_xor_rotw::{VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8},
    RISCVCycle,
};

use crate::zkvm::lookup_table::{virtual_xor_rotw::VirtualXORROTWTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

// Macro to implement traits for each specific rotation value
macro_rules! impl_virtual_xor_rotw {
    ($type:ty, $rotation:expr) => {
        impl<const XLEN: usize> InstructionLookup<XLEN> for $type {
            fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
                debug_assert_eq!(XLEN, 64);
                Some(VirtualXORROTWTable::<XLEN, $rotation>.into())
            }
        }

        impl InstructionFlags for $type {
            fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
                let mut flags = [false; NUM_CIRCUIT_FLAGS];
                flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
                flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
                flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
                flags[CircuitFlags::InlineSequenceInstruction as usize] =
                    self.inline_sequence_remaining.is_some();
                flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
                    self.inline_sequence_remaining.unwrap_or(0) != 0;
                flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
                flags
            }
        }

        impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<$type> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                match XLEN {
                    64 => (self.register_state.rs1, self.register_state.rs2 as i128),
                    _ => panic!("{XLEN}-bit word size is unsupported"),
                }
            }

            fn to_lookup_output(&self) -> u64 {
                let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                match XLEN {
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
