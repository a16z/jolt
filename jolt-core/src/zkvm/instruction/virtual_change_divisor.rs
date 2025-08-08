use tracer::instruction::{virtual_change_divisor::VirtualChangeDivisor, RISCVCycle};

use crate::zkvm::lookup_table::virtual_change_divisor::VirtualChangeDivisorTable;
use crate::zkvm::lookup_table::LookupTables;

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualChangeDivisor {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualChangeDivisorTable.into())
    }
}

impl InstructionFlags for VirtualChangeDivisor {
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

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualChangeDivisor> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as i128,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.register_state.rs2 as u32 as i128,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2 as i128),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                let remainder = remainder as i8;
                let divisor = divisor as i8;
                if remainder == i8::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u8 as u64
                }
            }
            32 => {
                let remainder = remainder as i32;
                let divisor = divisor as i32;
                if remainder == i32::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u32 as u64
                }
            }
            64 => {
                let remainder = remainder as i64;
                if remainder == i64::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u64
                }
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualChangeDivisor>();
    }
}
