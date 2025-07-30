use jolt_core::jolt::instruction::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use jolt_core::jolt::lookup_table::LookupTables;
use jolt_core::jolt::lookup_table::valid_signed_remainder::ValidSignedRemainderTable;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (remainder, divisor)
pub struct AssertValidSignedRemainderInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE>
    for AssertValidSignedRemainderInstruction<WORD_SIZE>
{
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidSignedRemainderTable.into())
    }
}

impl<const WORD_SIZE: usize> InstructionFlags for AssertValidSignedRemainderInstruction<WORD_SIZE> {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        // flags[CircuitFlags::InlineSequenceInstruction as usize] =
        //     self.virtual_sequence_remaining.is_some();
        // flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
        //     self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE>
    for AssertValidSignedRemainderInstruction<WORD_SIZE>
{
    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8 as u64, self.1 as u8 as i64),
            32 => (self.0 as u32 as u64, self.1 as u32 as i64),
            64 => (self.0, self.1 as i64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            32 => {
                let (remainder, divisor) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
                let remainder = remainder as u32 as i32;
                let divisor = divisor as u32 as i32;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            64 => {
                let (remainder, divisor) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
                let remainder = remainder as i64;
                let is_remainder_zero = remainder == 0;
                let is_divisor_zero = divisor == 0;

                if is_remainder_zero || is_divisor_zero {
                    1
                } else {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    (remainder.unsigned_abs() < divisor.unsigned_abs()
                        && remainder_sign == divisor_sign)
                        .into()
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        }
    }
}

// #[cfg(test)]
// mod test {
//     use crate::jolt::instruction::test::materialize_entry_test;

//     use super::*;
//     use ark_bn254::Fr;

//     #[test]
//     fn materialize_entry() {
//         materialize_entry_test::<Fr, VirtualAssertValidSignedRemainder>();
//     }
// }
