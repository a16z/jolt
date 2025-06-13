use tracer::instruction::{
    virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder, RISCVCycle,
};

use crate::jolt::lookup_table::{valid_signed_remainder::ValidSignedRemainderTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertValidSignedRemainder {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidSignedRemainderTable.into())
    }
}

impl InstructionFlags for VirtualAssertValidSignedRemainder {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE>
    for RISCVCycle<VirtualAssertValidSignedRemainder>
{
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (self.register_state.rs1, self.register_state.rs2 as i64)
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

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertValidSignedRemainder>();
    }
}
