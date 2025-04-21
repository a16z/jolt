use tracer::instruction::{
    virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder, RISCVCycle,
};

use crate::jolt::lookup_table::{valid_signed_remainder::ValidSignedRemainderTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertValidSignedRemainder {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidSignedRemainderTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        match WORD_SIZE {
            32 => {
                let (remainder, divisor) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
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
                let (remainder, divisor) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
                let remainder = remainder as i64;
                let divisor = divisor as i64;
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
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        }
    }
}
