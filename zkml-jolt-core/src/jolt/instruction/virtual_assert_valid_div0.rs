use jolt_core::jolt::instruction::{InstructionLookup, LookupQuery};
use jolt_core::jolt::lookup_table::LookupTables;
use jolt_core::jolt::lookup_table::valid_div0::ValidDiv0Table;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct AssertValidDiv0Instruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE>
    for AssertValidDiv0Instruction<WORD_SIZE>
{
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidDiv0Table.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for AssertValidDiv0Instruction<WORD_SIZE> {
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
        let (divisor, quotient) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient as u64 == u32::MAX as u64).into(),
                64 => (quotient as u64 == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
            }
        } else {
            1
        }
    }
}
