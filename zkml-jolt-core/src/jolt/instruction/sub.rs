use crate::subprotocols::sparse_dense_shout::TestInstructionTrait;
use jolt_core::jolt::instruction::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use jolt_core::jolt::lookup_table::{LookupTables, range_check::RangeCheckTable};
use rand::RngCore;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SUB<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for SUB<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

// TODO: Implement circuit flags for ONNX instructions
impl<const WORD_SIZE: usize> InstructionFlags for SUB<WORD_SIZE> {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::SubtractOperands as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for SUB<WORD_SIZE> {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        let x = x as u128;
        let y = (1u128 << WORD_SIZE) - y as u128;
        (0, (x + y) as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

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
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8).overflowing_sub(y as u8).0.into(),
            32 => (x as u32).overflowing_sub(y as u32).0.into(),
            64 => x.overflowing_sub(y as u64).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize> TestInstructionTrait for SUB<WORD_SIZE> {
    fn random(rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64, rng.next_u32() as u64),
            64 => Self(rng.next_u64(), rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
