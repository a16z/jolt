use crate::subprotocols::sparse_dense_shout::TestInstructionTrait;
use jolt_core::jolt::instruction::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use jolt_core::jolt::lookup_table::{LookupTables, range_check::RangeCheckTable};
use rand::RngCore;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ADVICEInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ADVICEInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> InstructionFlags for ADVICEInstruction<WORD_SIZE> {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Advice as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        // flags[CircuitFlags::InlineSequenceInstruction as usize] =
        //     self.virtual_sequence_remaining.is_some();
        // flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
        //     self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for ADVICEInstruction<WORD_SIZE> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (0, self.0 as u8 as u64),
            32 => (0, self.0 as u32 as u64),
            64 => (0, self.0),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8).into(),
            32 => (self.0 as u32).into(),
            64 => self.0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
