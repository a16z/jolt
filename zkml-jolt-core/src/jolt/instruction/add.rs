use jolt_core::jolt::instruction::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use jolt_core::jolt::lookup_table::{LookupTables, range_check::RangeCheckTable};
use onnx_tracer::trace_types::ONNXOpcode;

use crate::subprotocols::sparse_dense_shout::ElementWiseOpCycle;

pub struct ADD(pub u64, pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ADD {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl InstructionFlags for ADD {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        // flags[CircuitFlags::InlineSequenceInstruction as usize] =
        //     self.virtual_sequence_remaining.is_some();
        // flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
        //     self.virtual_sequence_remaining.unwrap_or(0) != 0; // TODO
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for ADD {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x + y as u64)
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
            8 => (x as u8).overflowing_add(y as u8).0.into(),
            32 => (x as u32).overflowing_add(y as u32).0.into(),
            64 => x.overflowing_add(y as u64).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
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
//         materialize_entry_test::<Fr, ADD>();
//     }
// }
