use jolt_core::jolt::{
    instruction::InstructionLookup,
    lookup_table::{LookupTables, range_check::RangeCheckTable},
};
use serde::{Deserialize, Serialize};

use crate::jolt::execution_trace::ONNXLookupQuery;

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReduceSumInstruction<const WORD_SIZE: usize>(pub Vec<u64>);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ReduceSumInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> ONNXLookupQuery<WORD_SIZE> for ReduceSumInstruction<WORD_SIZE> {
    fn to_instruction_inputs(&self) -> (Vec<u64>, Vec<i64>) {
        todo!()
    }

    fn to_lookup_operands(&self) -> (Vec<u64>, Vec<u64>) {
        todo!()
    }

    fn to_lookup_index(&self) -> Vec<u64> {
        todo!()
    }

    fn to_lookup_output(&self) -> Vec<u64> {
        todo!()
    }
}
