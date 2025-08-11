use crate::jolt::execution_trace::ONNXLookupQuery;
use jolt_core::jolt::{
    instruction::InstructionLookup,
    lookup_table::{LookupTables, range_check::RangeCheckTable},
};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReduceSumInstruction<const WORD_SIZE: usize>(pub Vec<u64>);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ReduceSumInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> ONNXLookupQuery<WORD_SIZE> for ReduceSumInstruction<WORD_SIZE> {
    fn to_instruction_inputs(&self) -> (Vec<u64>, Vec<i64>) {
        match WORD_SIZE {
            32 => (
                self.0.iter().map(|&x| x as u32 as u64).collect(),
                vec![0; MAX_TENSOR_SIZE],
            ),
            64 => (self.0.clone(), vec![0; MAX_TENSOR_SIZE]),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_operands(&self) -> (Vec<u64>, Vec<u64>) {
        let (x, _) = ONNXLookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        let mut right_lookup_operand = vec![0; MAX_TENSOR_SIZE];
        right_lookup_operand[0] = x.iter().sum::<u64>();
        (vec![0; MAX_TENSOR_SIZE], right_lookup_operand)
    }

    fn to_lookup_index(&self) -> Vec<u64> {
        ONNXLookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> Vec<u64> {
        let (x, _) = ONNXLookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            32 => {
                let mut output = vec![0; MAX_TENSOR_SIZE];
                output[0] = x
                    .iter()
                    .fold(0u32, |acc, &val| acc.overflowing_add(val as u32).0)
                    as u64;
                output
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
