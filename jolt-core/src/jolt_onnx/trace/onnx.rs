//! This module provides an implementation to get the lookup trace from the execution trace.

use crate::jolt_onnx::{
    common::onnx_trace::{ONNXTraceRow, Operator},
    instruction::relu::ReLUInstruction,
    vm::onnx_vm::ONNXInstructionSet,
};

use tracer::ELFInstruction;

/// Convert [`ONNXTraceRow`] to [`ONNXInstructionSet`]
pub fn onnxrow_to_lookup(row: &ONNXTraceRow) -> Option<Vec<ONNXInstructionSet>> {
    match row.instruction.opcode {
        // TODO: clean this up to make it extensible for other operators
        Operator::Relu => row
            .layer_state
            .input_vals
            .as_ref()
            .and_then(|inputs| inputs.first())
            .map(|tensor| {
                let lookups = tensor
                    .data
                    .iter()
                    .map(|value| ReLUInstruction(*value as u8 as u32 as u64).into())
                    .collect::<Vec<ONNXInstructionSet>>();
                lookups
            }),
        _ => None,
    }
}

/// Trivial [`TryFrom`] trait implementation for [`ELFInstruction`] to [`ONNX`] to make [`JoltInstructionSet`] trait happy
impl TryFrom<&ELFInstruction> for ONNXInstructionSet {
    type Error = &'static str;

    #[rustfmt::skip] // keep matches pretty
    fn try_from(_: &ELFInstruction) -> Result<Self, Self::Error> {
        Err("No corresponding ONNX instruction")
    }
}
