//! This module provides an implementation to get the lookup trace from the execution trace.

use crate::jolt_onnx::{
    common::onnx_trace::{ONNXTraceRow, Operator},
    instruction::relu::ReLUInstruction,
    precompiles::{matmult::MatMultPrecompile, PrecompileOperators},
    vm::{onnx_vm::ONNXInstructionSet, JoltONNXTraceStep},
};
use tracer::ELFInstruction;

impl ONNXTraceRow {
    /// Convert [`ONNXTraceRow`] to a vector of [`JoltONNXTraceStep<ONNXInstructionSet>`]
    pub fn to_trace_step(&self) -> Vec<JoltONNXTraceStep<ONNXInstructionSet>> {
        // Check if jolt vm will perfrom an instruction lookup
        if let Some(lookups) = self.to_lookup() {
            return lookups
                .into_iter()
                .map(|lookup| {
                    let mut step = JoltONNXTraceStep::no_op(); // TODO: We will change this from no-op when bytecodeproof and mcc are fully-fleshed out
                    step.instruction_lookup = Some(lookup);
                    step
                })
                .collect::<Vec<_>>();
        }

        // Check if jolt vm will perform a precompile operation
        if let Some(precompile) = self.to_precompile() {
            let mut step = JoltONNXTraceStep::no_op();
            step.precompile = Some(precompile);
            return vec![step];
        }

        // If no lookup or precompile is needed, return a no-op step
        vec![JoltONNXTraceStep::no_op()]
    }

    /// Convert [`ONNXTraceRow`] to [`ONNXInstructionSet`]
    pub fn to_lookup(&self) -> Option<Vec<ONNXInstructionSet>> {
        match self.instruction.opcode {
            Operator::Relu => self
                .layer_state
                .input_vals
                .first()
                .map(|tensor| {
                    let lookups = tensor
                        .data
                        .iter()
                        .map(|value| ReLUInstruction(*value as u64).into())
                        .collect::<Vec<ONNXInstructionSet>>();
                    lookups
                }),
            _ => None,
        }
    }

    /// Convert [`ONNXTraceRow`] to [`PrecompileOperators`]
    pub fn to_precompile(&self) -> Option<PrecompileOperators> {
        match self.instruction.opcode {
            Operator::MatMul => {
                // --- # Note ---
                // We pad the tensor dimensions to the next power of two
                let inputs = &self
                    .layer_state
                    .input_vals;
                let a = inputs[0].pad();
                let b = inputs[1].pad();
                Some(PrecompileOperators::MatMult(MatMultPrecompile::new(a, b)))
            }
            _ => None,
        }
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
