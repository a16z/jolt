//! # Tracer module for ONNX models
//!
//! This module defines the core data structures and logic for caputuring VM state during each execution cycle,
//! specifically tailored for ONNX model with quantized execution in the zkML-Jolt framework.
//!
//!
//! The `Tracer` struct is responsible for recording the execution trace of an ONNX model as it runs.
//! It captures the state of the virtual machine at each step, including the instructions executed and the
//! memory state changes.
//!
//! ## Overview of Components
//! - `ONNXCycle`: Represents a single step in the execution trace, containing the instruction executed and the memory state at that point.
//! - `MemoryState`: Holds the values of input tensors and the output tensor after an instruction is executed.
//! - `ONNXInstr`: Represents a single ONNX instruction, including its address, opcode, and input/output tensor indices.
//! - `Tracer`: The main struct that manages the execution trace, allowing for capturing pre- and post-execution states of instructions.
//!
//! The `Tracer` is used within the ONNX model execution context to log the operations performed by the virtual machine.
//! This module is part of the `onnx-tracer` crate, which provides functionality for tracing ONNX models in the zkML-Jolt framework.

use crate::{
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr},
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

#[derive(Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Keeps a record of what the VM did over the course of execution.
/// Constructs the execution trace for an ONNX model.
/// Used in [`super::model::Model::forward`]: pushes a new `ONNXCycle` to the execution trace
/// at each step of the VM execution cycle, documenting the instruction and state changes.
pub struct Tracer {
    /// We use RefCell to allow interior mutability of the execution trace,
    /// enabling us to mutate the trace (push new ONNXCycle entries) even when
    /// Tracer is shared immutably. This is necessary because the Tracer is often
    /// passed around as an immutable reference, but we still need to record state.
    pub execution_trace: RefCell<Vec<ONNXCycle>>,
}

impl Tracer {
    /// (Pre-execution) Records the operand reads used during execution of an ONNX instruction.
    ///
    /// This method should be called to log the input tensors (operands) that are read
    /// and used for the current instruction in the VM cycle. It appends a new entry
    /// to the execution trace, capturing the instruction and its associated memory state.
    ///
    /// # Panics
    /// Panics if `execution_trace` is already mutably borrowed elsewhere,
    /// which would indicate a bug in concurrent trace recording.
    pub fn capture_pre_state(&self, instr: ONNXInstr, inputs: Vec<Tensor<i128>>) {
        let mut cycle = ONNXCycle {
            instr: instr.clone(),
            memory_state: MemoryState::default(),
            advice_value: None,
        };
        if instr.ts1.is_some() {
            cycle.memory_state.ts1_val = Some(inputs[0].clone())
        };
        if instr.ts2.is_some() {
            cycle.memory_state.ts2_val = Some(inputs[1].clone())
        };
        self.execution_trace.borrow_mut().push(cycle);
    }

    /// (Post-execution) Records the output tensor after executing an ONNX instruction.
    /// This method should be called after the instruction has been executed,
    /// to log the result tensor that was produced.
    /// It updates the last entry in the execution trace
    /// with the output tensor, which is the result of the instruction.
    ///
    /// # Panics
    /// Panics if `execution_trace` is already mutably borrowed elsewhere,
    /// which would indicate a bug in concurrent trace recording.
    ///
    /// # Arguments
    /// - `output`: The output tensor produced by the instruction execution.
    ///
    /// This tensor is captured and stored in the last `ONNXCycle` entry
    /// in the execution trace.
    ///
    /// # Note
    /// This method assumes that the last `ONNXCycle` in the trace corresponds
    /// to the instruction that just executed. It updates the memory state
    /// of that cycle with the output tensor.
    /// It is crucial that this method is called immediately after the instruction execution
    /// to ensure the trace accurately reflects the state of the VM at that point.
    /// If the instruction has a destination tensor (`td`), this method will
    /// update the `td_post_val` field in the memory state of the last cycle.
    /// If the instruction does not have a destination tensor, this method will not update the trace.
    /// This is to ensure that the trace only records outputs for instructions
    /// that produce a result that is stored in the computation graph.
    pub fn capture_post_state(&self, output: Tensor<i128>) {
        let mut execution_trace = self.execution_trace.borrow_mut();
        let row = execution_trace.last_mut().unwrap();
        if row.instr.td.is_some() {
            row.memory_state.td_post_val = Some(output.clone())
        };
    }

    /// Clears the execution trace, resetting it to an empty state.
    /// This method is useful for reinitializing the tracer before a new execution run.
    pub fn clear(&self) {
        self.execution_trace.borrow_mut().clear();
    }
}
