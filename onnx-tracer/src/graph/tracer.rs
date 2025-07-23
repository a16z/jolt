//! # Tracer module for ONNX models
//!
//! This module defines the core data structures and logic for caputuring VM state during each execution cycle,
//! specifically tailored for ONNX model with quantized execution in the zkML-Jolt framework.
//!
//! ## Purpose
//! ## Overview of Components
//! ## Usage
//! ## Context

use std::cell::RefCell;

use crate::trace_types::{ONNXCycle, ONNXInstr};
use serde::{Deserialize, Serialize};

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
    pub fn capture_pre_state(&self, instr: Vec<ONNXInstr>) {
        self.execution_trace
            .try_borrow_mut()
            .unwrap()
            .extend(instr.into_iter().map(|instr| ONNXCycle { instr }));
    }

    pub fn capture_post_state() {
        todo!()
    }
}
