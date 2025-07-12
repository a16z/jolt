//! # Tracer module for ONNX models
//!
//! This module defines the core data structures and logic for caputuring VM state during each execution cycle,
//! specifically tailored for ONNX model with quantized execution in the zkML-Jolt framework.
//!
//! ## Purpose
//! ## Overview of Components
//! ## Usage
//! ## Context

use crate::trace_types::ONNXCycle;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Keeps a record of what the VM did over the course of execution.
/// Constructs the execution trace for an ONNX model.
/// Used in [`super::model::Model::forward`]: pushes a new `ONNXCycle` to the execution trace
/// at each step of the VM execution cycle, documenting the instruction and state changes.
pub struct Tracer {
    pub execution_trace: Vec<ONNXCycle>,
}

impl Tracer {
    pub fn capture_pre_state() {
        todo!()
    }

    pub fn capture_post_state() {
        todo!()
    }
}
