//! This module provides a proving system for the ONNX runtime.
//! It uses Jolt's proving system to create a proof of the ONNX model.

// TODO: Bring back this lint
// #![warn(missing_docs)]

pub mod instruction;
pub mod subtable;
pub mod trace;
pub mod vm;
