//! This module provides a proving system for the ONNX runtime.
//! It uses Jolt's proving system to create a proof of the ONNX model.

#![warn(missing_docs)]

#[cfg(test)]
mod tests;
mod trace;
pub mod vm;
