//! This module provides a proving system for the ONNX runtime.
//! It uses Jolt's proving system to create a proof of the ONNX model.

#![warn(missing_docs)]
#![allow(clippy::type_complexity)]

pub mod common;
pub mod instruction;
mod memory_checking;
pub mod onnx_host;
pub mod precompiles;
pub mod subtable;
pub mod trace;
pub mod tracer;
pub mod utils;
pub mod vm;
