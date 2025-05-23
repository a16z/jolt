//! This module provides a proving system for the ONNX runtime.
//! It uses Jolt's proving system to create a proof of the ONNX model.

// TODO: Bring back this lint
// #![warn(missing_docs)]
#![allow(clippy::type_complexity)]

pub mod common;
pub mod instruction;
pub mod onnx_host;
pub mod subtable;
pub mod trace; // TODO: Remove this module
pub mod utils;
pub mod vm;
