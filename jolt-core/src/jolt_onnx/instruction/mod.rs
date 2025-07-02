//! This module provides the custom jolt instructions for the ONNX runtime.

use crate::jolt_onnx::tracer::tensor::QuantizedTensor;

pub mod max;
pub mod relu;
pub mod sigmoid;