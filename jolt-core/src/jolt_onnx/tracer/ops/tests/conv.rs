use std::str::FromStr;

use crate::jolt_onnx::{
    common::onnx_trace::{ONNXInstruction, Operator},
    tracer::{model::QuantizedONNXModel, tensor::QuantizedTensor},
};

#[test]
fn test_conv_simple() {
    let mut instrs: Vec<ONNXInstruction> = Vec::new();
    instrs.push(ONNXInstruction::new(Operator::from_str("Conv").unwrap()));
    let weight = QuantizedTensor::new(
        vec![1, 1, 3, 3],
        vec![
            1.0, 0.0, -1.0, // Kernel row 1
            1.0, 0.0, -1.0, // Kernel row 2
            1.0, 0.0, -1.0, // Kernel row 3
        ],
    );
    let bias = QuantizedTensor::new(vec![1], vec![0.0]); // No bias
}
