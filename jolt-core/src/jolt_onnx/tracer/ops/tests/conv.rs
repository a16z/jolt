use std::{collections::HashMap, str::FromStr};

use crate::jolt_onnx::{
    common::onnx_trace::{ONNXInstruction, Operator},
    tracer::{
        model::{ONNXInitializerMap, QuantizedONNXModel},
        tensor::QuantizedTensor,
    },
};

#[test]
fn test_conv_simple() {
    let mut instrs: Vec<ONNXInstruction> = Vec::new();
    let mut conv_instr = ONNXInstruction::new(Operator::from_str("Conv").unwrap());
    conv_instr.inputs = vec![
        "input".to_string(),
        "weight".to_string(),
        "bias".to_string(),
    ];
    conv_instr.outputs = vec!["output".to_string()];
    let weight = QuantizedTensor::from_float_data(
        vec![1, 1, 3, 3],
        vec![
            1.0, 0.0, -1.0, // Kernel row 1
            1.0, 0.0, -1.0, // Kernel row 2
            1.0, 0.0, -1.0, // Kernel row 3
        ],
    );
    let bias = QuantizedTensor::from_float_data(vec![1], vec![0.0]); // No bias
    let intializer_map =
        HashMap::from([("weight".to_string(), weight), ("bias".to_string(), bias)]);

    let mut attributes = HashMap::<String, Vec<u64>>::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![1, 1]);
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]);
    attributes.insert("dilations".to_string(), vec![1, 1]);
    conv_instr.attributes = Some(attributes);

    instrs.push(conv_instr);

    let mut model = QuantizedONNXModel {
        instrs,
        input_shape: vec![1, 1, 4, 4],
        initializer_map: ONNXInitializerMap(intializer_map),
        ..Default::default()
    };

    // Input tensor
    let input_tensor = vec![
        1.0, 2.0, 3.0, 4.0, // Row 1
        5.0, 6.0, 7.0, 8.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
        13.0, 14.0, 15.0, 16.0, // Row 4
    ];

    let _ = model.execute_quantized(&input_tensor);
}
