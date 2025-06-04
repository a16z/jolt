use crate::jolt_onnx::tracer::model::QuantizedONNXModel;
use crate::jolt_onnx::tracer::trace;
use crate::jolt_onnx::utils::random_floatvec;
use ark_std::test_rng;
use std::path::PathBuf;
use tract_onnx::prelude::*;

// TODO: Improve these tests, for now they just check if the model can be parsed and executed

fn run_perceptron_test(path: &str, size: usize) {
    let rng = test_rng();

    // Build the ONNX model using tract
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    // Get some random input data
    let data = random_floatvec(rng, size);
    let input = Tensor::from_shape(&[1, size], &data).unwrap();

    // Get expected output from the ONNX model
    let output = model.run(tvec!(input.into_tvalue())).unwrap();
    let output = output[0].to_array_view::<f32>().unwrap();
    let _expected = output.as_slice().unwrap();

    // Check output with tracer
    let path = PathBuf::from(path);
    let (_trace, _io) = trace(&path, &data);
    let mut model = QuantizedONNXModel::parse(&path);
    println!("model: {:#?}", model.instrs);
    let res = model.execute_quantized(&data);
    println!("res: {:#?}", res.dequantized_data());
}

#[test]
fn test_1l_conv() {
    let path = PathBuf::from("onnx/conv/1l_conv.onnx");
    let mut model = QuantizedONNXModel::parse(&path);
}

#[test]
fn test_perceptron() {
    run_perceptron_test("onnx/mlp/perceptron.onnx", 10);
}

#[test]
fn test_perceptron_2() {
    run_perceptron_test("onnx/mlp/perceptron_2.onnx", 4);
}
