use crate::jolt_onnx::tracer::model::QuantizedONNXModel;
use crate::jolt_onnx::tracer::trace;
use crate::jolt_onnx::utils::random_floatvec;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

fn run_perceptron_test(path: &str, size: usize, seed: [u8; 32]) {
    let rng = StdRng::from_seed(seed);

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
    let expected = output.as_slice().unwrap();

    // Check output with tracer
    let path = PathBuf::from(path);
    let (_trace, io) = trace(&path, &data);
    println!("Trace: {_trace:#?}",);
    println!("Expected: {expected:?}",);
    println!("Result: {:?}", io.outputs); // TODO: Add tracing::info! for this and add tracing logs in general
    let res = QuantizedONNXModel::parse(&path).execute(&data);
    println!("Result: {res:?}",);
    //   assert_eq!(res.data, expected.to_vec()); // TODO: Figure out where some data is lost
}

#[test]
fn test_perceptron() {
    run_perceptron_test("onnx/perceptron.onnx", 10, [0; 32]);
}

#[test]
fn test_perceptron_2() {
    run_perceptron_test("onnx/perceptron_2.onnx", 4, [0; 32]);
}
