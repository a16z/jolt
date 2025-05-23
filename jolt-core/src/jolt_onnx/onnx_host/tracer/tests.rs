use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use tract_onnx::prelude::*;

use crate::jolt_onnx::onnx_host::tracer::parse;
use crate::jolt_onnx::onnx_host::ONNXProgram;

#[test]
fn test_perceptron() {
    let rng = StdRng::from_seed([0; 32]);
    let size = 10;
    let path = "onnx/perceptron.onnx";
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    let data = random_floatvec(rng, size);
    let input = Tensor::from_shape(&[1, size], &data).unwrap();
    let output = model.run(tvec!(input.into_tvalue())).unwrap();
    let output = output[0].to_array_view::<f32>().unwrap();
    let output = output.as_slice().unwrap();
    println!("Output: {output:?}",);

    let jolt_model = parse(&PathBuf::from(path)).execute(&data);
}

#[test]
fn test_perceptron_2() {
    let rng = StdRng::from_seed([0; 32]);
    let size = 4;
    let path = "onnx/perceptron_2.onnx";
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    let data = random_floatvec(rng, size);
    let input = Tensor::from_shape(&[1, size], &data).unwrap();
    let output = model.run(tvec!(input.into_tvalue())).unwrap();
    let output = output[0].to_array_view::<f32>().unwrap();
    let output = output.as_slice().unwrap();
    println!("Output: {output:?}",);

    let jolt_model = parse(&PathBuf::from(path)).execute(&data);
}

fn random_floatvec(mut rng: impl RngCore, size: usize) -> Vec<f32> {
    (0..size).map(|_| rng.gen::<f32>()).collect()
}
