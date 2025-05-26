//! This module provides a matmult sum-check precompile.

#[cfg(test)]
mod tests {
    use crate::jolt_onnx;
    use crate::jolt_onnx::onnx_host::ONNXProgram;
    use crate::jolt_onnx::utils::random_floatvec;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_matmult() {
        let mut rng = StdRng::from_seed([1; 32]);
        let input = random_floatvec(&mut rng, 4);
        let program = ONNXProgram::new("onnx/perceptron_2.onnx", &input);
        let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
    }
}
