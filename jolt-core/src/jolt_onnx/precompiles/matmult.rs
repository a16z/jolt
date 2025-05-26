//! This module provides a matmult sum-check precompile.

use crate::{field::JoltField, utils::transcript::Transcript};

use super::sumcheck_engine::BatchableSumcheckInstance;

/// Sum-check precompile for matrix multiplication
pub struct MatMultPrecompile;

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for MatMultPrecompile
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        todo!()
    }

    fn num_rounds(&self) -> usize {
        todo!()
    }

    fn input_claim(&self) -> F {
        todo!()
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        todo!()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        todo!()
    }

    fn cache_openings(&mut self) {
        todo!()
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        todo!()
    }
}

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
