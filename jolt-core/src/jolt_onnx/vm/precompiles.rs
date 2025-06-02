//! Precompile proofs for ONNX runtime.
//!
//! This module provides a custom SNARK for precompiles in the [`ONNXJoltVM`].
//! These precompile proofs are sum-check based.

use super::JoltONNXTraceStep;
use crate::{
    field::JoltField,
    jolt::instruction::JoltInstructionSet,
    jolt_onnx::{
        common::onnx_trace::Operator,
        precompiles::{
            matmult::{MatMultClaims, MatMultProverState, MatMultSumcheck, MatMultVerifierState},
            sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
            PrecompileOperators,
        },
        tracer::model::QuantizedONNXModel,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;

/// The dimensions used in the matrix multiplication precompile protocol.
///
/// mat_mult_precompile_dims = (m, n, k) where
/// - `m` is the number of rows in the resulting matrix,
/// - `n` is the number of columns in the resulting matrix,
/// - `k` is the number of columns in the lhs matrix
///
/// We use m, and n to get the required length of the challenge vectors in the sum-check matrix-multiplication precompile.
/// And k is used to determine the number of sum-check rounds
///
/// # Note: We pad the dimensions to the next power of two.
pub type MatMultPrecompileDims = (usize, usize, usize);

/// Preprocessing of the models matrices for the precompile proof.
/// Store the dimensions of the matrix multiplication precompile.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecompilePreprocessing {
    /// The dimensions used in the matrix multiplication precompile's.
    pub mat_mult_precompile_dims: Vec<MatMultPrecompileDims>,
}

impl PrecompilePreprocessing {
    /// Preprocess the ONNX model to extract the dimensions of the matrix multiplication precompile.
    #[tracing::instrument(skip_all, name = "PrecompilePreprocessing::preprocess")]
    pub fn preprocess(model: &QuantizedONNXModel) -> Self {
        let io_shapes = model.layers_io_shapes();

        // For each matmult instruction store the [`MatMultPrecompileDims`]
        // We pad the dimensions to the next power of two.
        let mat_mult_precompile_dims = model
            .instrs
            .iter()
            .zip_eq(io_shapes.iter())
            .filter_map(|(instr, (input_shape, output_shape))| match instr.opcode {
                Operator::MatMul => {
                    let m = output_shape[0].next_power_of_two();
                    let n = output_shape[1].next_power_of_two();
                    let k = input_shape[1].next_power_of_two();
                    Some((m, n, k))
                }
                _ => None,
            })
            .collect_vec();
        Self {
            mat_mult_precompile_dims,
        }
    }
}

/// A special-purpose SNARK designed for specific functionality, such as ONNX operators that are more efficient to prove using a sum-check precompile than an [`InstructionLookupProof`].
/// This is a sum-check-based precompile proof tailored for ONNX runtime.
/// It is used to prove the correctness of certain ONNX operators via a custom sum-check precompile instead of a lookup-based approach.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecompileProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    init_claims: Vec<F>,
    final_claims: Vec<MatMultClaims<F>>,
}

impl<F, ProofTranscript> PrecompileProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    /// Given the execution trace, construct the polynomials used in the batched sum-check proof.
    /// The witness polynomials are abstracted as `MatMultSumcheck` instances, which hold a MatMultProverState which contains the witness polynomials `a` & `b` for the matrix multiplication precompile.
    ///
    /// # Note
    /// - We require the `transcript` to generate the challenges for the matrix multiplication precompile.
    pub fn generate_witness<InstructionSet>(
        ops: &[JoltONNXTraceStep<InstructionSet>],
        transcript: &mut ProofTranscript,
    ) -> Vec<MatMultSumcheck<F>>
    where
        InstructionSet: JoltInstructionSet,
    {
        // Filter the operations to only include those that are proven with precompiles.
        // For each precompile operator, initialize the prover state and create a new `MatMultSumcheck`.
        ops.iter()
            .filter_map(|op| match &op.precompile {
                Some(PrecompileOperators::MatMult(mat_mult)) => {
                    // Initialize the prover state for the matrix multiplication precompile.
                    // `MatMultProverState::initialize` constructs the witness polynomials `a` & `b` for the matrix multiplication precompile.
                    // It takes the `transcript` as an argument to generate the challenges, rx & ry to compute the evaluation for Sum_k A(rx, k) & B(ry, k)
                    let prover_state: MatMultProverState<F> =
                        MatMultProverState::initialize(mat_mult, transcript);

                    // Create a new `MatMultSumcheck` instance with the prover state.
                    Some(MatMultSumcheck::new(Some(prover_state), None, None))
                }
                _ => None,
            })
            .collect_vec()
    }

    /// Run the precompile sum-check instances through [`BatchedSumcheck::prove`] protcol.
    #[tracing::instrument(skip_all, name = "PrecompileProof::prove")]
    pub fn prove(
        _pp: &PrecompilePreprocessing,
        witness: &mut [MatMultSumcheck<F>],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let init_claims = witness
            .iter()
            .map(|p| p.prover_state.as_ref().unwrap().input_claim)
            .collect_vec();
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<F, ProofTranscript>> = witness
            .iter_mut()
            .map(|p| p as &mut dyn BatchableSumcheckInstance<F, ProofTranscript>)
            .collect();
        let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, transcript);
        let final_claims = witness
            .iter()
            .map(|p| p.claims.as_ref().unwrap().clone())
            .collect_vec(); // TODO: Append these claims to opening accumulator
        Self {
            sumcheck_proof,
            init_claims,
            final_claims,
        }
    }

    /// Verify the sum-check precompile instances via [`BatchedSumcheck::verify`].
    #[tracing::instrument(skip_all, name = "PrecompileProof::verify")]
    pub fn verify(
        pp: &PrecompilePreprocessing,
        proof: &Self,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let vsumcheck_instances =
            Self::initialize_verifier(pp, &proof.init_claims, &proof.final_claims, transcript);
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> =
            vsumcheck_instances
                .iter()
                .map(|p| p as &dyn BatchableSumcheckInstance<F, ProofTranscript>)
                .collect();
        let _ = BatchedSumcheck::verify(&proof.sumcheck_proof, trait_objects, transcript)?;
        Ok(())
    }

    /// Initialize the verifier states for the precompile sum-check instances.
    /// Updates the transcript to be in sync with the prover's transcript.
    ///
    /// # Panics
    /// Panics if the length of `init_claims` and `final_claims` does not match the number of matrix multiplication precompile's
    fn initialize_verifier(
        pp: &PrecompilePreprocessing,
        init_claims: &[F],
        final_claims: &[MatMultClaims<F>],
        transcript: &mut ProofTranscript,
    ) -> Vec<MatMultSumcheck<F>> {
        let dims = &pp.mat_mult_precompile_dims;
        dims.iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
            .map(|((dim, init_claim), final_claim)| {
                // Initialize verifier state. We update transcript state as well, generating the challenges rx & ry & appending init_claim
                // for the matmult sum-check precompile proof.
                let verifier_state =
                    MatMultVerifierState::initialize(dim.0, dim.1, dim.2, *init_claim, transcript);

                // Create a new `MatMultSumcheck` instance with the verifier state and final claims.
                MatMultSumcheck::new(None, Some(verifier_state), Some(final_claim.clone()))
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::{PrecompilePreprocessing, PrecompileProof};
    use crate::{
        jolt_onnx::{
            onnx_host::ONNXProgram,
            precompiles::matmult::{MatMultPrecompile, MatMultProverState, MatMultSumcheck},
            tracer::tensor::QuantizedTensor,
            utils::random_floatvec,
            vm::precompiles::MatMultPrecompileDims,
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn test_precompile_proof() {
        let mut rng = test_rng();
        let mut program = ONNXProgram::new("onnx/mlp/perceptron_2.onnx", None);
        let pp = PrecompilePreprocessing::preprocess(&program.decode());
        let input = random_floatvec(&mut rng, 4);
        program.set_input(input);

        // Prover
        let (_io, trace) = program.trace();
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut witness = PrecompileProof::<Fr, _>::generate_witness(&trace, &mut ptranscript);
        assert!(!witness.is_empty());
        let proof = PrecompileProof::<Fr, _>::prove(&pp, &mut witness, &mut ptranscript);

        // Verifier
        let mut vtranscript = KeccakTranscript::new(b"test");
        PrecompileProof::<Fr, _>::verify(&pp, &proof, &mut vtranscript).unwrap();
    }

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 100;
        let mut pp: Vec<MatMultPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let m = (rng.next_u32() as usize % 100 + 1).next_power_of_two();
            let n = (rng.next_u32() as usize % 100 + 1).next_power_of_two();
            let k = (rng.next_u32() as usize % 100 + 1).next_power_of_two();
            pp.push((m, n, k));
            let a = QuantizedTensor::random(&mut rng, m, k);
            let b = QuantizedTensor::random(&mut rng, n, k);
            let precompile = MatMultPrecompile::new(a, b);
            let prover_state = MatMultProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = MatMultSumcheck::new(Some(prover_state), None, None);
            sumcheck_instances.push(sumcheck_instance);
        }

        // Preprocessing
        let pp = PrecompilePreprocessing {
            mat_mult_precompile_dims: pp,
        };

        // Prover
        let proof = PrecompileProof::<Fr, _>::prove(&pp, &mut sumcheck_instances, &mut ptranscript);

        // Verifier
        let mut vtranscript = KeccakTranscript::new(b"test");
        PrecompileProof::<Fr, _>::verify(&pp, &proof, &mut vtranscript)
            .expect("Verification failed");
    }
}
