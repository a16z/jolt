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

/// The dimensions of the matrix multiplication precompile.
/// mat_mult_precompile_dims = (m, n, k) where
/// - `m` is the number of rows in the resulting matrix,
/// - `n` is the number of columns in the resulting matrix,
/// - `k` is the number of columns in the lhs matrix
///
/// # Note: We pad the dimensions to the next power of two.
pub type MatMultPrecompileDims = (usize, usize, usize);

/// Preprocessing of the models matrices for the precompile proof.
/// Store the dimensions of the matrix multiplication precompile.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrecompilePreprocessing {
    /// The dimensions of the matrix multiplication precompile.
    pub mat_mult_precompile_dims: Vec<MatMultPrecompileDims>,
}

impl PrecompilePreprocessing {
    /// Preprocess the ONNX model to extract the dimensions of the matrix multiplication precompile.
    #[tracing::instrument(skip_all, name = "PrecompilePreprocessing::preprocess")]
    pub fn preprocess(model: &QuantizedONNXModel) -> Self {
        let io_shapes = model.track_io_shapes();
        let mut mat_mult_precompile_dims = Vec::new();
        for (instr, io_shape) in model.instrs.iter().zip_eq(io_shapes.iter()) {
            match instr.opcode {
                Operator::MatMul => {
                    let input_shape = io_shape.0.clone();
                    let output_shape = io_shape.1.clone();
                    let m = output_shape[0].next_power_of_two();
                    let n = output_shape[1].next_power_of_two();
                    let k = input_shape[1].next_power_of_two();
                    mat_mult_precompile_dims.push((m, n, k));
                }
                _ => continue,
            }
        }
        Self {
            mat_mult_precompile_dims,
        }
    }
}

/// A special-purpose SNARK designed for specific functionality, such as ONNX operators that are too expensive to prove using [`InstructionLookupProof`].
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
    /// Given the execution trace, constructs the polynomials used in the batched sum-check proof.
    pub fn generate_witness<InstructionSet>(
        ops: &[JoltONNXTraceStep<InstructionSet>],
        transcript: &mut ProofTranscript,
    ) -> Vec<MatMultSumcheck<F>>
    where
        InstructionSet: JoltInstructionSet,
    {
        let filter_ops = ops
            .iter()
            .filter(|ops| ops.precompile.is_some())
            .collect_vec();
        filter_ops
            .iter()
            .map(|op| {
                let precompile = op.precompile.as_ref().unwrap();
                match precompile {
                    PrecompileOperators::MatMult(mat_mult) => {
                        let prover_state: MatMultProverState<F> =
                            MatMultProverState::initialize(mat_mult, transcript);
                        MatMultSumcheck::new(Some(prover_state), None, None)
                    }
                }
            })
            .collect_vec()
    }

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
            .collect_vec();
        Self {
            sumcheck_proof,
            init_claims,
            final_claims,
        }
    }

    #[tracing::instrument(skip_all, name = "PrecompileProof::prove")]
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

    pub fn initialize_verifier(
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
                let verifier_state =
                    MatMultVerifierState::initialize(dim.0, dim.1, dim.2, *init_claim, transcript);
                MatMultSumcheck::new(None, Some(verifier_state), Some(final_claim.clone()))
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::{PrecompilePreprocessing, PrecompileProof};
    use crate::{
        jolt_onnx::{onnx_host::ONNXProgram, utils::random_floatvec},
        utils::transcript::{KeccakTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn test_precompile_proof() {
        let mut rng = test_rng();
        let mut program = ONNXProgram::new("onnx/perceptron_2.onnx", None);
        let pp = PrecompilePreprocessing::preprocess(&program.decode());
        let input = random_floatvec(&mut rng, 4);
        program.set_input(input);

        // Prover
        let (io, trace) = program.trace();
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut witness = PrecompileProof::<Fr, _>::generate_witness(&trace, &mut ptranscript);
        let proof = PrecompileProof::<Fr, _>::prove(&pp, &mut witness, &mut ptranscript);

        // Verifier
        let mut vtranscript = KeccakTranscript::new(b"test");
        let _ = PrecompileProof::<Fr, _>::verify(&pp, &proof, &mut vtranscript).unwrap();
    }
}
