//! Precompile proofs for ONNX runtime.
//!
//! This module provides a custom SNARK for precompiles in the [`ONNXJoltVM`].
//! These precompile proofs are sum-check based.

use super::JoltONNXTraceStep;
use crate::{
    field::JoltField,
    jolt::{instruction::JoltInstructionSet, vm::rv32i_vm::ProofTranscript},
    jolt_onnx::precompiles::{
        matmult::{MatMultProverState, MatMultSumcheck},
        sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
        PrecompileOperators,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use itertools::Itertools;

/// A special-purpose SNARK designed for specific functionality, such as ONNX operators that are too expensive to prove using [`InstructionLookupProof`].
/// This is a sum-check-based precompile proof tailored for ONNX runtime.
/// It is used to prove the correctness of certain ONNX operators via a custom sum-check precompile instead of a lookup-based approach.
pub struct PrecompileProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    init_claims: Vec<F>,
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
                        MatMultSumcheck::new(Some(prover_state), None)
                    }
                }
            })
            .collect_vec()
    }

    #[tracing::instrument(skip_all, name = "PrecompileProof::prove")]
    pub fn prove(witness: &mut [MatMultSumcheck<F>], transcript: &mut ProofTranscript) -> Self {
        let init_claims = witness
            .iter()
            .map(|p| p.prover_state.as_ref().unwrap().input_claim.clone())
            .collect_vec();
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<F, ProofTranscript>> = witness
            .iter_mut()
            .map(|p| p as &mut dyn BatchableSumcheckInstance<F, ProofTranscript>)
            .collect();
        let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, transcript);
        Self {
            sumcheck_proof,
            init_claims,
        }
    }

    #[tracing::instrument(skip_all, name = "PrecompileProof::prove")]
    pub fn verify(proof: &Self, transcript: &mut ProofTranscript) -> Result<(), ProofVerifyError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::test_rng;

    use crate::{
        jolt_onnx::{onnx_host::ONNXProgram, utils::random_floatvec},
        utils::transcript::{KeccakTranscript, Transcript},
    };

    use super::PrecompileProof;

    #[test]
    fn test_precompile_proof() {
        let mut rng = test_rng();
        let input = random_floatvec(&mut rng, 4);
        let program = ONNXProgram::new("onnx/perceptron_2.onnx", Some(input));

        // Prover
        let (io, trace) = program.trace();
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut witness = PrecompileProof::<Fr, _>::generate_witness(&trace, &mut ptranscript);
        let proof = PrecompileProof::<Fr, _>::prove(&mut witness, &mut ptranscript);

        // Verifier
        let mut vtranscript = KeccakTranscript::new(b"test");
    }
}
