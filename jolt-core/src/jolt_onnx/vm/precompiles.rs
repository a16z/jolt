//! Precompile proofs for ONNX runtime.
//!
//! This module provides a custom SNARK for precompiles in the [`ONNXJoltVM`].
//! These precompile proofs are sum-check based.

use itertools::Itertools;

use crate::{
    field::JoltField,
    jolt::instruction::JoltInstructionSet,
    jolt_onnx::precompiles::{
        matmult::{MatMultProverState, MatMultSumcheck},
        PrecompileOperators,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::transcript::Transcript,
};

use super::JoltONNXTraceStep;

/// A special-purpose SNARK designed for specific functionality, such as ONNX operators that are too expensive to prove using [`InstructionLookupProof`].
/// This is a sum-check-based precompile proof tailored for ONNX runtime.
/// It is used to prove the correctness of certain ONNX operators via a custom sum-check precompile instead of a lookup-based approach.
pub struct PrecompileProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
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
    ) where
        InstructionSet: JoltInstructionSet,
    {
        let filter_ops = ops
            .iter()
            .filter(|ops| ops.precompile.is_some())
            .collect_vec();
        let precompiles = filter_ops
            .iter()
            .map(|op| {
                let precompile = op.precompile.as_ref().unwrap();
                match precompile {
                    PrecompileOperators::MatMult(mat_mult) => {
                        let prover_state: MatMultProverState<F> =
                            MatMultProverState::initialize(mat_mult, transcript);
                        MatMultSumcheck::new(Some(prover_state))
                    }
                }
            })
            .collect_vec();
    }
}
