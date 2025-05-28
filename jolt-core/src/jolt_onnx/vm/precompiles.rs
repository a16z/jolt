//! Precompile proofs for ONNX runtime.
//!
//! This module provides a custom SNARK for precompiles in the [`ONNXJoltVM`].
//! These precompile proofs are sum-check based.

use crate::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, utils::transcript::Transcript,
};

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
    pub fn generate_witness() {}
}
