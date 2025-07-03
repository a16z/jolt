//! A sum-check precompile implementation for softmax operation.
//! Used for proving correctness of the execution of the softmax ONNX operator.
//! You can see it in action in [`crate::jolt_onnx::vm::precompiles`]
//!
//! # Overview:
//!   - [`SoftmaxPrecompile`] - We specify the precompile for softmax op, by defining the input (z) vector.
//!   - [`SoftmaxSumcheck`] - Defines the prover and verifier states that will be used to instantiate a [`super::sumcheck_engine::BatchedSumcheck`] instance.
//!     These sum-check instances are then fed into [`super::sumcheck_engine::BatchedSumcheck::prove`] and [`super::sumcheck_engine::BatchedSumcheck::verify`].
//!   - [`SoftmaxProverState`] - Handles/Defines the prover state for the softmax sum-check precompile (handles witness polynomials for sum-check prover).
//!   - [`SoftmaxVerifierState`] - Handles/Defines the verifier state for the softmax sum-check precompile.

use crate::{
    field::JoltField,
    jolt_onnx::precompiles::sumcheck_engine::BatchableSumcheckInstance,
    poly::{
        dense_mlpoly::DensePolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// A type defining the sum precompile in the execution trace.
/// The type is used to intialize the [`SumProverState`]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SumPrecompile {
    z: Vec<u64>,
}

impl SumPrecompile {
    /// Create a new instance of [`SumPrecompile`].
    pub fn new(z: Vec<u64>) -> Self {
        Self { z }
    }

    fn z_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        DensePolynomial::new(self.z.iter().map(|&x| F::from_u64(x as u64)).collect_vec())
    }
}

/// Container type to manage the prover state in the [`BatchableSumcheckInstance`] for the sum precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumProverState<F>
where
    F: JoltField,
{
    /// sum polynomial as evaluations over the boolean hypercube
    z_poly: DensePolynomial<F>,
    /// number of remaining folding rounds
    num_rounds: usize,
    /// z_poly(r)
    input_claim: F,
}

impl<F> SumProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`SumProverState`].
    ///
    /// We apply sum-check to the log(n) variate polynomial Σₖ z(k) * eq(k, r)
    pub fn initialize<ProofTranscript>(
        input: &SumPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let n = input.z.len();
        let num_rounds = n.log_2();
        // let r: Vec<F> = transcript.challenge_scalar_powers(num_rounds);

        let z_poly = input.z_poly();
        // let input_claim = z_poly.evaluate(&r);
        let input_claim = F::from_u64(input.z.iter().sum::<u64>());

        transcript.append_scalar(&input_claim);
        Self {
            z_poly,
            input_claim,
            num_rounds,
        }
    }

}

/// Dimensions for the sum inputs.
#[derive(Clone, Serialize, Deserialize, Debug, Copy)]
pub struct SumPrecompileDims {
    /// Length of the input vector
    pub n: usize,
}

/// Container type to manage the verifier state in the [`BatchableSumcheckInstance`] for the softmax precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> SumVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`SumVerifierState`].
    pub fn initialize<ProofTranscript>(
        dims: SumPrecompileDims,
        input_claim: F,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let num_rounds = dims.n.log_2();
        // let _ri: Vec<F> = transcript.challenge_scalar_powers(dims.n.log_2());
        transcript.append_scalar(&input_claim);
        Self {
            num_rounds,
            input_claim,
        }
    }
}

/// Stores the evaluations of the s polynomial at `r_i`
/// Where:
///   - `r_i` ∈ F^{log(n)}
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumClaims<F>
where
    F: JoltField,
{
    sum: F,
}

/// Batchable sum-check instance for softmax precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<SumProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<SumVerifierState<F>>,
    /// Holds the final claims for the softmax sum-check precompile.
    pub claims: Option<SumClaims<F>>,
}

impl<F> SumSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`SoftmaxSumcheck`]
    pub fn new(
        prover_state: Option<SumProverState<F>>,
        verifier_state: Option<SumVerifierState<F>>,
        claims: Option<SumClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for SumSumcheck<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().num_rounds
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().input_claim
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().input_claim
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        let SumProverState {
            z_poly, ..
        } = self.prover_state.as_ref().unwrap();
        let len = z_poly.len() / 2; 
        let g0 = (0..len)
            .into_iter()
            .map(|i| {
                z_poly[i]
            })
            .reduce(|acc, v| acc + v)
            .unwrap_or(F::zero());
        vec![g0]
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r: F, _: usize) {
        let SumProverState { z_poly, .. } = self.prover_state.as_mut().unwrap();
        z_poly.bind_parallel(r, BindingOrder::HighToLow);
    }

    fn cache_openings(&mut self) {
        let SumProverState {
            z_poly, ..
        } = self.prover_state.as_ref().unwrap();
        self.claims = Some(SumClaims {
            sum: z_poly[0],
        });
    }

    /// final check: Σ_i z_i
    fn expected_output_claim(&self, _: &[F]) -> F {
        let SumClaims { sum, .. } = self.claims.as_ref().unwrap();
        *sum
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        jolt_onnx::precompiles::{
            sum::{
                SumPrecompile, SumPrecompileDims, SumProverState, SumSumcheck, SumVerifierState,
            },
            sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng};
    use itertools::Itertools;
    use rand_core::RngCore;

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 10;
        let mut pp: Vec<SumPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let n = (rng.next_u32() as usize % 200 + 50).next_power_of_two();
            let z: Vec<u8> = (0..n)
                .map(|_| rng.gen())
                .collect_vec();
            let precompile = SumPrecompile::new(z.iter().map(|&x| x as u64).collect_vec());
            pp.push(SumPrecompileDims { n });
            let prover_state = SumProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = SumSumcheck::new(Some(prover_state), None, None);
            sumcheck_instances.push(sumcheck_instance);
        }
        let init_claims = sumcheck_instances
            .iter()
            .map(|p| p.prover_state.as_ref().unwrap().input_claim)
            .collect_vec();
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            sumcheck_instances
                .iter_mut()
                .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut ptranscript);
        let final_claims = sumcheck_instances
            .iter()
            .map(|p| p.claims.as_ref().unwrap().clone())
            .collect_vec();
        let mut vtranscript = KeccakTranscript::new(b"test");
        let mut vsumcheck_instances = Vec::with_capacity(trace_length);
        for ((dims, init_claim), final_claim) in pp
            .iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
        {
            let verifier_state =
                SumVerifierState::<Fr>::initialize(*dims, *init_claim, &mut vtranscript);
            vsumcheck_instances.push(SumSumcheck::new(
                None,
                Some(verifier_state),
                Some(final_claim.clone()),
            ))
        }
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            vsumcheck_instances
                .iter()
                .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let _r = BatchedSumcheck::verify(&sumcheck_proof, trait_objects, &mut vtranscript).unwrap();
    }
}