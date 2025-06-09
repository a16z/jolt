//! A sum-check precompile for matrix multiplication (where the rhs matrix is implicitly transposed).
//! Used when proving correctness of ONNX operators that do matrix multiplication.

use crate::{
    field::JoltField,
    jolt_onnx::tracer::tensor::QuantizedTensor,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::sumcheck_engine::BatchableSumcheckInstance;

/// This struct represents a precompile for matrix multiplication (where we implicitly transpose B).
/// Used to generate the witness for matrix multiplication in Jolt's ONNX execution.
///
/// # Note: We assume tensors are appropriately padded here.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MatMultPrecompile {
    a: QuantizedTensor,
    b: QuantizedTensor,
}

impl MatMultPrecompile {
    /// Create a new instance of [`MatMultPrecompile`]
    pub fn new(a: QuantizedTensor, b: QuantizedTensor) -> Self {
        Self { a, b }
    }

    /// Return the lhs matrix of the multiplication
    pub fn a(&self) -> &QuantizedTensor {
        &self.a
    }

    /// Return the rhs matrix of the multiplication
    pub fn b(&self) -> &QuantizedTensor {
        &self.b
    }
}

/// Handles the verifier state for the matrix multiplication sum-check precompile.
/// Used to create the sum-check verifier instance to input into [`BatchedSumcheck::verify`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> MatMultVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`MatMultVerifierState`].
    /// # Note: we mainly update the state by computing the necessary challenges used in the sum-check matmult protocol.
    ///         We also append the input claim to the transcript.
    pub fn initialize<ProofTranscript>(
        m: usize,
        n: usize,
        k: usize,
        input_claim: F,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let num_rounds = k.log_2();
        let log_m = m.log_2();
        let log_n = n.log_2();
        let _rx: Vec<F> = transcript.challenge_scalar_powers(log_m);
        let _ry: Vec<F> = transcript.challenge_scalar_powers(log_n);
        transcript.append_scalar(&input_claim);
        Self {
            num_rounds,
            input_claim,
        }
    }
}

/// Handles the prover state for the matrix multiplication sum-check precompile.
/// Used to create the sum-check prover instance to input into [`BatchedSumcheck::prove`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultProverState<F>
where
    F: JoltField,
{
    /// A(rx, k) evaluations over the boolean hypercube
    pub a: DensePolynomial<F>,
    /// B(ry, k) evaluations over the boolean hypercube
    pub b: DensePolynomial<F>,
    /// C(rx, ry)
    pub input_claim: F,
    /// Number of rounds in the sum-check precompile
    pub num_rounds: usize,
}

impl<F> MatMultProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`MatMultProverState`].
    /// We compute the evaluations of the polynomials A(rx, k) and B(ry, k) over the boolean hypercube,
    /// and also compute the input claim C(rx, ry) = Σₖ A(rx, k) * B(ry, k).
    ///
    /// These A(rx, k) and B(ry, k) evaluations serve as the witness for the matrix multiplication precompile.
    ///
    /// # Note: we implicitly transpose the rhs matrix B in the multiplication.
    ///         This is because the ONNX genneral matmul operator (GEMM) transposes the second matrix.
    pub fn initialize<ProofTranscript>(
        input: &MatMultPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let a = input.a();
        let b = input.b();
        let m = a.m();
        // b is implicitly transposed
        let n = b.m();
        let k = a.n();
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<F> = transcript.challenge_scalar_powers(log_m);
        let ry: Vec<F> = transcript.challenge_scalar_powers(log_n);
        let eq_rx = EqPolynomial::evals(&rx);
        let eq_ry = EqPolynomial::evals(&ry);
        let mut A_rx = vec![F::zero(); k];
        for i in 0..m {
            for j in 0..k {
                A_rx[j] += F::from_i64(a.data[i * k + j] as i64) * eq_rx[i];
            }
        }
        // We index B as if it were transposed, i.e., B[i][j] = b.data[j * k + i]
        let mut B_ry = vec![F::zero(); k];
        for i in 0..n {
            for j in 0..k {
                B_ry[j] += F::from_i64(b.data[i * k + j] as i64) * eq_ry[i]
            }
        }
        let (c, _c_shape) = a.matmul_rhs_transposed(b);
        let c_poly = DensePolynomial::new(c.iter().map(|&x| F::from_i64(x as i64)).collect_vec());
        let input_claim = c_poly.evaluate(&[rx.clone(), ry.clone()].concat());
        transcript.append_scalar(&input_claim);
        #[cfg(test)]
        {
            let sum: F = A_rx.iter().zip_eq(B_ry.iter()).map(|(a, b)| *a * b).sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            a: DensePolynomial::new(A_rx),
            b: DensePolynomial::new(B_ry),
            input_claim,
            num_rounds: k.log_2(),
        }
    }
}

/// The final claims for the matrix multiplication sum-check precompile.
///
/// a = A(rx, r_sc)
/// b = B(ry, r_sc)
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultClaims<F>
where
    F: JoltField,
{
    a: F,
    b: F,
}

/// Batchable sum-check instance for matrix multiplication precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<MatMultProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<MatMultVerifierState<F>>,
    /// Holds the final claims for the matrix multiplication sum-check precompile.
    pub claims: Option<MatMultClaims<F>>,
}

impl<F> MatMultSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`MatMultSumcheck`]
    pub fn new(
        prover_state: Option<MatMultProverState<F>>,
        verifier_state: Option<MatMultVerifierState<F>>,
        claims: Option<MatMultClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for MatMultSumcheck<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
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
        let MatMultProverState { a, b, .. } = self.prover_state.as_ref().unwrap();
        let len = a.len() / 2;
        let univariate_poly_evals: [F; 2] = (0..len)
            .into_par_iter()
            .map(|i| {
                let poly_A_bound_point = a[i + len] + a[i + len] - a[i];
                let poly_B_bound_point = b[i + len] + b[i + len] - b[i];
                [a[i] * b[i], poly_A_bound_point * poly_B_bound_point]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let MatMultProverState { a, b, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || a.bind_parallel(r_j, BindingOrder::HighToLow),
            || b.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let MatMultProverState { a, b, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(MatMultClaims { a: a[0], b: b[0] });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let MatMultClaims { a, b } = self.claims.as_ref().unwrap();
        *a * b
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        jolt_onnx::{
            precompiles::{
                matmult::{
                    MatMultPrecompile, MatMultProverState, MatMultSumcheck, MatMultVerifierState,
                },
                sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
            },
            tracer::tensor::QuantizedTensor,
            vm::precompiles::MatMultPrecompileDims,
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use itertools::Itertools;
    use rand_core::RngCore;

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 10;
        let mut pp: Vec<MatMultPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let m = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            let n = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            let k = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            pp.push((m, n, k));
            let a = QuantizedTensor::random(&mut rng, m, k);
            let b = QuantizedTensor::random(&mut rng, n, k);
            let precompile = MatMultPrecompile::new(a, b);
            let prover_state = MatMultProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = MatMultSumcheck::new(Some(prover_state), None, None);
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
        for (((m, n, k), init_claim), final_claim) in pp
            .iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
        {
            let verifier_state =
                MatMultVerifierState::<Fr>::initialize(*m, *n, *k, *init_claim, &mut vtranscript);
            vsumcheck_instances.push(MatMultSumcheck::new(
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
