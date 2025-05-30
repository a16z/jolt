use crate::{
    field::JoltField,
    jolt_onnx::tracer::tensor::QuantizedLiteTensor,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::sumcheck_engine::BatchableSumcheckInstance;

/// This struct represents a precompile for matrix multiplication.
/// Used to generate the witness for matrix multiplication in Jolt's ONNX execution.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MatMultPrecompile {
    a: QuantizedLiteTensor,
    b: QuantizedLiteTensor,
}

impl MatMultPrecompile {
    /// Create a new instance of [`MatMultPrecompile`]
    pub fn new(a: QuantizedLiteTensor, b: QuantizedLiteTensor) -> Self {
        Self { a, b }
    }

    /// Return the lhs matrix of the multiplication
    pub fn a(&self) -> &QuantizedLiteTensor {
        &self.a
    }

    /// Return the rhs matrix of the multiplication
    pub fn b(&self) -> &QuantizedLiteTensor {
        &self.b
    }
}

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
    /// Create a new instance of [`MatMultVerifierState`]
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultProverState<F>
where
    F: JoltField,
{
    a: DensePolynomial<F>,
    b: DensePolynomial<F>,
    pub input_claim: F,
    num_rounds: usize,
}

impl<F> MatMultProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`MatMultProverState`]
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultClaims<F>
where
    F: JoltField,
{
    a: F,
    b: F,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultSumcheck<F>
where
    F: JoltField,
{
    pub prover_state: Option<MatMultProverState<F>>,
    pub verifier_state: Option<MatMultVerifierState<F>>,
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
        let mut uni_poly_evals = vec![F::zero(); 2];
        for i in 0..len {
            uni_poly_evals[0] += a[i] * b[i];
            let poly_A_bound_point = a[i + len] + a[i + len] - a[i];
            let poly_B_bound_point = b[i + len] + b[i + len] - b[i];
            uni_poly_evals[1] += poly_A_bound_point * poly_B_bound_point;
        }
        uni_poly_evals
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let MatMultProverState { a, b, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || a.bind(r_j, BindingOrder::HighToLow),
            || b.bind(r_j, BindingOrder::HighToLow),
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
