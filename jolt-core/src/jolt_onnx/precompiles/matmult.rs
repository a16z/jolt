use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{math::Math, transcript::Transcript},
};
use itertools::Itertools;
use rand_core::RngCore;

use super::sumcheck_engine::BatchableSumcheckInstance;

#[derive(Clone, Debug)]
pub struct MatMultPrecompile<F>
where
    F: JoltField,
{
    a: DensePolynomial<F>,
    b: DensePolynomial<F>,
    input_claim: F,
    final_claims: (F, F),
    num_vars: usize,
}

impl<F> MatMultPrecompile<F>
where
    F: JoltField,
{
    fn new<ProofTranscript>(
        a: &Matrix<i8>,
        b: &Matrix<i8>,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let m = a.m;
        // b is implicitly transposed
        let n = b.m;
        let k = a.n;
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<F> = transcript.challenge_vector(log_m);
        let ry: Vec<F> = transcript.challenge_vector(log_n);
        let eq_rx = EqPolynomial::evals(&rx);
        let eq_ry = EqPolynomial::evals(&ry);
        let mut A_rx = vec![F::zero(); k];
        for i in 0..m {
            for j in 0..k {
                A_rx[j] += F::from_u8(a.entries[i * k + j] as u8) * eq_rx[i];
            }
        }
        let mut B_ry = vec![F::zero(); k];
        for i in 0..n {
            for j in 0..k {
                B_ry[j] += F::from_u8(b.entries[i * k + j] as u8) * eq_ry[i]
            }
        }
        let c = Matrix::<i8>::matmult_transposed(a, b);
        let c_poly = DensePolynomial::new(
            c.entries
                .iter()
                .map(|&x| F::from_u32(x as u32))
                .collect_vec(),
        );
        let input_claim = c_poly.evaluate(&[rx.clone(), ry.clone()].concat());
        let num_vars = A_rx.len().log_2();
        #[cfg(test)]
        {
            let sum: F = A_rx.iter().zip_eq(B_ry.iter()).map(|(a, b)| *a * b).sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            a: DensePolynomial::new(A_rx),
            b: DensePolynomial::new(B_ry),
            input_claim,
            num_vars,
            // we will populate this later
            final_claims: (F::zero(), F::zero()),
        }
    }

    pub fn prove<ProofTranscript>(
        &self,
        transcript: &mut ProofTranscript,
    ) -> SumcheckInstanceProof<F, ProofTranscript>
    where
        ProofTranscript: Transcript,
    {
        let mut polys = vec![
            MultilinearPolynomial::LargeScalars(self.a.clone()),
            MultilinearPolynomial::LargeScalars(self.b.clone()),
        ];
        let (proof, _, _) = SumcheckInstanceProof::prove_arbitrary(
            &self.input_claim,
            self.num_vars,
            &mut polys,
            |v| v[0] * v[1],
            2,
            transcript,
        );
        proof
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for MatMultPrecompile<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        let b_size = self.a.len() / 2;
        let mut uni_poly_evals = vec![F::zero(); 2];
        for b in 0..b_size {
            uni_poly_evals[0] += self.a[b] * self.b[b];
            let poly_A_bound_point = self.a[b + b_size] + self.a[b + b_size] - self.a[b];
            let poly_B_bound_point = self.b[b + b_size] + self.b[b + b_size] - self.b[b];
            uni_poly_evals[1] += poly_A_bound_point * poly_B_bound_point;
        }
        uni_poly_evals
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        rayon::join(
            || self.a.bind(r_j, BindingOrder::HighToLow),
            || self.b.bind(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings(&mut self) {
        self.final_claims = (self.a[0], self.b[0])
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        self.final_claims.0 * self.final_claims.1
    }
}

/// A dense matrix over a prime field.
pub struct Matrix<T> {
    /// Row-major entries of the matrix. Length must equal `m * n`.
    pub entries: Vec<T>,
    /// num rows
    pub m: usize,
    /// num cols
    pub n: usize,
}

impl<T> Matrix<T> {
    /// We implicitly treat b as transposed
    pub fn matmult_transposed(a: &Matrix<i8>, b: &Matrix<i8>) -> Matrix<i32> {
        // check inner dimensions
        assert_eq!(a.n, b.n);
        let m = a.m;
        // Implicitly transpose b
        let n = b.m;
        let k = a.n; // shared dimension
        let mut entries = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut dot_product = 0i32;
                for t in 0..k {
                    let a_val = a.entries[i * k + t] as i32;
                    let b_val = b.entries[j * k + t] as i32;
                    dot_product += a_val * b_val;
                }
                entries[i * n + j] = dot_product;
            }
        }

        Matrix::<i32>::new(entries, m, n)
    }

    /// Create a new matrix with the given entries, rows, and columns.
    pub fn new(entries: Vec<T>, m: usize, n: usize) -> Self {
        assert_eq!(entries.len(), m * n);
        Self { entries, m, n }
    }

    /// Create a random matrix of size N x N over the finite field F.
    ///
    /// # Note
    ///
    /// - Useful for testing and benchmarking.
    pub fn random(mut rng: impl RngCore, m: usize, n: usize) -> Matrix<i8> {
        let entries = (0..m * n)
            .map(|_| (rng.next_u32() % 256) as i8)
            .collect_vec();
        Matrix::<i8>::new(entries, m, n)
    }

    pub fn pad(&self) -> Self
    where
        T: Clone + Default,
    {
        let padded_m = self.m.next_power_of_two();
        let padded_n = self.n.next_power_of_two();
        let mut entries = self.entries.clone();
        entries.resize(padded_m * padded_n, T::default());
        Self {
            entries,
            m: padded_m,
            n: padded_n,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        jolt_onnx::precompiles::sumcheck_engine::BatchedSumcheck,
        subprotocols::sumcheck::SumcheckInstanceProof,
        utils::{
            math::Math,
            transcript::{KeccakTranscript, Transcript},
        },
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand::{rngs::StdRng, SeedableRng};

    use super::{MatMultPrecompile, Matrix};

    #[test]
    fn test_matmult() {
        let mut rng = test_rng();
        let m = 100;
        let n = 200;
        let k = 300;
        let a = Matrix::<i8>::random(&mut rng, m, k).pad();
        let b = Matrix::<i8>::random(&mut rng, n, k).pad();
        let m = a.m;
        // b is implicitly transposed
        let n = b.m;

        let mut transcript = KeccakTranscript::new(b"test");
        let mut precompile = MatMultPrecompile::<Fr>::new(&a, &b, &mut transcript);
        let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut precompile], &mut transcript);
        let mut vtranscript = KeccakTranscript::new(b"test");
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
        let ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        let _ = BatchedSumcheck::verify(&proof, vec![&mut precompile], &mut vtranscript).unwrap();
    }

    #[test]
    fn test_matmult_non_batched() {
        let mut rng = test_rng();
        let m = 1 << 3;
        let n = 1 << 4;
        let k = 1 << 3;
        let a = Matrix::<Fr>::random(&mut rng, m, k);
        let b = Matrix::<Fr>::random(&mut rng, k, n);

        let mut transcript = KeccakTranscript::new(b"test");
        let mut precompile = MatMultPrecompile::<Fr>::new(&a, &b, &mut transcript);
        let proof = precompile.prove(&mut transcript);
        // let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut precompile], &mut transcript);

        let mut vtranscript = KeccakTranscript::new(b"test");
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
        let ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        let _ = proof
            .verify(
                precompile.input_claim,
                precompile.num_vars,
                2,
                &mut vtranscript,
            )
            .unwrap();
        // let _ = BatchedSumcheck::verify(&proof, vec![&mut precompile], &mut vtranscript);
    }
}
