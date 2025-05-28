use crate::{
    field::JoltField,
    jolt_onnx::tracer::tensor::QuantizedLiteTensor,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use itertools::Itertools;

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
        a: &QuantizedLiteTensor,
        b: &QuantizedLiteTensor,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let m = a.m();
        // b is implicitly transposed
        let n = b.m();
        let k = a.n();
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<F> = transcript.challenge_vector(log_m);
        let ry: Vec<F> = transcript.challenge_vector(log_n);
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

#[cfg(test)]
mod tests {
    use crate::{
        jolt_onnx::{
            precompiles::sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
            tracer::tensor::QuantizedLiteTensor,
        },
        utils::{
            math::Math,
            transcript::{KeccakTranscript, Transcript},
        },
    };
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng};
    use itertools::Itertools;

    use super::MatMultPrecompile;

    #[test]
    fn test_matmult() {
        let mut rng = test_rng();
        let m = 100;
        let n = 200;
        let k = 300;
        let a = QuantizedLiteTensor::random(&mut rng, m, k).pad();
        let b = QuantizedLiteTensor::random(&mut rng, n, k).pad();
        let m = a.m();
        // b is implicitly transposed
        let n = b.m();

        let mut transcript = KeccakTranscript::new(b"test");
        let mut precompile = MatMultPrecompile::<Fr>::new(&a, &b, &mut transcript);
        let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut precompile], &mut transcript);
        let mut vtranscript = KeccakTranscript::new(b"test");
        let log_m = m.log_2();
        let log_n = n.log_2();
        let _rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
        let _ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        let _ = BatchedSumcheck::verify(&proof, vec![&mut precompile], &mut vtranscript).unwrap();
    }

    #[test]
    fn test_matmult_batch() {
        let execution_trace_length = 20;
        let mut rng = test_rng();
        let mut transcript = KeccakTranscript::new(b"test");
        let mut vtranscript = KeccakTranscript::new(b"test");

        let mut precompiles = Vec::with_capacity(execution_trace_length);
        for _ in 0..execution_trace_length {
            let m = rng.gen_range(1..=100);
            let n = rng.gen_range(1..=100);
            let k = rng.gen_range(1..=100);
            let a = QuantizedLiteTensor::random(&mut rng, m, k).pad();
            let b = QuantizedLiteTensor::random(&mut rng, n, k).pad();
            // Get new padded m & n values
            let m = a.m();
            // b is implicitly transposed
            let n = b.m();
            let precompile = MatMultPrecompile::<Fr>::new(&a, &b, &mut transcript);
            precompiles.push(precompile);

            // Update verifier transcript
            let log_m = m.log_2();
            let log_n = n.log_2();
            let _rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
            let _ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        }

        // prover
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            precompiles
                .iter_mut()
                .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let (proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut transcript);

        // verifier
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> = precompiles
            .iter_mut()
            .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
            .collect();
        let _ = BatchedSumcheck::verify(&proof, trait_objects, &mut vtranscript).unwrap();
    }
}
