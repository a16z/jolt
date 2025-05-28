//! This module provides a matmult sum-check precompile.

use itertools::Itertools;

use crate::{
    field::JoltField,
    jolt_onnx::tracer::tensor::QuantizedLiteTensor,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};

use super::sumcheck_engine::BatchableSumcheckInstance;

/// Sum-check precompile for matrix multiplication
pub struct MatMultPrecompile<F>
where
    F: JoltField,
{
    a: DensePolynomial<F>,
    b: DensePolynomial<F>,
    num_vars: usize,
    input_claim: F,
    final_claims: (F, F),
}

impl<F> MatMultPrecompile<F>
where
    F: JoltField,
{
    fn new_explicit<ProofTranscript>(
        a: &QuantizedLiteTensor,
        b: &QuantizedLiteTensor,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let m = a.shape[0];
        let n = b.shape[1];
        let k = a.shape[1];
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<F> = transcript.challenge_vector(log_m);
        let ry: Vec<F> = transcript.challenge_vector(log_n);
        let eq_rx = EqPolynomial::evals(&rx);
        let eq_ry = EqPolynomial::evals(&ry);
        let mut A_rx = vec![F::zero(); k];
        for i in 0..m {
            for j in 0..k {
                A_rx[j] += F::from_u64(a.data[i * k + j] as u8 as u32 as u64) * eq_rx[i];
            }
        }
        let mut B_ry = vec![F::zero(); k];
        for i in 0..k {
            for j in 0..n {
                B_ry[i] += F::from_u64(b.data[i * n + j] as u8 as u32 as u64) * eq_ry[j]
            }
        }
        let (c, _) = a.matmult(b);
        let c_poly =
            DensePolynomial::new(c.iter().map(|v| F::from_u64(*v as u32 as u64)).collect());
        let input_claim = c_poly.evaluate(&[rx.clone(), ry.clone()].concat());
        let num_vars = A_rx.len().log_2();
        #[cfg(test)]
        {
            let sum: F = A_rx.iter().zip_eq(B_ry.iter()).map(|(a, b)| *a * b).sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            num_vars,
            a: DensePolynomial::new(A_rx),
            b: DensePolynomial::new(B_ry),
            input_claim,
            // we will populate this later
            final_claims: (F::zero(), F::zero()),
        }
    }

    fn new<ProofTranscript>(
        a: &QuantizedLiteTensor,
        b: &QuantizedLiteTensor,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let m = a.shape[0];
        // rhs is transposed
        let n = b.shape[0];
        let k = a.shape[1];
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<F> = transcript.challenge_vector(log_m);
        let ry: Vec<F> = transcript.challenge_vector(log_n);
        let eq_rx = EqPolynomial::evals(&rx);
        let eq_ry = EqPolynomial::evals(&ry);
        let mut A_rx = vec![F::zero(); k];
        for i in 0..m {
            for t in 0..k {
                A_rx[t] += F::from_u8(a.data[i * k + t] as u8) * eq_rx[i];
            }
        }
        let mut B_ry = vec![F::zero(); k];
        for j in 0..n {
            for t in 0..k {
                B_ry[t] += F::from_u8(b.data[j * k + t] as u8) * eq_ry[j];
            }
        }
        let (c, _) = a.matmul_rhs_transposed(b);
        let c_poly = DensePolynomial::new(c.iter().map(|v| F::from_u32(*v as u32)).collect());
        let input_claim = c_poly.evaluate(&[rx.clone(), ry.clone()].concat());
        let num_vars = A_rx.len().log_2();
        #[cfg(test)]
        {
            let sum: F = A_rx.iter().zip_eq(B_ry.iter()).map(|(a, b)| *a * b).sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            num_vars,
            a: DensePolynomial::new(A_rx),
            b: DensePolynomial::new(B_ry),
            input_claim,
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
            || self.a.bind(r_j, BindingOrder::LowToHigh),
            || self.b.bind(r_j, BindingOrder::LowToHigh),
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
    use crate::jolt_onnx;
    use crate::jolt_onnx::common::onnx_trace::Operator;
    use crate::jolt_onnx::onnx_host::ONNXProgram;
    use crate::jolt_onnx::precompiles::matmult::MatMultPrecompile;
    use crate::jolt_onnx::precompiles::sumcheck_engine::BatchedSumcheck;
    use crate::jolt_onnx::tracer::tensor::QuantizedLiteTensor;
    use crate::jolt_onnx::utils::random_floatvec;
    use crate::utils::math::Math;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::Fr;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // TODO: create own execution trace
    #[test]
    fn test_matmult_own() {
        let mut rng = StdRng::from_seed([1; 32]);
        let m = 1 << 3;
        let n = 1 << 4;
        let k = 1 << 3;
        let A = QuantizedLiteTensor::random(&mut rng, m, k);
        let B = QuantizedLiteTensor::random(&mut rng, k, n);
        let mut transcript = KeccakTranscript::new(b"test");
        let mut sc_inst: MatMultPrecompile<Fr> =
            MatMultPrecompile::new_explicit(&A, &B, &mut transcript);
        let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut sc_inst], &mut transcript);

        let mut vtranscript = KeccakTranscript::new(b"test");
        let m = A.shape[0];
        let n = B.shape[1];
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
        let ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        let _ = BatchedSumcheck::verify(&proof, vec![&mut sc_inst], &mut vtranscript).unwrap();
    }
    #[test]
    fn test_matmult_explicit() {
        let mut rng = StdRng::from_seed([1; 32]);
        let input = random_floatvec(&mut rng, 10);
        let program = ONNXProgram::new("onnx/perceptron.onnx", &input);
        let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
        let (A, B) = {
            let filter_trace = trace
                .iter()
                .filter(|row| matches!(row.instruction.opcode, Operator::MatMul))
                .collect::<Vec<_>>();
            let step = filter_trace
                .first()
                .expect("execution trace should have at least one row");
            let step_inputs = step
                .layer_state
                .input_vals
                .as_ref()
                .expect("input values should be present");
            (step_inputs[0].clone(), step_inputs[1].clone())
        };
        println!("A: {A:#?}, B: {B:#?}");
        let A = A.pad();
        let B = B.transpose();
        let B = B.pad();
        let mut transcript = KeccakTranscript::new(b"test");
        let mut sc_inst: MatMultPrecompile<Fr> =
            MatMultPrecompile::new_explicit(&A, &B, &mut transcript);
        let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut sc_inst], &mut transcript);

        let mut vtranscript = KeccakTranscript::new(b"test");
        let m = A.shape[0];
        let n = B.shape[1];
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
        let ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        let _ = BatchedSumcheck::verify(&proof, vec![&mut sc_inst], &mut vtranscript).unwrap();
    }

    #[test]
    fn test_matmult() {
        let mut rng = StdRng::from_seed([1; 32]);
        let input = random_floatvec(&mut rng, 10);
        let program = ONNXProgram::new("onnx/perceptron.onnx", &input);
        let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
        let (A, B) = {
            let filter_trace = trace
                .iter()
                .filter(|row| matches!(row.instruction.opcode, Operator::MatMul))
                .collect::<Vec<_>>();
            let step = filter_trace
                .first()
                .expect("execution trace should have at least one row");
            let step_inputs = step
                .layer_state
                .input_vals
                .as_ref()
                .expect("input values should be present");
            (step_inputs[0].clone(), step_inputs[1].clone())
        };
        println!("A: {A:#?}, B: {B:#?}");
        let A = A.pad();
        let B = B.pad();
        let mut transcript = KeccakTranscript::new(b"test");
        let mut sc_inst: MatMultPrecompile<Fr> = MatMultPrecompile::new(&A, &B, &mut transcript);
        let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut sc_inst], &mut transcript);

        let mut vtranscript = KeccakTranscript::new(b"test");
        let m = A.shape[0];
        // rhs is transposed
        let n = B.shape[0];
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<Fr> = vtranscript.challenge_vector(log_m);
        let ry: Vec<Fr> = vtranscript.challenge_vector(log_n);
        let _ = BatchedSumcheck::verify(&proof, vec![&mut sc_inst], &mut vtranscript).unwrap();
    }

    #[test]
    fn test_matmult_2() {
        let mut rng = StdRng::from_seed([1; 32]);
        let input = random_floatvec(&mut rng, 4);
        let program = ONNXProgram::new("onnx/perceptron_2.onnx", &input);
        let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
    }
}
