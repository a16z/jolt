//! This module provides a matmult sum-check precompile.

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
    input_claim: F,
    final_claims: (F, F),
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
        let m = a.shape[0];
        // rhs is transposed
        let n = b.shape[0];
        let k = a.shape[1];
        let log_m = m.next_power_of_two().log_2();
        let log_n = n.next_power_of_two().log_2();
        let rx: Vec<F> = transcript.challenge_vector(log_m);
        let ry: Vec<F> = transcript.challenge_vector(log_n);
        let eq_rx = EqPolynomial::evals(&rx);
        let eq_ry = EqPolynomial::evals(&ry);
        let mut A_rx = vec![F::zero(); k];
        for i in 0..m {
            for j in 0..k {
                A_rx[j] += F::from_u8(a.data[i * k + k] as u8) * eq_rx[i];
            }
        }
        let mut B_rx = vec![F::zero(); k];
        for i in 0..n {
            for j in 0..k {
                B_rx[j] += F::from_u8(b.data[i * k + k] as u8) * eq_ry[i];
            }
        }
        A_rx.resize(k.next_power_of_two(), F::zero());
        B_rx.resize(k.next_power_of_two(), F::zero());
        let (mut c, _) = a.matmul_rhs_transposed(b);
        let c_len = c.len();
        c.resize(c_len.next_power_of_two(), 0);
        let c_poly = DensePolynomial::new(c.iter().map(|v| F::from_u32(*v as u32)).collect());
        let input_claim = c_poly.evaluate(&[rx.clone(), ry.clone()].concat());
        Self {
            a: DensePolynomial::new(A_rx),
            b: DensePolynomial::new(B_rx),
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
        self.a.num_vars
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

            // eval 2: bound_func is -A(low) + 2*A(high)
            let poly_A_bound_point = self.a[b + b_size] + self.a[b + b_size] - self.a[b];
            let poly_B_bound_point = self.b[b + b_size] + self.b[b + b_size] - self.b[b];
            uni_poly_evals[2] += poly_A_bound_point * poly_B_bound_point;
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
    use crate::jolt_onnx::utils::random_floatvec;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
        let (C, shape) = A.matmul_rhs_transposed(&B);
        println!("C: {C:#?}");
    }

    #[test]
    fn test_matmult_2() {
        let mut rng = StdRng::from_seed([1; 32]);
        let input = random_floatvec(&mut rng, 4);
        let program = ONNXProgram::new("onnx/perceptron_2.onnx", &input);
        let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
    }
}
