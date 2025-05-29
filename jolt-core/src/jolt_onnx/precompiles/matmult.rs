use std::marker::PhantomData;

use crate::{
    field::JoltField,
    jolt::vm::rv32i_vm::ProofTranscript,
    jolt_onnx::tracer::tensor::QuantizedLiteTensor,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
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
    _field: PhantomData<F>,
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
        #[cfg(test)]
        {
            let sum: F = A_rx.iter().zip_eq(B_ry.iter()).map(|(a, b)| *a * b).sum();
            assert_eq!(sum, input_claim)
        }
        let num_rounds = A_rx.len().log_2();
        Self {
            a: DensePolynomial::new(A_rx),
            b: DensePolynomial::new(B_ry),
            input_claim,
            num_rounds,
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
    verifier_state: Option<MatMultVerifierState<F>>,
    claims: Option<MatMultClaims<F>>,
}

impl<F> MatMultSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`MatMultSumcheck`]
    pub fn new(
        prover_state: Option<MatMultProverState<F>>,
        verifier_state: Option<MatMultVerifierState<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims: None,
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
            todo!()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().input_claim
        } else if self.verifier_state.is_some() {
            todo!()
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
        // self.final_claims.0 * self.final_claims.1
        todo!()
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::{
//         jolt_onnx::{
//             self,
//             common::onnx_trace::Operator,
//             onnx_host::ONNXProgram,
//             precompiles::sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
//             tracer::tensor::QuantizedLiteTensor,
//             utils::random_floatvec,
//         },
//         utils::{
//             math::Math,
//             transcript::{KeccakTranscript, Transcript},
//         },
//     };
//     use ark_bn254::Fr;
//     use ark_std::{rand::Rng, test_rng};

//     use super::MatMultSumcheck;

//     #[test]
//     fn test_perceptron_2() {
//         let mut rng = test_rng();
//         let input = random_floatvec(&mut rng, 4);
//         let program = ONNXProgram::new("onnx/perceptron_2.onnx", &input);
//         let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
//         let mut transcript = KeccakTranscript::new(b"test");
//         let mut vtranscript = KeccakTranscript::new(b"test");
//         let mut precompiles = Vec::new();
//         let filter_trace = trace
//             .iter()
//             .filter(|row| matches!(row.instruction.opcode, Operator::MatMul))
//             .collect::<Vec<_>>();
//         for step in filter_trace.iter() {
//             let step_inputs = step
//                 .layer_state
//                 .input_vals
//                 .as_ref()
//                 .expect("input values should be present");
//             let a = step_inputs[0].clone().pad();
//             let b = step_inputs[1].clone().pad();
//             // Get new padded m & n values
//             let m = a.m();
//             // b is implicitly transposed
//             let n = b.m();
//             let precompile = MatMultSumcheck::<Fr>::new(&a, &b, &mut transcript);
//             precompiles.push(precompile);

//             // Update verifier transcript
//             let _log_m = m.log_2();
//             let _log_n = n.log_2();
//             let _rx: Fr = vtranscript.challenge_scalar(); // HACK: Allow verifier to know matrix dimensions.
//             let _ry: Fr = vtranscript.challenge_scalar();
//         }

//         // prover
//         let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
//             precompiles
//                 .iter_mut()
//                 .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
//                 .collect();
//         let (proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut transcript);

//         // verifier
//         let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> = precompiles
//             .iter_mut()
//             .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
//             .collect();
//         let _ = BatchedSumcheck::verify(&proof, trait_objects, &mut vtranscript).unwrap();
//     }

//     #[test]
//     fn test_matmult() {
//         let mut rng = test_rng();
//         let m = 100;
//         let n = 200;
//         let k = 300;
//         let a = QuantizedLiteTensor::random(&mut rng, m, k).pad();
//         let b = QuantizedLiteTensor::random(&mut rng, n, k).pad();
//         let m = a.m();
//         // b is implicitly transposed
//         let n = b.m();

//         let mut transcript = KeccakTranscript::new(b"test");
//         let mut precompile = MatMultSumcheck::<Fr>::new(&a, &b, &mut transcript);
//         let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut precompile], &mut transcript);
//         let mut vtranscript = KeccakTranscript::new(b"test");
//         let log_m = m.log_2();
//         let log_n = n.log_2();
//         let _rx: Vec<Fr> = vtranscript.challenge_scalar_powers(log_m);
//         let _ry: Vec<Fr> = vtranscript.challenge_scalar_powers(log_n);
//         let _ = BatchedSumcheck::verify(&proof, vec![&mut precompile], &mut vtranscript).unwrap();
//     }

//     #[test]
//     fn test_perceptron() {
//         let mut rng = test_rng();
//         let input = random_floatvec(&mut rng, 10);
//         let program = ONNXProgram::new("onnx/perceptron.onnx", &input);
//         let (trace, _io) = jolt_onnx::tracer::trace(&program.model, &program.input);
//         let (a, b) = {
//             let filter_trace = trace
//                 .iter()
//                 .filter(|row| matches!(row.instruction.opcode, Operator::MatMul))
//                 .collect::<Vec<_>>();
//             let step = filter_trace
//                 .first()
//                 .expect("execution trace should have at least one row");
//             let step_inputs = step
//                 .layer_state
//                 .input_vals
//                 .as_ref()
//                 .expect("input values should be present");
//             (step_inputs[0].clone(), step_inputs[1].clone())
//         };
//         let a = a.pad();
//         let b = b.pad();
//         // Get new padded m & n values
//         let m = a.m();
//         // b is implicitly transposed
//         let n = b.m();

//         let mut transcript = KeccakTranscript::new(b"test");
//         let mut precompile = MatMultSumcheck::<Fr>::new(&a, &b, &mut transcript);
//         let (proof, _rsc) = BatchedSumcheck::prove(vec![&mut precompile], &mut transcript);
//         let mut vtranscript = KeccakTranscript::new(b"test");
//         let log_m = m.log_2();
//         let log_n = n.log_2();
//         let _rx: Vec<Fr> = vtranscript.challenge_scalar_powers(log_m);
//         let _ry: Vec<Fr> = vtranscript.challenge_scalar_powers(log_n);
//         let _ = BatchedSumcheck::verify(&proof, vec![&mut precompile], &mut vtranscript).unwrap();
//     }

//     #[test]
//     fn test_matmult_batch() {
//         let execution_trace_length = 20;
//         let mut rng = test_rng();
//         let mut transcript = KeccakTranscript::new(b"test");
//         let mut vtranscript = KeccakTranscript::new(b"test");

//         let mut precompiles = Vec::with_capacity(execution_trace_length);
//         for _ in 0..execution_trace_length {
//             let m = rng.gen_range(1..=100);
//             let n = rng.gen_range(1..=100);
//             let k = rng.gen_range(1..=100);
//             let a = QuantizedLiteTensor::random(&mut rng, m, k).pad();
//             let b = QuantizedLiteTensor::random(&mut rng, n, k).pad();
//             // Get new padded m & n values
//             let m = a.m();
//             // b is implicitly transposed
//             let n = b.m();
//             let precompile = MatMultSumcheck::<Fr>::new(&a, &b, &mut transcript);
//             precompiles.push(precompile);

//             // Update verifier transcript
//             let log_m = m.log_2();
//             let log_n = n.log_2();
//             let _rx: Vec<Fr> = vtranscript.challenge_scalar_powers(log_m);
//             let _ry: Vec<Fr> = vtranscript.challenge_scalar_powers(log_n);
//         }

//         // prover
//         let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
//             precompiles
//                 .iter_mut()
//                 .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
//                 .collect();
//         let (proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut transcript);

//         // verifier
//         let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> = precompiles
//             .iter_mut()
//             .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
//             .collect();
//         let _ = BatchedSumcheck::verify(&proof, trait_objects, &mut vtranscript).unwrap();
//     }
// }
