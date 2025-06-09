use crate::{
    field::JoltField,
    jolt_onnx::precompiles::{
        conv::computation::Tensor, conv1d::computation::conv1d_simple,
        sumcheck_engine::BatchableSumcheckInstance,
    },
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub type Conv1DPrecompileDims = (usize, usize);

/// # Note: We assume tensors are appropriately padded here
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Conv1DPrecompile {
    image: Tensor,
    kernel: Tensor,
}

impl Conv1DPrecompile {
    /// Create a new instance of [`Conv1DPrecompile`].
    pub fn new(image: Tensor, kernel: Tensor) -> Self {
        Self { image, kernel }
    }

    /// # Returns
    /// - `w_in`: Input spatial dimension
    /// - `k_w`: Kernel spatial dimension
    /// - `w_out`: Output spatial dimension
    fn dims(&self) -> (usize, usize, usize) {
        // Extract input spatial dimension
        let w_in = self.image.shape[2];
        // Extract kernel spatial dimension
        let k_w = self.kernel.shape[2];
        // Compute output spatial dimension
        let w_out = w_in - k_w + 1;
        (w_in, k_w, w_out)
    }

    fn y_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        let (mut y, _) = conv1d_simple(&self.image, &self.kernel);
        let new_y_len = y.len().next_power_of_two();
        // Pad y to the next power of two
        y.resize(new_y_len, 0);
        DensePolynomial::new(y.iter().map(|&x| F::from_i64(x as i64)).collect_vec())
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv1DProverState<F>
where
    F: JoltField,
{
    /// X(r, m) evaluations over the boolean hypercube
    pub x: DensePolynomial<F>,
    /// k multilinear polynomial
    pub k: DensePolynomial<F>,
    /// Evaluation at point r for multilinear polynomial Y
    pub input_claim: F,
    /// Number of rounds in the sum-check precompile
    pub num_rounds: usize,
}

impl<F> Conv1DProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`Conv1DProverState`].
    /// We compute the evaluations of the polynomials X(r, m) over the boolean hypercube,
    /// and also compute the input claim Y(r) = A(rx, k) * B(ry, k).
    ///
    /// These X(r, m) evaluations & k polynomial serve as the witness for the matrix multiplication precompile.
    pub fn initialize<ProofTranscript>(
        input: &Conv1DPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        // --- Calculate the evaluation's for X(r, m) over the boolean hypercube ---
        let (_w_in, k_w, w_out) = input.dims();
        let r: Vec<F> = transcript.challenge_scalar_powers(w_out.next_power_of_two().log_2());
        let X_r = Self::X_bounded(&input.image, &r, w_out, k_w);
        let k = Self::kernel_polynomial(&input.kernel);
        let input_claim = Self::input_claim(input, &r);
        transcript.append_scalar(&input_claim);
        #[cfg(test)]
        {
            let sum: F = X_r.Z.iter().zip_eq(k.Z.iter()).map(|(x, k)| *x * k).sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            x: X_r,
            k,
            input_claim,
            num_rounds: k_w.log_2(),
        }
    }

    fn input_claim(input: &Conv1DPrecompile, r: &[F]) -> F {
        input.y_poly().evaluate(r)
    }

    fn X_bounded(X: &Tensor, r: &[F], w_out: usize, k_w: usize) -> DensePolynomial<F> {
        let mut X_r = vec![F::zero(); k_w];
        let eq_r_evals = EqPolynomial::evals(r);
        for j in 0..w_out {
            for k in 0..k_w {
                let x = X.data[j + k];
                X_r[k] += eq_r_evals[j] * F::from_i64(x as i64);
            }
        }
        DensePolynomial::new(X_r)
    }

    fn kernel_polynomial(kernel: &Tensor) -> DensePolynomial<F> {
        DensePolynomial::new(
            kernel
                .data
                .iter()
                .map(|d| F::from_i64(*d as i64))
                .collect_vec(),
        )
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv1DVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> Conv1DVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`Conv1DVerifierState`].
    /// # Note: we mainly update the state by computing the necessary challenges used in the sum-check conv protocol.
    ///         We also append the input claim to the transcript.
    pub fn initialize<ProofTranscript>(
        k_w: usize,
        w_out: usize,
        input_claim: F,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let num_rounds = k_w.log_2();
        let _r: Vec<F> = transcript.challenge_scalar_powers(w_out.next_power_of_two().log_2());
        transcript.append_scalar(&input_claim);
        Self {
            num_rounds,
            input_claim,
        }
    }
}

/// The final claims for the conv sum-check precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv1DClaims<F>
where
    F: JoltField,
{
    x: F,
    k: F,
}

/// Batchable sum-check instance for conv precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv1DSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<Conv1DProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<Conv1DVerifierState<F>>,
    /// Holds the final claims for the conv sum-check precompile.
    pub claims: Option<Conv1DClaims<F>>,
}

impl<F> Conv1DSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`Conv1DSumcheck`]
    pub fn new(
        prover_state: Option<Conv1DProverState<F>>,
        verifier_state: Option<Conv1DVerifierState<F>>,
        claims: Option<Conv1DClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for Conv1DSumcheck<F>
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
        let Conv1DProverState { x, k, .. } = self.prover_state.as_ref().unwrap();
        let len = x.len() / 2;
        let univariate_poly_evals: [F; 2] = (0..len)
            .into_par_iter()
            .map(|i| {
                let poly_X_bound_point = x[i + len] + x[i + len] - x[i];
                let poly_K_bound_point = k[i + len] + k[i + len] - k[i];
                [x[i] * k[i], poly_X_bound_point * poly_K_bound_point]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let Conv1DProverState { x, k, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || x.bind_parallel(r_j, BindingOrder::HighToLow),
            || k.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let Conv1DProverState { x, k, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(Conv1DClaims { x: x[0], k: k[0] });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let Conv1DClaims { x, k } = self.claims.as_ref().unwrap();
        *x * k
    }
}

pub mod computation {
    use crate::jolt_onnx::precompiles::conv::computation::Tensor;

    /// Computes the ONNX 1D Convolution operation.
    ///
    /// The output pixel at (i) is computed as:
    ///
    ///     Y(i) = Σₖ X(i + k) · K(k)
    ///
    /// Where:
    ///   • i ∈ [0, W_out - 1]
    ///   • k ∈ [0, kW - 1]
    ///
    /// Performs 1D convolution assuming:
    ///   - Batch size = 1
    ///   - Input channels = 1
    ///   - Output channels = 1
    ///   - No padding, stride = 1, dilation = 1
    ///
    /// Therefore:
    ///   - Input shape: [1, 1, W_in]
    ///   - Kernel shape: [1, 1, kW]
    ///   - Output shape: [1, 1, W_out]
    ///
    /// The output spatial dimension is computed as:
    ///   W_out = W_in - kW + 1
    pub fn conv1d_simple(input: &Tensor, kernel: &Tensor) -> (Vec<i32>, Vec<usize>) {
        // Extract input spatial dimension
        let w_in = input.shape[2];

        // Extract kernel spatial dimension
        let k_w = kernel.shape[2];

        // Compute output spatial dimension
        let w_out = w_in - k_w + 1;

        // Create output tensor with correct shape and zero-initialized data
        let mut output = vec![0i32; w_out];

        // Perform the convolution operation
        for i in 0..w_out {
            for k in 0..k_w {
                output[i] += input.data[i + k] as i32 * kernel.data[k] as i32;
            }
        }
        let shape = vec![1, 1, w_out];
        (output, shape)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use itertools::Itertools;
    use rand_core::RngCore;

    use crate::{
        jolt_onnx::precompiles::{
            conv::computation::Tensor,
            conv1d::{
                Conv1DPrecompile, Conv1DPrecompileDims, Conv1DProverState, Conv1DSumcheck,
                Conv1DVerifierState,
            },
            sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 10;
        let mut pp: Vec<Conv1DPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let w_in = (rng.next_u32() as usize % 100 + 100).next_power_of_two();
            let k_w = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            let w_out = w_in - k_w + 1;
            pp.push((k_w, w_out));
            let image = Tensor::random(&mut rng, vec![1, 1, w_in]);
            let kernel = Tensor::random(&mut rng, vec![1, 1, k_w]);
            let precompile = Conv1DPrecompile::new(image, kernel);
            let prover_state = Conv1DProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = Conv1DSumcheck::new(Some(prover_state), None, None);
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
        for (((k_w, w_out), init_claim), final_claim) in pp
            .iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
        {
            let verifier_state =
                Conv1DVerifierState::<Fr>::initialize(*k_w, *w_out, *init_claim, &mut vtranscript);
            vsumcheck_instances.push(Conv1DSumcheck::new(
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
