use crate::{
    field::JoltField,
    jolt_onnx::precompiles::{
        conv::computation::Tensor, conv2d::computation::conv2d_simple,
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

// TODO: refactor duplicate code between this module and matmult.rs

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv2DProverState<F>
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

impl<F> Conv2DProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`Conv2DProverState`].
    /// We compute the evaluations of the polynomials X(r, m) over the boolean hypercube,
    /// and also compute the input claim Y(r) = A(rx, k) * B(ry, k).
    ///
    /// These X(r, m) evaluations & k polynomial serve as the witness for the matrix multiplication precompile.
    pub fn initialize<ProofTranscript>(
        input: &Conv2DPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let (_h_in, _w_in, dims) = input.dims();
        let ri: Vec<F> = transcript.challenge_scalar_powers(dims.h_out.next_power_of_two().log_2());
        let rj: Vec<F> = transcript.challenge_scalar_powers(dims.w_out.next_power_of_two().log_2());
        let X_r = Self::X_bounded(&input.image, &ri, &rj, dims);
        let k = Self::kernel_polynomial(&input.kernel);
        let input_claim = Self::input_claim(input, &ri, &rj);
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
            num_rounds: dims.k_w.log_2(), // TODO
        }
    }

    fn input_claim(input: &Conv2DPrecompile, ri: &[F], rj: &[F]) -> F {
        input.y_poly().evaluate(&[ri, rj].concat())
    }

    fn X_bounded(X: &Tensor, ri: &[F], rj: &[F], dims: Conv2DPrecompileDims) -> DensePolynomial<F> {
        let w_in = X.shape[3];
        let Conv2DPrecompileDims {
            k_h,
            k_w,
            h_out,
            w_out,
        } = dims;
        let mut X_r = vec![F::zero(); k_h * k_w];
        let eq_ri_evals = EqPolynomial::evals(ri);
        let eq_rj_evals = EqPolynomial::evals(rj);
        for i in 0..h_out {
            for j in 0..w_out {
                for m in 0..k_h {
                    for n in 0..k_w {
                        X_r[m * k_w + n] += eq_ri_evals[i]
                            * eq_rj_evals[j]
                            * F::from_i64(X.data[(i + m) * w_in + (j + n)] as i64);
                    }
                }
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

#[derive(Clone, Serialize, Deserialize, Debug, Copy)]
/// Specifies the dimensions in the Conv2D precompile to compute the challenges.
pub struct Conv2DPrecompileDims {
    /// Kernel spatial height dimension
    pub k_h: usize,
    /// Kernel spatial width dimension
    pub k_w: usize,
    /// Output spatial height dimension
    pub h_out: usize,
    /// Output spatial width dimension
    pub w_out: usize,
}

/// # Note: We assume tensors are appropriately padded here
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Conv2DPrecompile {
    image: Tensor,
    kernel: Tensor,
}

impl Conv2DPrecompile {
    /// Create a new instance of [`Conv2DPrecompile`].
    pub fn new(image: Tensor, kernel: Tensor) -> Self {
        Self { image, kernel }
    }

    /// # Returns
    /// A tuple containing:
    ///   - `h_in`: Input spatial height dimension
    ///   - `w_in`: Input spatial width dimension
    ///   - `dims`: An instance of [`Conv2DPrecompileDims`] containing the kernel and output dimensions.
    fn dims(&self) -> (usize, usize, Conv2DPrecompileDims) {
        let h_in = self.image.shape[2];
        let w_in = self.image.shape[3];
        let k_h = self.kernel.shape[2];
        let k_w = self.kernel.shape[3];
        let h_out = h_in - k_h + 1;
        let w_out = w_in - k_w + 1;
        let dims = Conv2DPrecompileDims {
            k_h,
            k_w,
            h_out,
            w_out,
        };
        (h_in, w_in, dims)
    }

    fn y_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        let (mut y, shape) = conv2d_simple(&self.image, &self.kernel);
        let h = shape[2];
        let w = shape[3];
        let new_y_len = h.next_power_of_two() * w.next_power_of_two();
        y.resize(new_y_len, 0);
        DensePolynomial::new(y.iter().map(|&x| F::from_i64(x as i64)).collect_vec())
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv2DVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> Conv2DVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`Conv2DVerifierState`].
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
pub struct Conv2DClaims<F>
where
    F: JoltField,
{
    x: F,
    k: F,
}

/// Batchable sum-check instance for conv precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv2DSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<Conv2DProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<Conv2DVerifierState<F>>,
    /// Holds the final claims for the conv sum-check precompile.
    pub claims: Option<Conv2DClaims<F>>,
}

impl<F> Conv2DSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`Conv2DSumcheck`]
    pub fn new(
        prover_state: Option<Conv2DProverState<F>>,
        verifier_state: Option<Conv2DVerifierState<F>>,
        claims: Option<Conv2DClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for Conv2DSumcheck<F>
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
        let Conv2DProverState { x, k, .. } = self.prover_state.as_ref().unwrap();
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
        let Conv2DProverState { x, k, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || x.bind_parallel(r_j, BindingOrder::HighToLow),
            || k.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let Conv2DProverState { x, k, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(Conv2DClaims { x: x[0], k: k[0] });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let Conv2DClaims { x, k } = self.claims.as_ref().unwrap();
        *x * k
    }
}

pub mod computation {
    use crate::jolt_onnx::precompiles::conv::computation::Tensor;

    /// Computes the ONNX Convolution integer operation.
    ///
    /// The output pixel at (i, j) is computed as:
    ///
    ///     Y(i, j) = Σₘₙ X(i + m, j + n) · K(m, n)
    ///
    /// Where:
    ///   • i ∈ [0, H_out - 1]
    ///   • j ∈ [0, W_out - 1]
    ///   • m ∈ [0, kH - 1]
    ///   • n ∈ [0, kW - 1]
    ///
    /// For each output pixel Y(i, j), we apply the kernel K(m, n)
    /// to the corresponding input region X(i + m, j + n).
    ///
    /// Performs 2D convolution assuming:
    ///   - Batch size = 1
    ///   - Input channels = 1
    ///   - Output channels = 1
    ///   - No padding, stride = 1, dilation = 1
    ///   
    /// Therefore:
    ///   - Input shape: [1, 1, H_in, W_in]
    ///   - Kernel shape: [1, 1, kH, kW]
    ///   - Output shape: [1, 1, H_out, W_out]
    ///
    /// The output spatial dimensions are computed as:
    ///   H_out = H_in - kH + 1
    ///   W_out = W_in - kW + 1
    ///
    /// Explanation:
    ///   The kernel has spatial size (kH x kW) and must fully fit inside the input
    ///   image to compute a valid dot product. At each output position (i, j),
    ///   the kernel is aligned with a (kH x kW) region of the input starting at (i, j).
    ///
    ///   The last valid top-left position for the kernel is at:
    ///     i = H_in - kH
    ///     j = W_in - kW
    ///
    ///   Since i and j are zero-based, we add 1 to count all valid positions:
    ///     H_out = (H_in - kH) + 1
    ///     W_out = (W_in - kW) + 1
    pub fn conv2d_simple(input: &Tensor, kernel: &Tensor) -> (Vec<i32>, Vec<usize>) {
        // Extract input spatial dimensions
        let h_in = input.shape[2];
        let w_in = input.shape[3];

        // Extract kernel spatial dimensions
        let k_h = kernel.shape[2];
        let k_w = kernel.shape[3];

        // Compute output spatial dimensions
        let h_out = h_in - k_h + 1;
        let w_out = w_in - k_w + 1;

        // Create output tensor with correct shape and zero-initialized data
        let mut output = vec![0i32; h_out * w_out];

        // Perform the convolution operation

        //  Y(i, j) = Σₘₙ X(i + m, j + n) · K(m, n)
        for i in 0..h_out {
            for j in 0..w_out {
                let mut sum = 0i32;
                for m in 0..k_h {
                    for n in 0..k_w {
                        let x_index = (i + m) * w_in + (j + n);
                        let k_index = m * k_w + n;
                        sum += input.data[x_index] as i32 * kernel.data[k_index] as i32;
                    }
                }
                output[i * w_out + j] = sum;
            }
        }
        let shape = vec![1, 1, h_out, w_out];
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
            conv2d::{Conv2DPrecompile, Conv2DProverState},
            sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };

    #[test]
    fn test_init() {
        let mut rng = test_rng();
        let mut ptranscript = KeccakTranscript::new(b"test");
        let h_in = (rng.next_u32() as usize % 100 + 100).next_power_of_two();
        let w_in = h_in;
        let k_h = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
        let k_w = k_h;
        let image = Tensor::random(&mut rng, vec![1, 1, h_in, w_in]);
        let kernel = Tensor::random(&mut rng, vec![1, 1, k_h, k_w]);
        let precompile = Conv2DPrecompile::new(image, kernel);
        let _prover_state = Conv2DProverState::<Fr>::initialize(&precompile, &mut ptranscript);
    }

    #[test]
    fn test_init_vec() {
        let mut rng = test_rng();
        let mut ptranscript = KeccakTranscript::new(b"test");
        let h_in = (rng.next_u32() as usize % 100 + 100).next_power_of_two();
        let w_in = 1;
        let k_h = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
        let k_w = 1;
        let image = Tensor::random(&mut rng, vec![1, 1, h_in, w_in]);
        let kernel = Tensor::random(&mut rng, vec![1, 1, k_h, k_w]);
        let precompile = Conv2DPrecompile::new(image, kernel);
        let _prover_state = Conv2DProverState::<Fr>::initialize(&precompile, &mut ptranscript);
    }

    // #[test]
    // fn test_random_execution_trace() {
    //     let mut rng = test_rng();
    //     let trace_length = 10;
    //     let mut pp: Vec<Conv2DPrecompileDims> = Vec::with_capacity(trace_length);
    //     let mut ptranscript = KeccakTranscript::new(b"test");
    //     let mut sumcheck_instances = Vec::with_capacity(trace_length);
    //     for _ in 0..trace_length {
    //         let w_in = (rng.next_u32() as usize % 100 + 100).next_power_of_two();
    //         let k_w = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
    //         let w_out = w_in - k_w + 1;
    //         pp.push((k_w, w_out));
    //         let image = Tensor::random(&mut rng, vec![1, 1, w_in]);
    //         let kernel = Tensor::random(&mut rng, vec![1, 1, k_w]);
    //         let precompile = Conv2DPrecompile::new(image, kernel);
    //         let prover_state = Conv2DProverState::<Fr>::initialize(&precompile, &mut ptranscript);
    //         let sumcheck_instance = Conv2DSumcheck::new(Some(prover_state), None, None);
    //         sumcheck_instances.push(sumcheck_instance);
    //     }
    //     let init_claims = sumcheck_instances
    //         .iter()
    //         .map(|p| p.prover_state.as_ref().unwrap().input_claim)
    //         .collect_vec();
    //     let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
    //         sumcheck_instances
    //             .iter_mut()
    //             .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
    //             .collect();
    //     let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut ptranscript);
    //     let final_claims = sumcheck_instances
    //         .iter()
    //         .map(|p| p.claims.as_ref().unwrap().clone())
    //         .collect_vec();
    //     let mut vtranscript = KeccakTranscript::new(b"test");
    //     let mut vsumcheck_instances = Vec::with_capacity(trace_length);
    //     for (((k_w, w_out), init_claim), final_claim) in pp
    //         .iter()
    //         .zip_eq(init_claims.iter())
    //         .zip_eq(final_claims.iter())
    //     {
    //         let verifier_state =
    //             Conv2DVerifierState::<Fr>::initialize(*k_w, *w_out, *init_claim, &mut vtranscript);
    //         vsumcheck_instances.push(Conv2DSumcheck::new(
    //             None,
    //             Some(verifier_state),
    //             Some(final_claim.clone()),
    //         ))
    //     }
    //     let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
    //         vsumcheck_instances
    //             .iter()
    //             .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
    //             .collect();
    //     let _r = BatchedSumcheck::verify(&sumcheck_proof, trait_objects, &mut vtranscript).unwrap();
    // }
}
