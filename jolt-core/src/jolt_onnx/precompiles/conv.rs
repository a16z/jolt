//! A sum-check precompile implementation for convolution operation.
//! Used for proving correctness of the execution of the conv ONNX operator.
//! You can see it in action in [`crate::jolt_onnx::vm::precompiles`]
//!
//! # Overview:
//!   - [`ConvPrecompile`] - We specify the precompile for conv op, by defining the input (image) tensor and the kernel tensor.
//!   - [`ConvSumcheck`] - Defines the prover and verifier states that will be used to instantiate a [`super::sumcheck_engine::BatchedSumcheck`] instance.
//!     These sum-check instances are then fed into [`super::sumcheck_engine::BatchedSumcheck::prove`] and [`super::sumcheck_engine::BatchedSumcheck::verify`].
//!   - [`ConvProverState`] - Handles/Defines the prover state for the conv sum-check precompile (handles witness polynomials for sum-check prover).
//!   - [`ConvVerifierState`] - Handles/Defines the verifier state for the conv sum-check precompile.

use crate::{
    field::JoltField,
    jolt_onnx::precompiles::sumcheck_engine::BatchableSumcheckInstance,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// TODO: Padding, strides and dilations support
// TODO: refactor duplicate code between this module and matmult.rs, not doing it atm since code is subject to heavy change with padding, strides and dilations support

/// A type defining the conv precompile in the execution trace.
/// The type is used to intialize the [`ConvProverState`]
///
/// We define the conv precompile by its input tensors:
///   - `image`: The input tensor representing the image.
///   - `kernel`: The kernel tensor used for the convolution operation.
///
/// These inputs are processed and passed into as inputs to a sum-check instance
///
/// # Note: We assume tensors are appropriately padded here
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ConvPrecompile {
    image: Tensor,
    kernel: Tensor,
}

impl ConvPrecompile {
    /// Create a new instance of [`ConvPrecompile`].
    pub fn new(image: Tensor, kernel: Tensor) -> Self {
        Self { image, kernel }
    }

    /// Determine the dimensions used in the conv precompile.
    /// This function computes the input image & kernel dimensions and the output dimensions.
    ///
    /// # Returns
    ///   - A tuple containing:
    ///     - `h_in`: The height of the input/image tensor.
    ///     - `w_in`: The width of the input/image tensor.
    ///   - `dims`: An instance of [`ConvPrecompileDims`] containing the kernel and output dimensions.
    ///
    /// # Note: we "pad" the output dims to next nearest power of two
    fn dims(&self) -> (usize, usize, ConvPrecompileDims) {
        // We assume tensors are in the format [N, C, H, W]
        // where N is batch size, C is number of channels, H is height and W is width.

        // Extract input spatial dimensions
        let h_in = self.image.shape[2];
        let w_in = self.image.shape[3];

        // Extract kernel spatial dimensions
        let k_h = self.kernel.shape[2];
        let k_w = self.kernel.shape[3];

        // Compute output spatial dimensions
        // # Note: We need to evaluate the mle of the resulting matrix at a random point, thus we need to pad the output shape.
        let h_out = (h_in - k_h + 1).next_power_of_two();
        let w_out = (w_in - k_w + 1).next_power_of_two();
        (
            h_in,
            w_in,
            ConvPrecompileDims {
                k_h,
                k_w,
                h_out,
                w_out,
            },
        )
    }

    /// Returns the y polynomial in the Convolution operation
    /// Y(i, j) = Σₘₙ X(i + m, j + n) · K(m, n).
    ///
    /// Used to compute the input claim Y(r_i, r_j) for the Conv protocol.
    fn y_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        let y = self.execute_conv();
        DensePolynomial::new(y.iter().map(|&x| F::from_i64(x as i64)).collect_vec())
    }

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
    ///
    /// # Note:
    ///   - We return a 32 bit integer vector as output instead of 8 bit integers,
    ///     because how these linear operators (conv and sum-check) work is: they take the quantized inputs and accumulate the sum in 32 bits — i.e., the output of these operators will be 32 bits.
    ///     Later, this output will be fed into a requant operator to convert it to 8-bit. We’ll use a lookup to prove the requant op.
    pub fn execute_conv(&self) -> Vec<i32> {
        let (
            _,
            w_in,
            ConvPrecompileDims {
                k_h,
                k_w,
                h_out,
                w_out,
            },
        ) = self.dims();

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
                        sum += *self.image.data.get(x_index).unwrap_or(&0) as i32
                            * self.kernel.data[k_index] as i32;
                    }
                }
                output[i * w_out + j] = sum;
            }
        }
        output
    }
}

/// Container type to manage the prover state in the [`BatchableSumcheckInstance`] for the conv precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct ConvProverState<F>
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

impl<F> ConvProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`ConvProverState`].
    ///
    /// We want to apply sum-check to the log(kH) + log(kW) variate polynomial g(m, n) = X(rᵢ, rⱼ, m, n) * K(m, n)
    ///
    /// We compute the evaluations of the polynomial X(m, n, ri rj) over the boolean hypercube,
    /// and also compute the input claim Y(ri, rj) = Σₘₙ X(m, n, ri rj) * K(m, n).
    ///
    /// These X(r, m) evaluations & k polynomial serve as the witness for the conv precompile protocol.
    pub fn initialize<ProofTranscript>(
        input: &ConvPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        // Compute challenge vectors to bound the input polynomial X(i, j, m, n).
        let (_, _, dims) = input.dims();
        let ri: Vec<F> = transcript.challenge_scalar_powers(dims.h_out.log_2());
        let rj: Vec<F> = transcript.challenge_scalar_powers(dims.w_out.log_2());
        let X_r = Self::X_bounded(&input.image, &ri, &rj, dims);

        // Convert the kernel tensor into a multilinear polynomial to be fed into the sum-check protocol.
        let k = Self::kernel_polynomial(&input.kernel);

        // Compute the sum-check input claim Y(ri, rj) and send to verifier.
        let input_claim = Self::input_claim(input, &ri, &rj);
        transcript.append_scalar(&input_claim);
        #[cfg(test)]
        {
            let sum: F = X_r.Z.iter().zip_eq(k.Z.iter()).map(|(x, k)| *x * k).sum();
            assert_eq!(sum, input_claim)
        }
        let num_rounds = (dims.k_h * dims.k_w).log_2();
        Self {
            x: X_r,
            k,
            input_claim,
            num_rounds,
        }
    }

    /// Given the challenge vectors compute Y(ri, rj)
    fn input_claim(input: &ConvPrecompile, ri: &[F], rj: &[F]) -> F {
        input.y_poly().evaluate(&[ri, rj].concat())
    }

    /// Compute the boolean evaluations for the polynomial X_{ri, rj}(m, n).
    /// Used as input to the conv sum-check-precompile protocol.
    ///
    /// Bounds the input polynomial X as follows:
    ///
    ///     X(rᵢrⱼ, mn) = ∑_{uv}^{kH,kW} eq(uv, mn) · X_{mn}(rᵢrⱼ)
    ///
    /// where:
    ///
    ///     X_{mn}(rᵢrⱼ) = ∑_{p,q}^{H_out,W_out} eq(p, i) · eq(q, j) · x_{i+m, j+n}
    fn X_bounded(X: &Tensor, ri: &[F], rj: &[F], dims: ConvPrecompileDims) -> DensePolynomial<F> {
        // Used to index into the input/image tensor
        let w_in = X.shape[3];
        let ConvPrecompileDims {
            k_h,
            k_w,
            h_out,
            w_out,
        } = dims;

        // X is a log(k_H) + log(k_W) variate polynomial
        let mut X_r = vec![F::zero(); k_h * k_w];
        let eq_ri_evals = EqPolynomial::evals(ri);
        let eq_rj_evals = EqPolynomial::evals(rj);
        for i in 0..h_out {
            for j in 0..w_out {
                for m in 0..k_h {
                    for n in 0..k_w {
                        let x_index = (i + m) * w_in + (j + n);
                        X_r[m * k_w + n] += eq_ri_evals[i]
                            * eq_rj_evals[j]
                            * F::from_i64(*X.data.get(x_index).unwrap_or(&0) as i64);
                    }
                }
            }
        }
        DensePolynomial::new(X_r)
    }

    /// Convert kernel tensor into a multilinear polynomial to be accessible by the sum-check protocol.
    #[inline(always)]
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
/// Used as preprocessing material in the [`crate::jolt_onnx::vm::precompiles::PrecompileProof`]
/// Prover uses these dimensions for multiple things, such as:
///  - Determine the challenge vector lengths.
///  - Iterating over the input image and kernel tensors for bounding the input (x) polynomial to the challenges.
///  - Computing the number of rounds in the sum-check precompile.
///
/// For the verifier this specifies the lengths of the precompile challenge vectors.
///
/// # Note:
///   - Verifier does not need to know the height and width of the input image tensor
pub struct ConvPrecompileDims {
    /// Kernel spatial height dimension
    pub k_h: usize,
    /// Kernel spatial width dimension
    pub k_w: usize,
    /// Output spatial height dimension
    pub h_out: usize,
    /// Output spatial width dimension
    pub w_out: usize,
}

/// Container type to manage the verifier state in the [`BatchableSumcheckInstance`] for the conv precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct ConvVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> ConvVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`ConvVerifierState`].
    /// # Note: we mainly update the state by computing the necessary challenges used in the sum-check conv protocol.
    ///         We also append the input claim to the transcript.
    pub fn initialize<ProofTranscript>(
        dims: ConvPrecompileDims,
        input_claim: F,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let num_rounds = (dims.k_h * dims.k_w).log_2();
        let _ri: Vec<F> = transcript.challenge_scalar_powers(dims.h_out.log_2());
        let _rj: Vec<F> = transcript.challenge_scalar_powers(dims.w_out.log_2());
        transcript.append_scalar(&input_claim);
        Self {
            num_rounds,
            input_claim,
        }
    }
}

/// Store the final claims/openings to later prove the openings.
/// The final claims for the conv sum-check precompile.
///
/// Stores the evaluations of the (bounded) X and kernel polynomials at `r_sc`
/// Where:
///   - `rs_c` ∈ F^{log(k_H) + log(k_W)}
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct ConvClaims<F>
where
    F: JoltField,
{
    /// X(r_i, r_j, r_sc)
    x: F,
    /// k(r_sc)
    k: F,
}

/// Batchable sum-check instance for conv precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct ConvSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<ConvProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<ConvVerifierState<F>>,
    /// Holds the final claims for the conv sum-check precompile.
    pub claims: Option<ConvClaims<F>>,
}

impl<F> ConvSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`ConvSumcheck`]
    pub fn new(
        prover_state: Option<ConvProverState<F>>,
        verifier_state: Option<ConvVerifierState<F>>,
        claims: Option<ConvClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for ConvSumcheck<F>
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
        let ConvProverState { x, k, .. } = self.prover_state.as_ref().unwrap();
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
        let ConvProverState { x, k, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || x.bind_parallel(r_j, BindingOrder::HighToLow),
            || k.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let ConvProverState { x, k, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(ConvClaims { x: x[0], k: k[0] });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let ConvClaims { x, k } = self.claims.as_ref().unwrap();
        *x * k
    }
}

// # TODO: This is a temp tensor, when we fully furnish our ONNX runtime we will use the runtimes tensor type.

/// Represents a quantized tensor used in the ONNX execution trace.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Tensor {
    /// The quantized data of the tensor, stored as a vector of i8.
    pub data: Vec<i8>,
    /// The shape of the tensor, represented as a vector of usize.
    pub shape: Vec<usize>, // Always [N, C, H, W]
}

impl Tensor {
    /// Return an randomly initialized quantized tensor with the given shape.
    pub fn random(mut rng: impl rand_core::RngCore, shape: Vec<usize>) -> Self {
        // Generate random f32 data for the tensor.
        let size = shape.iter().product::<usize>();
        let data: Vec<i8> = (0..size).map(|_| rng.next_u32() as i8).collect();
        Self { shape, data }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::{
        jolt_onnx::precompiles::{
            conv::{
                ConvPrecompile, ConvPrecompileDims, ConvProverState, ConvSumcheck,
                ConvVerifierState,
            },
            sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
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
        let mut pp: Vec<ConvPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let h_in = (rng.next_u32() as usize % 200 + 50).next_power_of_two();
            let w_in = (rng.next_u32() as usize % 200 + 50).next_power_of_two();
            let k_h = (rng.next_u32() as usize % 20 + 1).next_power_of_two();
            let k_w = (rng.next_u32() as usize % 20 + 1).next_power_of_two();
            let image = Tensor::random(&mut rng, vec![1, 1, h_in, w_in]);
            let kernel = Tensor::random(&mut rng, vec![1, 1, k_h, k_w]);
            let precompile = ConvPrecompile::new(image, kernel);
            pp.push(precompile.dims().2);
            let prover_state = ConvProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = ConvSumcheck::new(Some(prover_state), None, None);
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
                ConvVerifierState::<Fr>::initialize(*dims, *init_claim, &mut vtranscript);
            vsumcheck_instances.push(ConvSumcheck::new(
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
