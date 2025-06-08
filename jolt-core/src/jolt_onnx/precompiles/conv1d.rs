use crate::{
    field::JoltField,
    jolt_onnx::precompiles::{conv::computation::Tensor, conv1d::computation::conv1d_simple},
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// # Note: We assume tensors are appropriately padded here
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Conv1DPrecompile {
    image: Tensor,
    kernel: Tensor,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct Conv1DProverState<F>
where
    F: JoltField,
{
    /// X(r, m) evaluations over the boolean hypercube
    pub X: DensePolynomial<F>,
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
        //
        // Extract input spatial dimension
        let w_in = input.image.shape[2];
        // Extract kernel spatial dimension
        let k_w = input.kernel.shape[2];
        // Compute output spatial dimension
        let w_out = w_in - k_w + 1; // TODO: Padding?
        let mut X_r = vec![F::zero(); k_w];
        let r: Vec<F> = transcript.challenge_scalar_powers(w_out.log_2());
        let eq_r_evals = EqPolynomial::evals(&r);
        for j in 0..w_out {
            for k in 0..k_w {
                let x = input.image.data[j + k];
                X_r[k] += eq_r_evals[j] * F::from_i64(x as i64);
            }
        }
        let X_r = DensePolynomial::new(X_r);
        let k = DensePolynomial::new(
            input
                .kernel
                .data
                .iter()
                .map(|d| F::from_i64(*d as i64))
                .collect_vec(),
        );
        let input_claim = {
            let (y, _) = conv1d_simple(&input.image, &input.kernel);
            let y_poly =
                DensePolynomial::new(y.iter().map(|&x| F::from_i64(x as i64)).collect_vec());
            y_poly.evaluate(&r)
        };
        transcript.append_scalar(&input_claim);
        Self {
            X: X_r,
            k,
            input_claim,
            num_rounds: k_w.log_2(),
        }
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
        // let mut output = Tensor {
        //     shape: vec![1, 1, w_out],
        //     data: ,
        // };
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
