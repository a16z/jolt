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
        let w_out = w_in - k_w + 1; // TODO: Padding?
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
            X: X_r,
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

    use crate::{
        jolt_onnx::precompiles::{
            conv::computation::Tensor,
            conv1d::{Conv1DPrecompile, Conv1DProverState},
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };

    #[test]
    fn test_conv2d() {
        // Input: shape [1, 1, 8]
        let input = Tensor {
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            shape: vec![1, 1, 8],
        };

        // Kernel: shape [1, 1, 2]
        let weight = Tensor {
            data: vec![9, 10],
            shape: vec![1, 1, 2],
        };
        let mut ptranscript = KeccakTranscript::new(b"test");
        let precompile = Conv1DPrecompile::new(input.clone(), weight.clone());
        let prover = Conv1DProverState::<Fr>::initialize(&precompile, &mut ptranscript);
    }
}
