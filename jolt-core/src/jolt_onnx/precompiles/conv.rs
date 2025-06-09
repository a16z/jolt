//! This module provides a sum-check precompile for verifying the execution of the convolution operator.

pub mod computation {
    use serde::{Deserialize, Serialize};

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
    fn conv2d_simple(input: &Tensor, kernel: &Tensor) -> Tensor {
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
        let mut output = Tensor {
            data: vec![0; h_out * w_out],
            shape: vec![1, 1, h_out, w_out],
        };

        // Loop over each output pixel position (i, j)
        for i in 0..h_out {
            for j in 0..w_out {
                let mut acc = 0i32;

                // At this point, we're computing Y(i, j)

                // Loop over each kernel position (m, n)
                for m in 0..k_h {
                    for n in 0..k_w {
                        // X(i + m, j + n) from the input
                        let x = input.get4d(0, 0, i + m, j + n) as i32;

                        // K(m, n) from the kernel
                        let k = kernel.get4d(0, 0, m, n) as i32;

                        // Multiply and accumulate: X(i + m, j + n) · K(m, n)
                        acc += x * k;
                    }
                }

                // Store the result of Y(i, j), clamped to i8 range
                output.set4d(0, 0, i, j, acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8);
            }
        }
        output
    }

    fn conv2d_sample_code(input: &Tensor, weight: &Tensor) -> Tensor {
        // - n: batch size
        // - c_in: input channels
        // - img_h: input height
        // - img_w: input width
        let [n, c_in, img_h, img_w] = input.shape[..] else {
            panic!("Expected input shape NCHW");
        };

        // - c_out: output channels
        // - c_in: input channels (should match input)
        // - k_h: kernel height
        // - k_w: kernel width
        let [c_out, _c_in, k_h, k_w] = weight.shape[..] else {
            panic!("Expected weight shape [C_out, C_in, kH, kW]");
        };
        let [n, c_in, h_in, w_in] = input.shape[..] else {
            panic!("Expected input shape NCHW");
        };
        let [c_out, _, k_h, k_w] = weight.shape[..] else {
            panic!("Expected weight shape [C_out, C_in, kH, kW]");
        };

        let h_out = h_in - k_h + 1;
        let w_out = w_in - k_w + 1;

        let mut output = Tensor {
            data: vec![0; n * c_out * h_out * w_out],
            shape: vec![n, c_out, h_out, w_out],
        };

        // batch size
        for b in 0..n {
            // output channels
            for co in 0..c_out {
                // output height
                for ho in 0..h_out {
                    // output width
                    for wo in 0..w_out {
                        let mut acc: i32 = 0;
                        // input channels
                        for ci in 0..c_in {
                            // kernel height
                            for kh in 0..k_h {
                                // kernel width
                                for kw in 0..k_w {
                                    let ih = ho + kh;
                                    let iw = wo + kw;
                                    let x = input.get4d(b, ci, ih, iw) as i32;
                                    let w = weight.get4d(co, ci, kh, kw) as i32;
                                    acc += x * w;
                                }
                            }
                        }
                        // Clamping the result to i8 range
                        output.set4d(
                            b,
                            co,
                            ho,
                            wo,
                            acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8,
                        );
                    }
                }
            }
        }

        output
    }

    #[derive(Clone, Serialize, Deserialize, Debug)]
    pub struct Tensor {
        pub data: Vec<i8>,
        pub shape: Vec<usize>, // Always [N, C, H, W]
    }

    impl Tensor {
        /// Returns the value at position (n, c, h, w)
        pub fn get4d(&self, n: usize, c: usize, h: usize, w: usize) -> i8 {
            // Destructure the shape for clarity: [batch, channels, height, width]
            let [n_dim, c_dim, h_dim, w_dim] = self.shape[..] else {
                panic!("Shape must be 4D");
            };

            // Compute the flat 1D index for a 4D tensor stored in row-major (NCHW) order:
            //
            // Index = n * C * H * W        → skip to the right batch
            //       + c * H * W            → skip to the right channel
            //       + h * W                → skip to the correct row
            //       + w                    → pick the exact element
            //
            // Note: this matches NCHW layout used in ONNX/PyTorch
            let index = n * c_dim * h_dim * w_dim + c * h_dim * w_dim + h * w_dim + w;

            self.data[index]
        }

        /// Sets the value at position (n, c, h, w)
        pub fn set4d(&mut self, n: usize, c: usize, h: usize, w: usize, value: i8) {
            let [n_dim, c_dim, h_dim, w_dim] = self.shape[..] else {
                panic!("Shape must be 4D");
            };

            let index = n * c_dim * h_dim * w_dim + c * h_dim * w_dim + h * w_dim + w;

            self.data[index] = value;
        }

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
        use super::*;

        #[test]
        fn test_conv2d() {
            // Input: shape [1, 1, 3, 3]
            let input = Tensor {
                data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
                shape: vec![1, 1, 3, 3],
            };

            // Kernel: shape [1, 1, 2, 2]
            let weight = Tensor {
                data: vec![1, 0, 0, -1],
                shape: vec![1, 1, 2, 2],
            };

            let output = conv2d_sample_code(&input, &weight);

            println!("Output shape: {:?}", output.shape);
            println!("Output data:");

            for h in 0..output.shape[2] {
                for w in 0..output.shape[3] {
                    print!("{:>4}", output.get4d(0, 0, h, w));
                }
                println!();
            }
        }
    }
}
