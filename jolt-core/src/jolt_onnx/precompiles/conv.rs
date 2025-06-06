//! This module provides a sum-check precompile for verifying the execution of the convolution operator.

mod computation {
    //! Computes the ONNX Convolution integer operation:
    //!
    //!     Y(i, j) = Σₘₙ X(i + m, j + n) · K(m, n)
    //!
    //! This is the operation performed by ONNX `Conv` (no kernel flipping).
    //! Padding, strides, and dilation (if any) modify the indexing behavior.

    #[derive(Debug)]
    struct Tensor {
        data: Vec<i8>,
        shape: Vec<usize>, // Always [N, C, H, W]
    }

    impl Tensor {
        fn get(&self, n: usize, c: usize, h: usize, w: usize) -> i8 {
            let [n_dim, c_dim, h_dim, w_dim] = self.shape[..] else {
                panic!("Shape must be 4D");
            };
            let index = n * c_dim * h_dim * w_dim + c * h_dim * w_dim + h * w_dim + w;
            self.data[index]
        }

        fn set(&mut self, n: usize, c: usize, h: usize, w: usize, value: i8) {
            let [n_dim, c_dim, h_dim, w_dim] = self.shape[..] else {
                panic!("Shape must be 4D");
            };
            let index = n * c_dim * h_dim * w_dim + c * h_dim * w_dim + h * w_dim + w;
            self.data[index] = value;
        }
    }

    fn conv2d(input: &Tensor, weight: &Tensor) -> Tensor {
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

        for b in 0..n {
            for co in 0..c_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc: i32 = 0;
                        for ci in 0..c_in {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = ho + kh;
                                    let iw = wo + kw;
                                    let x = input.get(b, ci, ih, iw) as i32;
                                    let w = weight.get(co, ci, kh, kw) as i32;
                                    acc += x * w;
                                }
                            }
                        }
                        // Clamping the result to i8 range
                        output.set(
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

            let output = conv2d(&input, &weight);

            println!("Output shape: {:?}", output.shape);
            println!("Output data:");

            for h in 0..output.shape[2] {
                for w in 0..output.shape[3] {
                    print!("{:>4}", output.get(0, 0, h, w));
                }
                println!();
            }
        }
    }
}
