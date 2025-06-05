//! This module provides the tensor types used by the ONNX runtime in Jolt.

// TODO: figure out a better quantization strategy

use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents a quantized [`tract_onnx`] tensor for this codebase. Used in our quantized execution trace.
pub struct QuantizedTensor {
    /// The shape of the tensor, represented as a vector of dimensions.
    pub shape: Vec<usize>,
    /// The data of the tensor, represented as a vector of 8-bit integers.
    pub data: Vec<i8>,
    /// The scale factor used for quantization.
    pub scale: f32,
}

impl QuantizedTensor {
    /// Create a new instance of [`QuantizedTensor`] from the given shape, and float data.
    /// # Note:
    /// - The data is quantized to i8 using the [`quantize`] function.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        // Quantize the data to i8 and get the scale factor.
        let (data, scale) = quantize(&data);
        // Create a new QuantizedTensor with the given shape, quantized data, and scale.
        Self { shape, data, scale }
    }

    /// Dequantize the tensor back to a vector of f32 values.
    pub fn dequantized_data(&self) -> Vec<f32> {
        // Dequantize the data by multiplying each quantized value by the scale factor.
        self.data.iter().map(|&x| x as f32 * self.scale).collect()
    }

    /// Matrix multiplication of two quantized tensors.
    /// Implicitly transposes the rhs matrix.
    ///
    /// # Note:
    /// - Intermediate results are stored as i32 to prevent overflow.
    pub fn matmul_rhs_transposed(&self, other: &QuantizedTensor) -> (Vec<i32>, Vec<usize>) {
        // Ensure the inner dimensions match for matrix multiplication (B is transposed)
        assert_eq!(
            self.shape[1], other.shape[1],
            "Inner dimensions must match for matmul"
        );
        // rows in A
        let m = self.shape[0];
        // cols in A == cols in B^T
        let k = self.shape[1];
        // rows in B == output cols
        let n = other.shape[0];

        // Output shape is [M, N]
        let mut result = vec![0i32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for t in 0..k {
                    let a_val = self.data[i * k + t] as i32;
                    let b_val = other.data[j * k + t] as i32;
                    acc += a_val * b_val;
                }
                result[i * n + j] = acc;
            }
        }
        (result, vec![m, n])
    }

    /// Pads the tensors dimensions to the next power of two, and resizes the data with zeros.
    pub fn pad(&self) -> Self {
        let m = self.shape[0].next_power_of_two();
        let n = self.shape[1].next_power_of_two();
        let mut data = self.data.clone();
        data.resize(m * n, 0);
        Self {
            data,
            shape: vec![m, n],
            scale: self.scale,
        }
    }

    /// Returns the number of rows in the tensor.
    pub fn m(&self) -> usize {
        self.shape[0]
    }

    /// Returns the number of columns in the tensor.
    pub fn n(&self) -> usize {
        self.shape[1]
    }

    /// Return an randomly initialized quantized tensor with the given shape.
    pub fn random(mut rng: impl rand_core::RngCore, m: usize, n: usize) -> Self {
        // Generate random f32 data for the tensor.
        let size = m * n;
        let data: Vec<i8> = (0..size).map(|_| rng.next_u32() as i8).collect();
        Self {
            shape: vec![m, n],
            data,
            scale: 1.0, // Default scale, will be adjusted later
        }
    }
}

/// Quantize a slice of f32 data to i8
pub fn quantize(data: &[f32]) -> (Vec<i8>, f32) {
    // Get the maximum absolute value in the data to determine the scale.
    // The scale is calculated as the maximum absolute value divided by 127,
    // which allows us to fit the data into the range [-128, 127].
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = 127.0 / max_abs;

    // Quantize the data elements by dividing the floats by the scale,
    // rounding them, clamping to the range [-128, 127], and casting to i8.
    // This ensures that the quantized values fit within the i8 range.
    let quantized_data: Vec<i8> = data
        .iter()
        .map(|&x| {
            let scaled = (x / scale).round(); // bring float into [-127, 127]
            scaled.clamp(-128.0, 127.0) as i8 // clamp in case of overflow, cast to i8
        })
        .collect();
    (quantized_data, scale)
}

/// Dequantize a quantized tensor back to f32
pub fn dequantize(tensor: &QuantizedTensor) -> Vec<f32> {
    // Dequantize the data by multiplying each quantized value by the scale factor.
    tensor
        .data
        .iter()
        .map(|&x| x as f32 * tensor.scale)
        .collect()
}

impl From<Tensor> for QuantizedTensor {
    fn from(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let data = tensor.as_slice::<f32>().unwrap().to_vec();
        let (data, scale) = quantize(&data);
        Self { shape, data, scale }
    }
}
