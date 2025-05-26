//! This module provides the tensor types used by the ONNX runtime in Jolt.

// TODO: Refactor duplicate code

use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents a [`tract_onnx`] tensor for this codebase
pub struct QuantizedLiteTensor {
    /// The shape of the tensor, represented as a vector of dimensions.
    pub shape: Vec<usize>,
    /// The data of the tensor, represented as a vector of 8-bit integers.
    pub data: Vec<i8>,
    /// The scale factor used for quantization.
    pub scale: f32,
    /// The zero point used for quantization, which is the value that corresponds to zero in the original data.
    pub zero_point: i8,
}

/// Dequantize a slice of i8 data using the scale and zero point
/// This function converts quantized data back to floating-point values.
pub fn dequantize(data: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
    data.iter()
        .map(|&q| scale * (q as f32 - zero_point as f32))
        .collect()
}

impl QuantizedLiteTensor {
    /// Dequantize the tensor
    pub fn dequantize(&self) -> LiteTensor {
        let dequantized_data: Vec<f32> = dequantize(&self.data, self.scale, self.zero_point);
        LiteTensor {
            shape: self.shape.clone(),
            data: dequantized_data,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents a [`tract_onnx`] tensor for this codebase
pub struct LiteTensor {
    /// The shape of the tensor, represented as a vector of dimensions.
    pub shape: Vec<usize>,
    /// The data of the tensor, represented as a vector of floating-point numbers.
    pub data: Vec<f32>,
}

/// Quantize a slice of f32 data to i8
pub fn quantize(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = 127.0 / max_abs;
    let quantized_data: Vec<i8> = data
        .iter()
        .map(|&x| {
            let scaled = (x / scale).round(); // bring float into [-127, 127]
            scaled.clamp(-128.0, 127.0) as i8 // clamp in case of overflow, cast to i8
        })
        .collect();
    (quantized_data, scale)
}

/// Quantize a slice of f32 data to i8 using asymmetric quantization
pub fn quantize_affine_i8(data: &[f32]) -> (Vec<i8>, f32, i8) {
    // Find min and max in the data
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Edge case: constant tensor
    if (max - min).abs() < 1e-8 {
        return (vec![0; data.len()], 1.0, 0); // or handle differently
    }

    // Quantization range for i8 is [-128, 127]
    let qmin = -128i32;
    let qmax = 127i32;

    // Compute scale
    let scale = (max - min) / (qmax as f32 - qmin as f32); // float -> int range
    let zero_point_f = qmin as f32 - min / scale;
    let zero_point = zero_point_f.round().clamp(qmin as f32, qmax as f32) as i8;

    // Quantize each value
    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| {
            let scaled = zero_point_f + x / scale;
            scaled.round().clamp(qmin as f32, qmax as f32) as i8
        })
        .collect();

    (quantized, scale, zero_point)
}

#[allow(dead_code)]
impl LiteTensor {
    /// Create an instance of [`QuantizedLiteTensor`] from a [`LiteTensor`]
    /// and quantize the data to 8-bit integers.
    /// The quantization is done by scaling the data to fit in the range [-127, 127].
    /// The scale is calculated as the maximum absolute value of the data divided by 127.
    pub fn quantize(&self) -> QuantizedLiteTensor {
        let tensor_data = &self.data;
        let tensor_shape = self.shape.clone();
        let (quantized_data, scale, zero_point) = quantize_affine_i8(tensor_data);
        QuantizedLiteTensor {
            shape: tensor_shape,
            data: quantized_data,
            scale,
            zero_point,
        }
    }

    fn transposed(&self, alpha: f32) -> LiteTensor {
        let mut tensor_shape = self.shape.clone();
        // Reverse the shape to get the correct dimensions for transposing
        tensor_shape.reverse();
        let tensor_data = &self.data;
        let (m, n) = (tensor_shape[0], tensor_shape[1]);
        let mut transposed_data = vec![0.0; tensor_data.len()];

        // Transpose the data matrix
        for i in 0..m {
            for j in 0..n {
                transposed_data[i * n + j] = tensor_data[j * m + i] * alpha;
            }
        }
        LiteTensor {
            shape: tensor_shape,
            data: transposed_data,
        }
    }

    /// Multiply tensor with a scalar
    fn multiply(&self, beta: f32) -> LiteTensor {
        let tensor_data = &self.data;
        let tensor_shape = self.shape.clone();
        let multiplied_data = tensor_data.iter().map(|&x| x * beta).collect::<Vec<f32>>();
        LiteTensor {
            shape: tensor_shape,
            data: multiplied_data,
        }
    }
}

impl From<Tensor> for LiteTensor {
    fn from(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let data = tensor.as_slice::<f32>().unwrap().to_vec();
        Self { shape, data }
    }
}
