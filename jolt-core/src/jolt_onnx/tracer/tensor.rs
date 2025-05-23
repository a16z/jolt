use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents a [`tract_onnx`] tensor for this codebase
pub struct QuantizedLiteTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i32>,
    pub scale: f64,
}

pub fn dequantize(data: &[i32], scale: f64) -> Vec<f32> {
    data.iter().map(|&x| (x as f64 / scale) as f32).collect()
}

impl QuantizedLiteTensor {
    // /// Dequantize the tensor
    // pub fn dequantize(&self) -> LiteTensor {
    //     let tensor_data = &self.data;
    //     let tensor_shape = self.shape.clone();
    //     let scale = self.scale;
    //     let dequantized_data: Vec<f32> = tensor_data.iter().map(|&x| (x as f32) * scale).collect();
    //     LiteTensor {
    //         shape: tensor_shape,
    //         data: dequantized_data,
    //     }
    // }
    /// Dequantize the tensor
    pub fn dequantize(&self) -> LiteTensor {
        let tensor_data = &self.data;
        let tensor_shape = self.shape.clone();
        let scale = self.scale;
        let dequantized_data: Vec<f32> = dequantize(tensor_data, scale);
        LiteTensor {
            shape: tensor_shape,
            data: dequantized_data,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents a [`tract_onnx`] tensor for this codebase
pub struct LiteTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

// /// Quantize a slice of f32 data to i8
// pub fn quantize(data: &[f32]) -> (Vec<i8>, f32) {
//     let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
//     let scale = 127.0 / max_abs;
//     let quantized_data: Vec<i8> = data
//         .iter()
//         .map(|&x| {
//             let scaled = (x / scale).round(); // bring float into [-127, 127]
//             scaled.clamp(-128.0, 127.0) as i8 // clamp in case of overflow, cast to i8
//         })
//         .collect();
//     (quantized_data, scale)
// }

/// Quantize a slice of f32 data to i32 using full dynamic range
pub fn quantize(data: &[f32]) -> (Vec<i32>, f64) {
    // 1. Find the max absolute value to determine dynamic range
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

    // 2. Compute the scale: map [-max_abs, max_abs] to [-2^31, 2^31 - 1]
    let scale = (2_i64.pow(31) as f64) / max_abs as f64;

    // 3. Quantize each float to i32
    let quantized_data: Vec<i32> = data
        .iter()
        .map(|&x| (x as f64 * scale).round() as i32)
        .collect();

    (quantized_data, scale)
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
        let (quantized_data, scale) = quantize(tensor_data);
        QuantizedLiteTensor {
            shape: tensor_shape,
            data: quantized_data,
            scale,
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
