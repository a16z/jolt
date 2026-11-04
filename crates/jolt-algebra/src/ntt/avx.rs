//! x86 runtime dispatch helpers for CRT NTT SIMD kernels.
//!
//! `AKITA_SCALAR_NTT=1` forces the scalar fallback for all CRT NTT SIMD.
//! Pointwise and transform kernels default to AVX2; `AKITA_AVX512_NTT=1` opts
//! into the 512-bit instantiations on hosts with AVX-512F/DQ/BW.

mod field_limbs;
mod lanes;
mod montgomery;
mod pointwise;
mod runtime;
#[cfg(test)]
mod tests;
mod transform;
mod twiddles;

pub use runtime::{avx_ntt_mode, use_avx2_transform_ntt, AvxNttMode};

pub(crate) use field_limbs::field_residues_i32;

pub use pointwise::{add_reduce_i16, add_reduce_i32, add_reduce_i32_avx512};
pub(crate) use pointwise::{
    pointwise_dot_acc_i32, pointwise_mul_acc_i16, pointwise_mul_acc_i32,
    pointwise_mul_acc_i32_avx512,
};
#[cfg(test)]
use runtime::{select_avx_ntt_mode, AvxCpuFeatures};
pub(crate) use transform::{
    forward_ntt_centered_i16_i16, forward_ntt_centered_i16_i32, forward_ntt_cyclic_i16,
    forward_ntt_cyclic_i32, forward_ntt_cyclic_i8_i16, forward_ntt_cyclic_i8_i32, forward_ntt_i16,
    forward_ntt_i32, forward_ntt_i8_i16, forward_ntt_i8_i32, inverse_ntt_cyclic_i16,
    inverse_ntt_cyclic_i32, inverse_ntt_i16, inverse_ntt_i32,
};
pub(crate) use twiddles::MontQuotients;
