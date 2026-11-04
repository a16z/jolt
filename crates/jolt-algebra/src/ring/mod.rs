//! Cyclotomic ring types, their CRT-NTT representation, and evaluation
//! kernels.

pub mod crt_ntt_repr;
pub mod cyclotomic;
pub mod eval;
mod residue;

pub use crt_ntt_repr::{
    cyclic_ntt_with_i16_tail_to_ring, mat_vec_i16_with_tail, ntt_with_i16_tail_to_ring,
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
    I16TailParams,
};
pub use cyclotomic::{
    balanced_decompose_coefficients_pow2_i8_into, CyclotomicRing, NegacyclicShiftWindows,
    WideCyclotomicRing,
};
pub use eval::{
    eval_ring_at, evaluate_power_sequence_mle, scalar_powers, scalar_powers_with_stride,
};
pub use residue::{residue_kernel, terminal_residue_kernel, ResidueKernelPoint};
