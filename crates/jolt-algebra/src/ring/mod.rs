//! Cyclotomic ring types and their evaluation kernels.

pub mod cyclotomic;
pub mod eval;
mod residue;

pub use cyclotomic::{
    balanced_decompose_coefficients_pow2_i8_into, CyclotomicRing, NegacyclicShiftWindows,
    WideCyclotomicRing,
};
pub use eval::{
    eval_ring_at, evaluate_power_sequence_mle, scalar_powers, scalar_powers_with_stride,
};
pub use residue::{residue_kernel, terminal_residue_kernel, ResidueKernelPoint};
