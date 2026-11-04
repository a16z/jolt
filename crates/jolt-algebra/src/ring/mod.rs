//! Cyclotomic ring types.

pub mod cyclotomic;

pub use cyclotomic::{
    balanced_decompose_coefficients_pow2_i8_into, CyclotomicRing, NegacyclicShiftWindows,
    WideCyclotomicRing,
};
