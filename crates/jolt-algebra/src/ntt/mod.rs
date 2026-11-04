//! NTT-friendly small-prime arithmetic and CRT helpers.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx;
#[cfg(target_arch = "aarch64")]
mod batched_four_point_policy;
pub mod butterfly;
pub mod crt;
mod digit_validation;
pub(crate) mod field_limbs;
pub mod ifma52;
pub(crate) mod montgomery;
#[cfg(target_arch = "aarch64")]
pub mod neon;
mod plan;
pub mod prime;
pub mod tables;

#[cfg(target_arch = "aarch64")]
pub(crate) use batched_four_point_policy::batched_four_point_eligible;
pub use butterfly::NttTwiddles;
pub use crt::{CrtCapacity, GarnerData, LimbQ, RADIX_BITS};
pub use digit_validation::i16_values_in_balanced_range;
pub use plan::NttKernelPlan;
pub use prime::{MontCoeff, NttPrime, PrimeWidth, I32_LAZY_DOT_BATCH};
use std::sync::OnceLock;

/// Whether `AKITA_SCALAR_NTT=1` forces the scalar CRT NTT kernels on every
/// SIMD backend. Read once; the NEON, AVX and IFMA52 dispatchers all ask here.
pub(crate) fn scalar_ntt_forced() -> bool {
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| std::env::var("AKITA_SCALAR_NTT").is_ok_and(|value| value == "1"))
}
