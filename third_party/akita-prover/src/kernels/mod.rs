//! Low-level NTT and digit-decomposition kernels.

pub mod crt_ntt;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod decompose_fold_avx;
#[cfg(target_arch = "aarch64")]
pub(crate) mod decompose_fold_neon;
pub mod linear;

pub use crt_ntt::{build_ntt_slot, select_crt_ntt_params, NttSlotCache, ProtocolCrtNttParams};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use decompose_fold_avx as avx_decompose_fold;
#[cfg(target_arch = "aarch64")]
pub(crate) use decompose_fold_neon as neon_decompose_fold;
