//! Coarse CPU kernels used by generated Bolt/Jolt Rust.
//!
//! This crate is intentionally above the primitive protocol crates and below
//! generated code. It owns the temporary coarse CPU ABI while the compiler
//! grows finer compute lowerings.

#[cfg(feature = "cuda")]
pub mod cuda;
mod dense;
mod split_eq;

#[doc(hidden)]
pub use dense::{bind_dense_evals_reuse, bind_dense_evals_reuse_serial};
#[doc(hidden)]
pub use stage2::round_poly_from_factor_slices;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use dense::bind_dense_evals_reuse_cuda;
#[doc(hidden)]
pub use split_eq::SplitEqState;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use split_eq::CudaSplitEqState;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use stage1::cuda::{CudaDenseOuterState, DenseOuterInputs};

pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6;
pub mod stage7;
pub mod trace;
