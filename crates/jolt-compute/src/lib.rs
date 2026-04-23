//! Backend-agnostic compute device abstraction for the Jolt zkVM.
//!
//! **This crate defines [`ComputeBackend`] and its supporting buffer /
//! reduce-input types. Only that.** No witness data pipeline, no protocol
//! types, no runtime state.
//!
//! # Surface
//!
//! - [`ComputeBackend`] — the backend trait. Typed buffer management,
//!   kernel compilation (`KernelSpec` → `CompiledKernel`), and the unified
//!   [`ComputeBackend::reduce`] entry point.
//! - [`DeviceBuffer`] / [`Buf`] — the enum of buffer kinds a backend holds
//!   (field, u64, compact-i128).
//! - [`ReduceInputs`] — borrow view of runtime state (buffers, outer-eq
//!   tables, per-instance claims, compiled kernels) passed to `reduce`.
//! - [`HandleId`] / [`HandleShape`] — opaque state handles for
//!   per-sumcheck-instance resources (eq tables, scratch).
//!
//! # What lives elsewhere
//!
//! - `BufferProvider` / `LookupTraceData` (witness data pipeline) → `jolt-compiler`.
//! - `eval_scalar_expr` / `ScalarExpr` IR + interpreter → `jolt-compiler`.
//! - Runtime op walker, PCS orchestration, `prove()` → `jolt-zkvm`.
//!
//! # Concrete backends
//!
//! Implementations live in separate crates:
//! - `jolt-cpu` — CPU backend with Rayon parallelism.
//! - `jolt-metal` — Apple Metal GPU backend.
//!
//! Additional GPU backends (CUDA, WebGPU) follow the same pattern.

pub mod linker;
mod traits;

pub use linker::{link, Executable};
pub use traits::{
    BindingOrder, Buf, ComputeBackend, DeviceBuffer, HandleId, HandleShape, ReduceInputs, Scalar,
};
