//! Backend-agnostic compute device abstraction for the Jolt zkVM.
//!
//! This crate defines the [`ComputeBackend`] trait — a protocol-agnostic
//! interface over typed buffer management, kernel compilation, and parallel
//! primitives (pairwise interpolation, composition-reduce, product tables).
//! All methods are named for what they compute, not what protocol uses them.
//!
//! Concrete backends live in separate crates:
//! - `jolt-cpu` — CPU backend with Rayon parallelism
//! - `jolt-metal` — Apple Metal GPU backend
//!
//! GPU backends (CUDA, WebGPU) will follow the same pattern.

pub mod linker;
mod traits;

pub use linker::{link, Executable, FuseDebugMode, ReduceDebugMode};
pub use traits::{
    per_instance_batch_evaluate, per_instance_reference_reduce, BatchInstanceSpec, BatchReduceKind,
    BindingOrder, Buf, BufferProvider, ComputeBackend, DeviceBuffer, HandleId, HandleShape,
    LookupTraceData, ReduceInputs, Scalar,
};
