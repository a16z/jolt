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

mod traits;

pub use traits::{BindingOrder, ComputeBackend, Scalar};
