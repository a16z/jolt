//! Backend-agnostic compute device abstraction for the Jolt zkVM.
//!
//! This crate defines the [`ComputeBackend`] trait — a protocol-agnostic
//! interface over typed buffer management and parallel primitives (pairwise
//! interpolation, composition-reduce, product tables). All methods are named
//! for what they compute, not what protocol uses them.
//!
//! The [`CpuBackend`] implementation uses `Vec<T>` buffers with Rayon
//! parallelism. After monomorphization every trait method compiles to a
//! direct function call — identical codegen to hand-written Rayon code.
//!
//! GPU backends (Metal, CUDA, WebGPU) live in separate crates and implement
//! the same trait with device memory buffers and compiled shader kernels.

mod any_buffer;
mod cpu;
mod traits;

pub use any_buffer::AnyBuffer;
pub use cpu::{CpuBackend, CpuKernel};
pub use traits::{ComputeBackend, Scalar};
