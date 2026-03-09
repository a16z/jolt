//! Kernel compiler: `jolt-ir::KernelDescriptor` → MSL source → `MetalKernel`.
//!
//! Two compilation paths:
//!
//! - **ProductSum**: Generates a fully-unrolled MSL kernel with Toom-Cook
//!   evaluation specialized for the product degree D. Produces D
//!   evaluations per thread.
//!
//! - **Custom**: Walks the `jolt-ir::Expr` DAG and emits an MSL function
//!   body. Challenge values are baked as `constant` parameters. Produces
//!   `degree + 1` evaluations per thread on the standard grid.
//!
//! Both paths produce a `MetalKernel` containing a compiled
//! `MTLComputePipelineState`.

use crate::kernel::MetalKernel;
use jolt_field::Field;

/// Compile a `KernelDescriptor` into a Metal compute pipeline.
pub fn compile<F: Field>(
    _device: &metal::Device,
    _descriptor: &jolt_ir::KernelDescriptor,
) -> MetalKernel<F> {
    todo!()
}
