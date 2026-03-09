use std::marker::PhantomData;

use jolt_field::Field;

/// Compiled Metal compute pipeline for a specific kernel shape.
///
/// Wraps a `MTLComputePipelineState` compiled from MSL source that was
/// generated from a `jolt-ir::KernelDescriptor`. The pipeline is
/// specialized for a particular field type and kernel shape (ProductSum
/// or Custom).
pub struct MetalKernel<F: Field> {
    pub(crate) pipeline: metal::ComputePipelineState,
    pub(crate) num_evals: usize,
    pub(crate) num_inputs: usize,
    pub(crate) _marker: PhantomData<F>,
}

/// SAFETY: MTLComputePipelineState is immutable after creation and safe
/// to share across threads and command encoders.
unsafe impl<F: Field> Send for MetalKernel<F> {}
/// SAFETY: See above — pipeline states are immutable and thread-safe.
unsafe impl<F: Field> Sync for MetalKernel<F> {}
