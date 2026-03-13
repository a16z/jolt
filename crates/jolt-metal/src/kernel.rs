use std::marker::PhantomData;

use jolt_field::Field;

/// Compiled Metal compute pipelines for a specific kernel shape.
///
/// Contains five pipeline variants compiled from a single MSL source:
/// - `pipeline_l2h`: Weighted, pairs as `(buf[2i], buf[2i+1])` (interleaved)
/// - `pipeline_h2l`: Weighted, pairs as `(buf[i], buf[i+half])` (split-half)
/// - `pipeline_tensor`: Weighted tensor-product weights, always LowToHigh
/// - `pipeline_l2h_unw`: Unweighted LowToHigh (no weight buffer, no weight mul)
/// - `pipeline_h2l_unw`: Unweighted HighToLow (no weight buffer, no weight mul)
pub struct MetalKernel<F: Field> {
    pub(crate) pipeline_l2h: metal::ComputePipelineState,
    pub(crate) pipeline_h2l: metal::ComputePipelineState,
    pub(crate) pipeline_tensor: metal::ComputePipelineState,
    pub(crate) pipeline_l2h_unw: metal::ComputePipelineState,
    pub(crate) pipeline_h2l_unw: metal::ComputePipelineState,
    pub(crate) num_evals: usize,
    pub(crate) num_inputs: usize,
    pub(crate) _marker: PhantomData<F>,
}

/// SAFETY: MTLComputePipelineState is immutable after creation and safe
/// to share across threads and command encoders.
unsafe impl<F: Field> Send for MetalKernel<F> {}
/// SAFETY: See above — pipeline states are immutable and thread-safe.
unsafe impl<F: Field> Sync for MetalKernel<F> {}
