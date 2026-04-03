use std::marker::PhantomData;
use std::sync::Arc;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compute::BindingOrder;
use jolt_field::Field;

/// Compiled pipeline state for a single kernel variant.
///
/// Cached by `MetalBackend` and shared across `MetalKernel` instances that
/// use the same kernel shape + variant. Pipeline compilation is the expensive
/// step (minutes for large kernels), so caching makes all subsequent
/// `compile` calls near-free.
pub(crate) struct CompiledPipeline {
    pub pipeline: metal::ComputePipelineState,
    pub num_evals: usize,
    pub num_inputs: usize,
    /// Whether the kernel signature includes a `device const Fr* challenges`
    /// buffer between weights and partials. Only true for Custom kernels
    /// whose expression contains `Var::Challenge` nodes.
    pub has_challenges: bool,
}

// SAFETY: MTLComputePipelineState is immutable after creation and safe
// to share across threads and command encoders.
unsafe impl Send for CompiledPipeline {}
// SAFETY: See above — pipeline states are immutable and thread-safe.
unsafe impl Sync for CompiledPipeline {}

/// Compiled Metal compute pipeline for a specific kernel shape and variant.
///
/// Pipeline state is shared via `Arc<CompiledPipeline>` so that multiple
/// stages using the same kernel shape pay zero recompilation cost. Challenge
/// values are passed at dispatch time via `reduce`, not stored here.
pub struct MetalKernel<F: Field> {
    pub(crate) compiled: Arc<CompiledPipeline>,
    pub(crate) iteration: Iteration,
    pub(crate) binding_order: BindingOrder,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: Field> MetalKernel<F> {
    #[inline]
    pub(crate) fn num_evals(&self) -> usize {
        self.compiled.num_evals
    }

    #[inline]
    pub fn pipeline(&self) -> &metal::ComputePipelineState {
        &self.compiled.pipeline
    }
}

// SAFETY: MTLComputePipelineState is immutable after creation and safe to
// share across threads and command encoders. Arc is inherently Send+Sync
// for Send+Sync contents.
unsafe impl<F: Field> Send for MetalKernel<F> {}
// SAFETY: See above — pipeline states and buffers are immutable and thread-safe.
unsafe impl<F: Field> Sync for MetalKernel<F> {}
