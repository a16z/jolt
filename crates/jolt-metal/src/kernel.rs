use std::marker::PhantomData;
use std::sync::Arc;

use jolt_field::Field;

/// Shared, immutable pipeline states compiled from a single MSL source.
///
/// Cached by `MetalBackend` and shared across `MetalKernel` instances that
/// use the same kernel shape. Pipeline compilation is the expensive step
/// (minutes for large kernels), so caching here makes all subsequent
/// `compile_kernel_with_challenges` calls near-free.
pub(crate) struct CachedPipelines {
    pub pipeline_l2h: metal::ComputePipelineState,
    pub pipeline_h2l: metal::ComputePipelineState,
    pub pipeline_tensor: metal::ComputePipelineState,
    pub pipeline_l2h_unw: metal::ComputePipelineState,
    pub pipeline_h2l_unw: metal::ComputePipelineState,
    pub num_evals: usize,
    pub num_inputs: usize,
    /// Whether the kernel signature includes a `device const Fr* challenges`
    /// buffer between weights and partials. Only true for Custom kernels
    /// whose expression contains `Var::Challenge` nodes.
    pub has_challenges: bool,
}

// SAFETY: MTLComputePipelineState is immutable after creation and safe
// to share across threads and command encoders.
unsafe impl Send for CachedPipelines {}
// SAFETY: See above — pipeline states are immutable and thread-safe.
unsafe impl Sync for CachedPipelines {}

/// Compiled Metal compute pipelines for a specific kernel shape, plus
/// optional per-instance challenge values for Custom kernels.
///
/// Pipeline states are shared via `Arc<CachedPipelines>` so that multiple
/// stages using the same kernel shape pay zero recompilation cost. Only
/// the challenge buffer differs between instances.
pub struct MetalKernel<F: Field> {
    pub(crate) pipelines: Arc<CachedPipelines>,
    /// Runtime challenge buffer for Custom kernels. `None` for ProductSum,
    /// EqProduct, HammingBooleanity (which have no challenge variables).
    pub(crate) challenges_buf: Option<metal::Buffer>,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: Field> MetalKernel<F> {
    #[inline]
    pub(crate) fn num_evals(&self) -> usize {
        self.pipelines.num_evals
    }

    #[inline]
    pub(crate) fn num_inputs(&self) -> usize {
        self.pipelines.num_inputs
    }

    /// Returns the challenges buffer to bind, only if the pipeline expects it.
    ///
    /// # Panics (debug)
    ///
    /// Panics in debug builds if the pipeline expects challenges but no buffer
    /// was provided via `compile_kernel_with_challenges`.
    #[inline]
    pub(crate) fn active_challenges_buf(&self) -> Option<&metal::Buffer> {
        if self.pipelines.has_challenges {
            debug_assert!(
                self.challenges_buf.is_some(),
                "kernel expects challenges buffer but none was provided"
            );
            self.challenges_buf.as_ref()
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn pipeline_l2h(&self) -> &metal::ComputePipelineState {
        &self.pipelines.pipeline_l2h
    }

    #[inline]
    pub(crate) fn pipeline_h2l(&self) -> &metal::ComputePipelineState {
        &self.pipelines.pipeline_h2l
    }

    #[inline]
    pub(crate) fn pipeline_tensor(&self) -> &metal::ComputePipelineState {
        &self.pipelines.pipeline_tensor
    }

    #[inline]
    pub(crate) fn pipeline_l2h_unw(&self) -> &metal::ComputePipelineState {
        &self.pipelines.pipeline_l2h_unw
    }

    #[inline]
    pub(crate) fn pipeline_h2l_unw(&self) -> &metal::ComputePipelineState {
        &self.pipelines.pipeline_h2l_unw
    }
}

// SAFETY: MTLComputePipelineState and MTLBuffer are immutable after creation
// and safe to share across threads and command encoders. Arc is inherently
// Send+Sync for Send+Sync contents.
unsafe impl<F: Field> Send for MetalKernel<F> {}
// SAFETY: See above — pipeline states and buffers are immutable and thread-safe.
unsafe impl<F: Field> Sync for MetalKernel<F> {}
