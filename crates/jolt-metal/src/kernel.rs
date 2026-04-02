use std::marker::PhantomData;
use std::sync::Arc;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compute::BindingOrder;
use jolt_field::Field;

/// Shared, immutable pipeline states compiled from a single MSL source.
///
/// Cached by `MetalBackend` and shared across `MetalKernel` instances that
/// use the same kernel shape. Pipeline compilation is the expensive step
/// (minutes for large kernels), so caching here makes all subsequent
/// `compile` calls near-free.
pub(crate) struct CachedPipelines {
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

/// Compiled Metal compute pipelines for a specific kernel shape.
///
/// Pipeline states are shared via `Arc<CachedPipelines>` so that multiple
/// stages using the same kernel shape pay zero recompilation cost. Challenge
/// values are passed at dispatch time via `reduce`, not stored here.
///
/// Stores [`Iteration`] and [`BindingOrder`] from the [`KernelSpec`] to
/// control dispatch strategy (dense vs tensor, pair layout) at runtime.
pub struct MetalKernel<F: Field> {
    pub(crate) pipelines: Arc<CachedPipelines>,
    pub(crate) iteration: Iteration,
    pub(crate) binding_order: BindingOrder,
    pub(crate) _marker: PhantomData<F>,
}

/// Occupancy info for a single pipeline variant.
pub struct PipelineOccupancy {
    pub name: &'static str,
    pub max_threads_per_threadgroup: u64,
    pub thread_execution_width: u64,
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

    /// Returns occupancy info for all active pipeline variants.
    pub fn occupancy(&self) -> Vec<PipelineOccupancy> {
        let p = &self.pipelines;
        vec![
            PipelineOccupancy {
                name: "tensor",
                max_threads_per_threadgroup: p.pipeline_tensor.max_total_threads_per_threadgroup(),
                thread_execution_width: p.pipeline_tensor.thread_execution_width(),
            },
            PipelineOccupancy {
                name: "l2h_unw",
                max_threads_per_threadgroup: p.pipeline_l2h_unw.max_total_threads_per_threadgroup(),
                thread_execution_width: p.pipeline_l2h_unw.thread_execution_width(),
            },
            PipelineOccupancy {
                name: "h2l_unw",
                max_threads_per_threadgroup: p.pipeline_h2l_unw.max_total_threads_per_threadgroup(),
                thread_execution_width: p.pipeline_h2l_unw.thread_execution_width(),
            },
        ]
    }
}

// SAFETY: MTLComputePipelineState is immutable after creation and safe to
// share across threads and command encoders. Arc is inherently Send+Sync
// for Send+Sync contents.
unsafe impl<F: Field> Send for MetalKernel<F> {}
// SAFETY: See above — pipeline states and buffers are immutable and thread-safe.
unsafe impl<F: Field> Sync for MetalKernel<F> {}
