//! Metal reduction utilities for `pairwise_reduce` and `tensor_pairwise_reduce`.
//!
//! Implements the two-level reduction strategy:
//!
//! 1. **Per-thread**: Each thread processes one position, evaluates the
//!    kernel, multiplies by weight, and writes partial results to
//!    threadgroup shared memory.
//!
//! 2. **Threadgroup reduction**: Parallel tree reduction within the
//!    threadgroup (log₂(threadgroup_size) steps with barriers).
//!
//! 3. **Device reduction**: A second dispatch reduces threadgroup partial
//!    sums to final output. For small threadgroup counts (<256), the
//!    final reduction is done on CPU after readback.
//!
//! The tensor variant uses the `TensorSplit` decomposition: outer
//! weights index threadgroups, inner weights are loaded into shared
//! memory, and the per-thread weight is `outer[gid] * inner[tid]`.

use jolt_field::Field;

use crate::buffer::MetalBuffer;
use crate::kernel::MetalKernel;

/// Dispatch a pairwise reduction over paired input buffers.
pub fn dispatch_pairwise_reduce<F: Field>(
    _queue: &metal::CommandQueue,
    _inputs: &[&MetalBuffer<F>],
    _weights: &MetalBuffer<F>,
    _kernel: &MetalKernel<F>,
    _num_evals: usize,
) -> Vec<F> {
    todo!()
}

/// Dispatch a tensor (split-eq) pairwise reduction.
pub fn dispatch_tensor_reduce<F: Field>(
    _queue: &metal::CommandQueue,
    _inputs: &[&MetalBuffer<F>],
    _outer_weights: &MetalBuffer<F>,
    _inner_weights: &MetalBuffer<F>,
    _kernel: &MetalKernel<F>,
    _num_evals: usize,
) -> Vec<F> {
    todo!()
}
