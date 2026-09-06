#include <metal_stdlib>
using namespace metal;

struct NegativeCountParams {
    ulong columns;
    ulong rows_per_fragment;
    ulong tasks;
};

kernel void diagnostic_negative_counts(
    device const uchar *lanes [[buffer(0)]],
    device ushort *counts [[buffer(1)]],
    constant NegativeCountParams &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint histogram[128];
    threadgroup uint simd_totals[4];
    atomic_store_explicit(histogram + thread_index, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong column = (ulong)task % params.columns;
    ulong row_base = ((ulong)task / params.columns) * params.rows_per_fragment;
    for (ulong row = (ulong)thread_index; row < params.rows_per_fragment; row += 128ul) {
        uint shift = (uint)lanes[(row_base + row) * params.columns + column] & 127u;
        if (shift != 0u)
            atomic_fetch_add_explicit(histogram + shift, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint value = atomic_load_explicit(histogram + thread_index, memory_order_relaxed);
    uint prefix = simd_prefix_inclusive_sum(value);
    uint simdgroup = thread_index >> 5u;
    if ((thread_index & 31u) == 31u) simd_totals[simdgroup] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 0u;
    for (uint index = 0u; index < 4u; ++index) {
        total += simd_totals[index];
        if (index < simdgroup) prefix += simd_totals[index];
    }
    counts[(ulong)task * 128ul + (ulong)thread_index] = (ushort)(total - prefix);
}
