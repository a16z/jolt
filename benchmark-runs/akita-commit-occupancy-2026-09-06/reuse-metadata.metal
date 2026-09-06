#include <metal_stdlib>
using namespace metal;

struct ReuseParams {
    ulong columns, rows_per_block, blocks, zero_mask;
};

inline uint reuse_symbol(device const uchar *lanes, device const ulong *zeros,
                         constant ReuseParams &params, ulong row, ulong column) {
    uint value = lanes[row * params.columns + column];
    if (value != 0u) return value;
    bool selected_zero = ((params.zero_mask >> column) & 1ul)
        && ((zeros[row >> 6ul] >> (row & 63ul)) & 1ul);
    return selected_zero ? 256u : 0u;
}

kernel void diagnostic_reuse_metadata(
    device const uchar *lanes [[buffer(0)]],
    device const ulong *zeros [[buffer(1)]],
    device uint2 *metadata [[buffer(2)]],
    constant ReuseParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]])
{
    threadgroup uint2 simd_totals[4];
    ulong column = (ulong)task % params.columns;
    ulong block = (ulong)task / params.columns;
    uint first = params.blocks > 1ul ? 1u : 0u;
    uint references = (uint)min(8ul, params.blocks > 1ul ? params.blocks - 1ul : 1ul);
    uint matches = (1u << references) - 1u;
    uint hot = 0u;
    for (ulong local = thread_index; local < params.rows_per_block; local += 128ul) {
        uint value = reuse_symbol(lanes, zeros, params, block * params.rows_per_block + local, column);
        hot += value != 0u;
        for (uint reference = 0u; reference < references && matches != 0u; ++reference) {
            uint bit = 1u << reference;
            if ((matches & bit) != 0u && block != first + reference
                && value != reuse_symbol(lanes, zeros, params,
                    (ulong)(first + reference) * params.rows_per_block + local, column))
                matches &= ~bit;
        }
    }
    hot = simd_sum(hot);
    matches = simd_and(matches);
    if ((thread_index & 31u) == 0u) simd_totals[thread_index >> 5u] = uint2(hot, matches);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
        hot = 0u;
        matches = (1u << references) - 1u;
        for (uint index = 0u; index < 4u; ++index) {
            hot += simd_totals[index].x;
            matches &= simd_totals[index].y;
        }
        uint representative = (uint)block;
        if (hot != 0u && matches != 0u) representative = first + ctz(matches);
        metadata[task] = uint2(hot, representative);
    }
}
