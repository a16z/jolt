kernel void diagnostic_reuse_hot_counts(
    device const uchar *lanes [[buffer(0)]],
    device const ulong *zeros [[buffer(1)]],
    device uint *hot_counts [[buffer(2)]],
    constant ReuseParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]])
{
    threadgroup uint simd_totals[4];
    ulong column = (ulong)task % params.columns;
    ulong first_row = ((ulong)task / params.columns) * params.rows_per_block;
    uint hot = 0u;
    for (ulong local = thread_index; local < params.rows_per_block; local += 128ul)
        hot += reuse_symbol(lanes, zeros, params, first_row + local, column) != 0u;
    hot = simd_sum(hot);
    if ((thread_index & 31u) == 0u) simd_totals[thread_index >> 5u] = hot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u)
        hot_counts[task] = simd_totals[0] + simd_totals[1] + simd_totals[2] + simd_totals[3];
}

kernel void diagnostic_short_reuse_metadata(
    device const uchar *lanes [[buffer(0)]],
    device const ulong *zeros [[buffer(1)]],
    device uint2 *metadata [[buffer(2)]],
    constant ReuseParams &params [[buffer(3)]],
    device const uint *hot_counts [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]])
{
    threadgroup uint simd_matches[4];
    threadgroup uint group_matches;
    ulong column = (ulong)task % params.columns;
    uint block = (uint)((ulong)task / params.columns);
    uint hot = hot_counts[task];
    uint representative = block;
    uint first = params.blocks > 1ul ? 1u : 0u;
    uint references = (uint)min(8ul, params.blocks > 1ul ? params.blocks - 1ul : 1ul);
    // Every branch around a barrier depends only on this group's task or reduction.
    if (hot != 0u) {
        for (uint reference = first; reference < first + references; ++reference) {
            if (hot_counts[(ulong)reference * params.columns + column] != hot) continue;
            if (reference == block) break;
            uint equal = 1u;
            for (ulong local = thread_index; local < params.rows_per_block && equal != 0u; local += 128ul)
                equal = reuse_symbol(lanes, zeros, params, (ulong)block * params.rows_per_block + local, column)
                    == reuse_symbol(lanes, zeros, params, (ulong)reference * params.rows_per_block + local, column);
            equal = simd_and(equal);
            if ((thread_index & 31u) == 0u) simd_matches[thread_index >> 5u] = equal;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_index == 0u)
                group_matches = simd_matches[0] & simd_matches[1] & simd_matches[2] & simd_matches[3];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint matched = group_matches;
            // All readers finish before a later reference can overwrite the flag.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (matched != 0u) {
                representative = reference;
                break;
            }
        }
    }
    if (thread_index == 0u) metadata[task] = uint2(hot, representative);
}
