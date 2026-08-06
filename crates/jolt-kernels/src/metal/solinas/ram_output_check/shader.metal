// Concatenate after offset-specialized fp128.metal and simd_reduce.metal.

struct RamOutputCheckFoldParams {
    uint block_elements;
    uint blocks;
    uint chunks_per_block;
    uint chunk_elements;
};

inline SolinasFp128 ram_output_check_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

kernel void solinas_ram_output_check_fold_partials(
    device const ulong* val_final [[buffer(0)]],
    device const SolinasFp128* low_weights [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant RamOutputCheckFoldParams& params [[buffer(3)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint partial [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint partial_count = params.blocks * params.chunks_per_block;
    if (partial >= partial_count) {
        return;
    }

    uint block = partial / params.chunks_per_block;
    uint chunk = partial - block * params.chunks_per_block;
    uint low_start = chunk * params.chunk_elements;
    uint low_end = min(low_start + params.chunk_elements, params.block_elements);
    uint value_start = block * params.block_elements;

    SolinasFp128 sum = solinas_zero();
    for (uint low = low_start + tid; low < low_end; low += threads) {
        sum = solinas_add(
            sum,
            solinas_mul_wide(
                ram_output_check_from_u64(val_final[value_start + low]),
                low_weights[low]));
    }

    sum = solinas_simd_sum_32(sum);
    if (lane == 0u) {
        shared[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane == 0u) {
        SolinasFp128 block_sum = solinas_zero();
        uint simdgroups = threads / 32u;
        for (uint group = 0u; group < simdgroups; group++) {
            block_sum = solinas_add(block_sum, shared[group]);
        }
        partials[partial] = block_sum;
    }
}

kernel void solinas_ram_output_check_fold_reduce(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamOutputCheckFoldParams& params [[buffer(2)]],
    uint block [[thread_position_in_grid]])
{
    if (block >= params.blocks) {
        return;
    }

    SolinasFp128 sum = solinas_zero();
    uint start = block * params.chunks_per_block;
    for (uint chunk = 0u; chunk < params.chunks_per_block; chunk++) {
        sum = solinas_add(sum, partials[start + chunk]);
    }
    output[block] = sum;
}
