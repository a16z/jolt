inline void diagnostic_cached_task_tile(
    thread AkitaTransposedFp128Accumulator &accumulator,
    device const AkitaFp128 *matrix,
    ulong matrix_cursor,
    device const uchar *lanes,
    device const ulong *active_zero_rows,
    constant PackedOneHotCommitParams &params,
    ulong tile_row_base,
    uint task_column,
    uint simd_lane)
{
    uint local_hot = 0u;
    bool local_selected = false;
    if (simd_lane < PACKED_FP128_D128_RANK3_ROWS_PER_TILE) {
        ulong trace_row = tile_row_base + (ulong)simd_lane;
        local_hot = (uint)lanes[
            (trace_row - params.lane_row_offset) * params.lane_stride + (ulong)task_column];
        local_selected = local_hot != 0u;
        if (!local_selected && ((params.zero_column_mask >> task_column) & 1ul) != 0ul) {
            ulong active_word = active_zero_rows[trace_row >> 6ul];
            local_selected = ((active_word >> (trace_row & 63ul)) & 1ul) != 0ul;
        }
    }
    uint selected = uint(simd_ballot(local_selected).operator unsigned long());
    uint4 coefficients = uint4(simd_lane, simd_lane + 32u, simd_lane + 64u, simd_lane + 96u);
    while (selected != 0u) {
        uint selected_lane = ctz(selected);
        uint selected_hot = simd_shuffle(local_hot, selected_lane);
        uint local_position = 2u * selected_lane + (selected_hot >> 7u);
        uint4 shift = uint4(selected_hot & 127u);
        uint4 sources = (coefficients - shift) & uint4(127u);
        ulong base = matrix_cursor + (ulong)local_position * 128ul;
        AkitaFp128 value_0 = matrix[base + (ulong)sources.x];
        AkitaFp128 value_1 = matrix[base + (ulong)sources.y];
        AkitaFp128 value_2 = matrix[base + (ulong)sources.z];
        AkitaFp128 value_3 = matrix[base + (ulong)sources.w];
        akita_fp128_d512_accumulate_value(accumulator,
            uint4(value_0.limb.x, value_1.limb.x, value_2.limb.x, value_3.limb.x),
            uint4(value_0.limb.y, value_1.limb.y, value_2.limb.y, value_3.limb.y),
            uint4(value_0.limb.z, value_1.limb.z, value_2.limb.z, value_3.limb.z),
            uint4(value_0.limb.w, value_1.limb.w, value_2.limb.w, value_3.limb.w),
            coefficients >= shift);
        selected &= selected - 1u;
    }
}
