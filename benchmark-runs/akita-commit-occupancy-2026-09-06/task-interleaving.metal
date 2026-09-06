inline void diagnostic_interleaved_task_tile(
    thread AkitaTransposedFp128Accumulator &accumulator_0,
    thread AkitaTransposedFp128Accumulator &accumulator_1,
    threadgroup const uint *shared_matrix,
    device const uchar *lanes,
    device const ulong *active_zero_rows,
    constant PackedOneHotCommitParams &params,
    ulong row_0,
    ulong row_1,
    uint column_0,
    uint column_1,
    uint simd_lane,
    bool active_0,
    bool active_1)
{
    uint local_hot_0 = 0u;
    uint local_hot_1 = 0u;
    bool local_selected_0 = false;
    bool local_selected_1 = false;
    if (simd_lane < PACKED_FP128_D128_RANK3_ROWS_PER_TILE) {
        row_0 += (ulong)simd_lane;
        row_1 += (ulong)simd_lane;
        if (active_0) {
            local_hot_0 = (uint)lanes[
                (row_0 - params.lane_row_offset) * params.lane_stride + (ulong)column_0];
            local_selected_0 = local_hot_0 != 0u;
            if (!local_selected_0 && ((params.zero_column_mask >> column_0) & 1ul) != 0ul)
                local_selected_0 = ((active_zero_rows[row_0 >> 6ul] >> (row_0 & 63ul)) & 1ul) != 0ul;
        }
        if (active_1) {
            local_hot_1 = (uint)lanes[
                (row_1 - params.lane_row_offset) * params.lane_stride + (ulong)column_1];
            local_selected_1 = local_hot_1 != 0u;
            if (!local_selected_1 && ((params.zero_column_mask >> column_1) & 1ul) != 0ul)
                local_selected_1 = ((active_zero_rows[row_1 >> 6ul] >> (row_1 & 63ul)) & 1ul) != 0ul;
        }
    }
    uint selected_0 = uint(simd_ballot(local_selected_0).operator unsigned long());
    uint selected_1 = uint(simd_ballot(local_selected_1).operator unsigned long());
    uint4 coefficients = uint4(simd_lane, simd_lane + 32u, simd_lane + 64u, simd_lane + 96u);
    while ((selected_0 | selected_1) != 0u) {
        if (selected_0 != 0u) {
            uint selected_lane = ctz(selected_0);
            uint selected_hot = simd_shuffle(local_hot_0, selected_lane);
            uint local_position = 2u * selected_lane + (selected_hot >> 7u);
            uint4 shift = uint4(selected_hot & 127u);
            akita_fp128_d512_accumulate_mixed(
                accumulator_0, shared_matrix, local_position * PACKED_FP128_D128_RANK3_D,
                (coefficients - shift) & uint4(127u), coefficients >= shift);
            selected_0 &= selected_0 - 1u;
        }
        if (selected_1 != 0u) {
            uint selected_lane = ctz(selected_1);
            uint selected_hot = simd_shuffle(local_hot_1, selected_lane);
            uint local_position = 2u * selected_lane + (selected_hot >> 7u);
            uint4 shift = uint4(selected_hot & 127u);
            akita_fp128_d512_accumulate_mixed(
                accumulator_1, shared_matrix, local_position * PACKED_FP128_D128_RANK3_D,
                (coefficients - shift) & uint4(127u), coefficients >= shift);
            selected_1 &= selected_1 - 1u;
        }
    }
}
