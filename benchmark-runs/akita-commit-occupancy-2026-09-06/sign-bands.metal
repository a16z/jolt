template<uint Band>
inline void diagnostic_sign_band_add(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources,
    bool mixed_positive)
{
    bool4 positive = bool4(
        Band == 0u && mixed_positive,
        Band < 1u || (Band == 1u && mixed_positive),
        Band < 2u || (Band == 2u && mixed_positive),
        Band < 3u || (Band == 3u && mixed_positive));
    akita_fp128_d512_accumulate_mixed(accumulator, matrix, matrix_base, sources, positive);
}

inline void diagnostic_sign_bands_task_tile(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *shared_matrix,
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
        uint shift = selected_hot & 127u;
        uint4 sources = (coefficients - uint4(shift)) & uint4(127u);
        uint base = local_position * PACKED_FP128_D128_RANK3_D;
        bool mixed_positive = simd_lane >= (shift & 31u);
        switch (shift >> 5u) {
            case 0u:
                diagnostic_sign_band_add<0u>(accumulator, shared_matrix, base, sources, mixed_positive);
                break;
            case 1u:
                diagnostic_sign_band_add<1u>(accumulator, shared_matrix, base, sources, mixed_positive);
                break;
            case 2u:
                diagnostic_sign_band_add<2u>(accumulator, shared_matrix, base, sources, mixed_positive);
                break;
            default:
                diagnostic_sign_band_add<3u>(accumulator, shared_matrix, base, sources, mixed_positive);
                break;
        }
        selected &= selected - 1u;
    }
}
