inline void diagnostic_complement_add(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint matrix_base,
    uint4 sources,
    bool4 positive)
{
    uint4 value = akita_fp128_d512_gather_word(matrix, 0u, matrix_base, sources);
    value = select(~value, value, positive);
    uint4 sum = accumulator.word_0 + value;
    uint4 carry = uint4(sum < accumulator.word_0);
    accumulator.word_0 = sum;
    value = akita_fp128_d512_gather_word(matrix, 1u, matrix_base, sources);
    accumulator.word_1 = akita_add_transposed_word(
        accumulator.word_1, select(~value, value, positive), carry);
    value = akita_fp128_d512_gather_word(matrix, 2u, matrix_base, sources);
    accumulator.word_2 = akita_add_transposed_word(
        accumulator.word_2, select(~value, value, positive), carry);
    value = akita_fp128_d512_gather_word(matrix, 3u, matrix_base, sources);
    accumulator.word_3 = akita_add_transposed_word(
        accumulator.word_3, select(~value, value, positive), carry);
    accumulator.wraps += int4(carry);
}

inline void diagnostic_deferred_task_tile(
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
        uint4 shift = uint4(selected_hot & 127u);
        diagnostic_complement_add(
            accumulator, shared_matrix, local_position * PACKED_FP128_D128_RANK3_D,
            (coefficients - shift) & uint4(127u), coefficients >= shift);
        selected &= selected - 1u;
    }
}

inline AkitaFp128 diagnostic_deferred_reduce(
    AkitaTransposedFp128Accumulator accumulator,
    uint component,
    ushort negative_count)
{
    ulong word = (ulong)negative_count * ((ulong)AKITA_OFFSET - 1ul);
    AkitaFp128 correction = akita_zero();
    correction.limb.x = (uint)word;
    correction.limb.y = (uint)(word >> 32ul);
    return akita_sub(akita_reduce_transposed_fp128(accumulator, component), correction);
}

inline void diagnostic_deferred_store(
    device AkitaFp128 *partials,
    AkitaTransposedFp128Accumulator accumulator,
    device const ushort *negative_counts,
    constant PackedOneHotCommitParams &params,
    uint task_column,
    uint task_block,
    uint element,
    uint position_partial,
    uint simd_lane)
{
    ulong block = (ulong)task_column * params.blocks_per_column + (ulong)task_block;
    ulong output_base = (block * params.n_a + (ulong)element) * 128ul;
    ulong partial_base = (ulong)position_partial * params.output_coefficients + output_base;
    ulong count_base = (((ulong)task_block * params.position_partials_per_block
        + (ulong)position_partial) * params.num_columns + (ulong)task_column) * 128ul;
    partials[partial_base + simd_lane] = diagnostic_deferred_reduce(
        accumulator, 0u, negative_counts[count_base + simd_lane]);
    partials[partial_base + simd_lane + 32ul] = diagnostic_deferred_reduce(
        accumulator, 1u, negative_counts[count_base + simd_lane + 32ul]);
    partials[partial_base + simd_lane + 64ul] = diagnostic_deferred_reduce(
        accumulator, 2u, negative_counts[count_base + simd_lane + 64ul]);
    partials[partial_base + simd_lane + 96ul] = diagnostic_deferred_reduce(
        accumulator, 3u, negative_counts[count_base + simd_lane + 96ul]);
}
