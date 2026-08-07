// Candidate opening layout. Concatenate after shader.metal.

#define OUTER_REMAINDER_PADDED_TILE_ROWS 56u
#define OUTER_REMAINDER_PADDED_SOURCE_WORDS 20u
#define OUTER_REMAINDER_PADDED_ROW_STRIDE_WORDS 21u

inline ulong outer_padded_staged_word(
    threadgroup const ulong* row_words,
    uint row,
    uint word)
{
    return row_words[
        row * OUTER_REMAINDER_PADDED_ROW_STRIDE_WORDS + word];
}

inline SolinasFp128 outer_padded_opening_value(
    threadgroup const ulong* row_words,
    uint row,
    uint column)
{
    ulong flags = outer_padded_staged_word(row_words, row, 5u);
    bool load = outer_flag(flags, 0u) != 0;
    bool store = outer_flag(flags, 1u) != 0;
    ulong rs2 = outer_padded_staged_word(row_words, row, 2u);
    ulong memory_0 = outer_padded_staged_word(row_words, row, 12u);
    ulong memory_1 = outer_padded_staged_word(row_words, row, 13u);
    ulong ram_address = load || store ? memory_0 : 0ul;
    ulong rd_write = store ? 0ul : (load ? memory_1 : memory_0);
    ulong ram_read = load || store ? memory_1 : 0ul;
    ulong ram_write = load ? memory_1 : (store ? rs2 : 0ul);

    switch (column) {
        case 0u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 6u));
        case 1u:
            return outer_from_signed_u128(
                outer_padded_staged_word(row_words, row, 7u),
                outer_padded_staged_word(row_words, row, 8u),
                outer_flag(flags, 17u) != 0);
        case 2u:
            return outer_from_signed_u128(
                outer_padded_staged_word(row_words, row, 9u),
                outer_padded_staged_word(row_words, row, 10u),
                outer_flag(flags, 19u) != 0);
        case 3u: return outer_from_u64((ulong)outer_flag(flags, 6u));
        case 4u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 11u));
        case 5u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 1u));
        case 6u:
            return outer_from_signed_u128(
                outer_padded_staged_word(row_words, row, 3u),
                outer_padded_staged_word(row_words, row, 4u),
                outer_flag(flags, 18u) != 0);
        case 7u: return outer_from_u64(ram_address);
        case 8u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 0u));
        case 9u: return outer_from_u64(rs2);
        case 10u: return outer_from_u64(rd_write);
        case 11u: return outer_from_u64(ram_read);
        case 12u: return outer_from_u64(ram_write);
        case 13u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 14u));
        case 14u:
            return outer_from_signed_u128(
                outer_padded_staged_word(row_words, row, 15u),
                outer_padded_staged_word(row_words, row, 16u),
                true);
        case 15u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 17u));
        case 16u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 18u));
        case 17u: return outer_from_u64((ulong)outer_flag(flags, 11u));
        case 18u: return outer_from_u64((ulong)outer_flag(flags, 12u));
        case 19u:
            return outer_from_u64(outer_padded_staged_word(row_words, row, 19u));
        case 20u: return outer_from_u64((ulong)outer_flag(flags, 8u));
        case 21u: return outer_from_u64((ulong)outer_flag(flags, 2u));
        case 22u: return outer_from_u64((ulong)outer_flag(flags, 3u));
        case 23u: return outer_from_u64((ulong)outer_flag(flags, 4u));
        case 24u: return outer_from_u64((ulong)outer_flag(flags, 0u));
        case 25u: return outer_from_u64((ulong)outer_flag(flags, 1u));
        case 26u: return outer_from_u64((ulong)outer_flag(flags, 5u));
        case 27u: return outer_from_u64((ulong)outer_flag(flags, 14u));
        case 28u: return outer_from_u64((ulong)outer_flag(flags, 9u));
        case 29u: return outer_from_u64((ulong)outer_flag(flags, 7u));
        case 30u: return outer_from_u64((ulong)outer_flag(flags, 15u));
        case 31u: return outer_from_u64((ulong)outer_flag(flags, 13u));
        case 32u: return outer_from_u64((ulong)outer_flag(flags, 16u));
        case 33u: return outer_from_u64((ulong)outer_flag(flags, 24u));
        case 34u: return outer_from_u64((ulong)outer_flag(flags, 10u));
        case 35u:
            return outer_product_uniskip_endpoint(
                outer_from_u64(outer_padded_staged_word(row_words, row, 6u)),
                outer_from_signed_u128(
                    outer_padded_staged_word(row_words, row, 7u),
                    outer_padded_staged_word(row_words, row, 8u),
                    outer_flag(flags, 17u) != 0),
                outer_from_u64(outer_padded_staged_word(row_words, row, 19u)),
                flags,
                false);
        default:
            return outer_product_uniskip_endpoint(
                outer_from_u64(outer_padded_staged_word(row_words, row, 6u)),
                outer_from_signed_u128(
                    outer_padded_staged_word(row_words, row, 7u),
                    outer_padded_staged_word(row_words, row, 8u),
                    outer_flag(flags, 17u) != 0),
                outer_from_u64(outer_padded_staged_word(row_words, row, 19u)),
                flags,
                true);
    }
}

kernel void solinas_outer_remainder_opening_tiles_padded_56(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterUniskipResidualRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant OuterRemainderOpeningParams& params [[buffer(5)]],
    threadgroup ulong* row_words [[threadgroup(0)]],
    threadgroup SolinasFp128* tile_weights [[threadgroup(1)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint simdgroups = threads / OUTER_REMAINDER_SIMD_WIDTH;
    if (tid < params.columns) {
        partials[block * params.columns + tid] = solinas_zero();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint x_out = block;
         x_out < params.e_out_length;
         x_out += params.blocks) {
        SolinasFp128 sums[OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP];
        for (uint slot = 0u;
             slot < OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP;
             slot++) {
            sums[slot] = solinas_zero();
        }
        uint block_start = x_out * params.e_in_length;
        for (uint tile_start = 0u;
             tile_start < params.e_in_length;
             tile_start += OUTER_REMAINDER_PADDED_TILE_ROWS) {
            uint tile_count = min(
                OUTER_REMAINDER_PADDED_TILE_ROWS,
                params.e_in_length - tile_start);
            for (uint flat = tid;
                 flat < tile_count * OUTER_REMAINDER_PADDED_SOURCE_WORDS;
                 flat += threads) {
                uint tile_row = flat / OUTER_REMAINDER_PADDED_SOURCE_WORDS;
                uint word = flat -
                    tile_row * OUTER_REMAINDER_PADDED_SOURCE_WORDS;
                uint source_row = block_start + tile_start + tile_row;
                row_words[
                    tile_row * OUTER_REMAINDER_PADDED_ROW_STRIDE_WORDS + word
                ] = word < 6u
                    ? instruction_input_row_word(compact_rows[source_row], word)
                    : spartan_outer_residual_word(
                        residual_rows[source_row], word - 6u);
            }
            for (uint tile_row = tid; tile_row < tile_count; tile_row += threads) {
                tile_weights[tile_row] = e_in[tile_start + tile_row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint slot = 0u;
                 slot < OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP;
                 slot++) {
                uint column = simdgroup + slot * simdgroups;
                if (column < params.columns) {
                    SolinasFp128 sum = sums[slot];
                    for (uint tile_row = lane;
                         tile_row < tile_count;
                         tile_row += OUTER_REMAINDER_SIMD_WIDTH) {
                        SolinasFp128 value = outer_padded_opening_value(
                            row_words, tile_row, column);
                        SolinasFp128 contribution;
                        if (outer_opening_is_boolean(column)) {
                            bool set = value.limb[0] != 0u;
                            contribution = set
                                ? tile_weights[tile_row]
                                : solinas_zero();
                        } else {
                            contribution = solinas_mul_wide(
                                tile_weights[tile_row], value);
                        }
                        sum = solinas_add(sum, contribution);
                    }
                    sums[slot] = sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint slot = 0u;
             slot < OUTER_REMAINDER_MAX_COLUMNS_PER_SIMDGROUP;
             slot++) {
            uint column = simdgroup + slot * simdgroups;
            if (column < params.columns) {
                SolinasFp128 column_sum = solinas_simd_sum_32(sums[slot]);
                if (lane == 0u) {
                    uint output = block * params.columns + column;
                    partials[output] = solinas_add(
                        partials[output],
                        solinas_mul_wide(e_out[x_out], column_sum));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
