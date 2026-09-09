constant uint REGISTERS_READ_WRITE_STATE_TILE_BLOCKS = 256u;
constant uint registers_read_write_operand_carry_kind [[function_constant(1)]];

inline uint registers_read_write_tiled_state_index(
    uint block,
    device const uint* tile_bases,
    device const ushort* offsets,
    uint slot)
{
    return tile_bases[block / REGISTERS_READ_WRITE_STATE_TILE_BLOCKS]
        + uint(offsets[block])
        + slot;
}

inline RegistersReadWriteIndexedStateCell registers_read_write_bind_source_cells(
    RegistersReadWriteCell low,
    RegistersReadWriteCell high,
    SolinasFp128 challenge)
{
    RegistersReadWriteIndexedStateCell output;
    ulong low_value = low.present ? low.value : high.previous;
    ulong high_value = high.present ? high.value : low.next;
    output.value = low_value == high_value
        ? registers_read_write_from_u64(low_value)
        : ram_read_write_bind(
            registers_read_write_from_u64(low_value),
            registers_read_write_from_u64(high_value),
            challenge);
    output.previous = low.present ? low.previous : high.previous;
    output.next = high.present ? high.next : low.next;
    output.ra = (ushort)((high.read_mask << 2u) | low.read_mask);
    output.wa = (uchar)((uint(high.write) << 1u) | uint(low.write));
    return output;
}

inline RegistersReadWriteIndexedStateCell registers_read_write_bind_indexed_cells(
    bool low_present,
    RegistersReadWriteIndexedStateCell low,
    bool high_present,
    RegistersReadWriteIndexedStateCell high,
    SolinasFp128 challenge,
    uint ra_bits,
    uint wa_bits)
{
    RegistersReadWriteIndexedStateCell output;
    SolinasFp128 low_value = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 high_value = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    output.value = all(low_value.limb == high_value.limb)
        ? low_value
        : ram_read_write_bind(low_value, high_value, challenge);
    output.previous = low_present ? low.previous : high.previous;
    output.next = high_present ? high.next : low.next;
    uint low_ra = low_present ? uint(low.ra) : 0u;
    uint high_ra = high_present ? uint(high.ra) : 0u;
    output.ra = (ushort)((high_ra << ra_bits) | low_ra);
    uint low_wa = low_present ? uint(low.wa) : 0u;
    uint high_wa = high_present ? uint(high.wa) : 0u;
    output.wa = (uchar)((high_wa << wa_bits) | low_wa);
    return output;
}

inline RegistersReadWriteDirectStateCell registers_read_write_transition_cells(
    bool low_present,
    RegistersReadWriteIndexedStateCell low,
    bool high_present,
    RegistersReadWriteIndexedStateCell high,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 challenge)
{
    RegistersReadWriteDirectStateCell output;
    SolinasFp128 low_value = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 high_value = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    output.value = ram_read_write_bind(low_value, high_value, challenge);
    output.previous = low_present ? low.previous : high.previous;
    output.next = high_present ? high.next : low.next;
    SolinasFp128 low_ra = low_present ? ra_lut[low.ra] : solinas_zero();
    SolinasFp128 high_ra = high_present ? ra_lut[high.ra] : solinas_zero();
    output.ra = all(low_ra.limb == high_ra.limb)
        ? low_ra
        : ram_read_write_bind(low_ra, high_ra, challenge);
    output.wa = ram_read_write_bind(
        low_present ? wa_lut[low.wa] : solinas_zero(),
        high_present ? wa_lut[high.wa] : solinas_zero(),
        challenge);
    return output;
}

inline RegistersReadWriteDirectStateCell registers_read_write_bind_direct_cells(
    bool low_present,
    RegistersReadWriteDirectStateCell low,
    bool high_present,
    RegistersReadWriteDirectStateCell high,
    SolinasFp128 challenge)
{
    RegistersReadWriteDirectStateCell output;
    SolinasFp128 low_value = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 high_value = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    output.value = ram_read_write_bind(low_value, high_value, challenge);
    output.previous = low_present ? low.previous : high.previous;
    output.next = high_present ? high.next : low.next;
    SolinasFp128 low_ra = low_present ? low.ra : solinas_zero();
    SolinasFp128 high_ra = high_present ? high.ra : solinas_zero();
    output.ra = all(low_ra.limb == high_ra.limb)
        ? low_ra
        : ram_read_write_bind(low_ra, high_ra, challenge);
    output.wa = ram_read_write_bind(
        low_present ? low.wa : solinas_zero(),
        high_present ? high.wa : solinas_zero(),
        challenge);
    return output;
}

inline void registers_read_write_store_indexed_cell(
    uint index,
    uchar column,
    RegistersReadWriteIndexedStateCell cell,
    device uchar* columns,
    device ulong* previous,
    device ulong* next,
    device SolinasFp128* values,
    device ushort* ra,
    device uchar* wa)
{
    columns[index] = column;
    previous[index] = cell.previous;
    next[index] = cell.next;
    values[index] = cell.value;
    ra[index] = cell.ra;
    wa[index] = cell.wa;
}

inline void registers_read_write_store_direct_cell(
    uint index,
    uchar column,
    RegistersReadWriteDirectStateCell cell,
    device uchar* columns,
    device ulong* previous,
    device ulong* next,
    device SolinasFp128* values,
    device SolinasFp128* ra,
    device SolinasFp128* wa)
{
    columns[index] = column;
    previous[index] = cell.previous;
    next[index] = cell.next;
    values[index] = cell.value;
    ra[index] = cell.ra;
    wa[index] = cell.wa;
}

inline void registers_read_write_accumulate_indexed_bound_pair(
    bool low_present,
    RegistersReadWriteIndexedStateCell low,
    bool high_present,
    RegistersReadWriteIndexedStateCell high,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_slope,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    thread SolinasFp128& q_zero,
    thread SolinasFp128& q_infinity)
{
    SolinasFp128 ra_zero = low_present ? ra_lut[low.ra] : solinas_zero();
    SolinasFp128 ra_high = high_present ? ra_lut[high.ra] : solinas_zero();
    SolinasFp128 wa_zero = low_present ? wa_lut[low.wa] : solinas_zero();
    SolinasFp128 wa_high = high_present ? wa_lut[high.wa] : solinas_zero();
    SolinasFp128 val_zero = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 val_high = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    if (low_present) {
        q_zero = solinas_add(
            q_zero,
            solinas_add(
                solinas_mul_wide(ra_zero, val_zero),
                solinas_mul_wide(wa_zero, solinas_add(val_zero, inc_zero))));
    }
    SolinasFp128 val_slope = solinas_sub(val_high, val_zero);
    q_infinity = solinas_add(
        q_infinity,
        solinas_add(
            solinas_mul_wide(solinas_sub(ra_high, ra_zero), val_slope),
            solinas_mul_wide(
                solinas_sub(wa_high, wa_zero),
                solinas_add(val_slope, inc_slope))));
}

inline void registers_read_write_accumulate_direct_bound_pair(
    bool low_present,
    RegistersReadWriteDirectStateCell low,
    bool high_present,
    RegistersReadWriteDirectStateCell high,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_slope,
    thread SolinasFp128& q_zero,
    thread SolinasFp128& q_infinity)
{
    SolinasFp128 ra_zero = low_present ? low.ra : solinas_zero();
    SolinasFp128 ra_high = high_present ? high.ra : solinas_zero();
    SolinasFp128 wa_zero = low_present ? low.wa : solinas_zero();
    SolinasFp128 wa_high = high_present ? high.wa : solinas_zero();
    SolinasFp128 val_zero = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 val_high = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    if (low_present) {
        q_zero = solinas_add(
            q_zero,
            solinas_add(
                solinas_mul_wide(ra_zero, val_zero),
                solinas_mul_wide(wa_zero, solinas_add(val_zero, inc_zero))));
    }
    SolinasFp128 val_slope = solinas_sub(val_high, val_zero);
    q_infinity = solinas_add(
        q_infinity,
        solinas_add(
            solinas_mul_wide(solinas_sub(ra_high, ra_zero), val_slope),
            solinas_mul_wide(
                solinas_sub(wa_high, wa_zero),
                solinas_add(val_slope, inc_slope))));
}

inline uint registers_read_write_rows_write_pattern(
    thread const PackedRegisterCycleRow* rows,
    uint row_base,
    uint row_count)
{
    uint pattern = 0u;
    for (uint index = 0u; index < row_count; index++) {
        pattern |= uint(
            rows[row_base + index].rd_index != REGISTERS_READ_WRITE_NO_REGISTER)
            << index;
    }
    return pattern;
}

inline void registers_read_write_accumulate_indexed_factored_bound_pair(
    bool low_present,
    RegistersReadWriteIndexedStateCell low,
    bool high_present,
    RegistersReadWriteIndexedStateCell high,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    thread SolinasFp128& q_zero,
    thread SolinasFp128& q_infinity)
{
    SolinasFp128 ra_zero = low_present ? ra_lut[low.ra] : solinas_zero();
    SolinasFp128 ra_high = high_present ? ra_lut[high.ra] : solinas_zero();
    SolinasFp128 wa_zero = low_present ? wa_lut[low.wa] : solinas_zero();
    SolinasFp128 wa_high = high_present ? wa_lut[high.wa] : solinas_zero();
    SolinasFp128 val_zero = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 val_high = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    if (low_present) {
        q_zero = solinas_add(
            q_zero,
            solinas_mul_wide(solinas_add(ra_zero, wa_zero), val_zero));
    }
    SolinasFp128 val_slope = solinas_sub(val_high, val_zero);
    q_infinity = solinas_add(
        q_infinity,
        solinas_mul_wide(
            solinas_add(
                solinas_sub(ra_high, ra_zero),
                solinas_sub(wa_high, wa_zero)),
            val_slope));
}

inline RegistersReadWriteMessageTerm registers_read_write_bootstrap_fused(
    uint work,
    thread const PackedRegisterCycleRow* rows_local,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device SolinasFp128* output_increments,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    SolinasFp128 bound_increments[2];
    for (uint pair = 0u; pair < 2u; pair++) {
        bound_increments[pair] = ram_read_write_bind(
            registers_read_write_increment(rows_local[2u * pair]),
            registers_read_write_increment(rows_local[2u * pair + 1u]),
            challenge);
        output_increments[2u * work + pair] = bound_increments[pair];
    }
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    ulong output_mask[2] = {
        registers_read_write_row_mask(rows_local[0])
            | registers_read_write_row_mask(rows_local[1]),
        registers_read_write_row_mask(rows_local[2])
            | registers_read_write_row_mask(rows_local[3]),
    };
    ulong union_mask = output_mask[0] | output_mask[1];
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;
    uint output_base[2] = {
        registers_read_write_tiled_state_index(
            2u * work, output_tile_bases, output_offsets, 0u),
        registers_read_write_tiled_state_index(
            2u * work + 1u, output_tile_bases, output_offsets, 0u),
    };
    uint output_length[2] = {0u, 0u};
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        union_mask &= union_mask - 1ul;
        RegistersReadWriteCell source[4];
        for (uint index = 0u; index < 4u; index++) {
            source[index] = registers_read_write_cell(rows_local[index], column);
        }
        bool low_present = source[0].present || source[1].present;
        bool high_present = source[2].present || source[3].present;
        RegistersReadWriteIndexedStateCell low;
        RegistersReadWriteIndexedStateCell high;
        if (low_present) {
            low = registers_read_write_bind_source_cells(
                source[0], source[1], challenge);
            registers_read_write_store_indexed_cell(
                output_base[0] + output_length[0],
                (uchar)column,
                low,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[0] += 1u;
        }
        if (high_present) {
            high = registers_read_write_bind_source_cells(
                source[2], source[3], challenge);
            registers_read_write_store_indexed_cell(
                output_base[1] + output_length[1],
                (uchar)column,
                high,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[1] += 1u;
        }
        registers_read_write_accumulate_indexed_bound_pair(
            low_present,
            low,
            high_present,
            high,
            bound_increments[0],
            inc_slope,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    output_lengths[2u * work] = (uchar)output_length[0];
    output_lengths[2u * work + 1u] = (uchar)output_length[1];
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline RegistersReadWriteMessageTerm registers_read_write_stateless_bootstrap(
    uint work,
    thread const PackedRegisterCycleRow* rows_local,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    SolinasFp128 bound_increments[2];
    ulong output_mask[2];
    for (uint pair = 0u; pair < 2u; pair++) {
        uint low = 2u * pair;
        uint high = low + 1u;
        bound_increments[pair] = ram_read_write_bind(
            registers_read_write_increment(rows_local[low]),
            registers_read_write_increment(rows_local[high]),
            challenge);
        output_mask[pair] = registers_read_write_row_mask(rows_local[low])
            | registers_read_write_row_mask(rows_local[high]);
    }
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    ulong union_mask = output_mask[0] | output_mask[1];
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;
    uint write_pattern[2] = {
        registers_read_write_rows_write_pattern(rows_local, 0u, 2u),
        registers_read_write_rows_write_pattern(rows_local, 2u, 2u),
    };
    SolinasFp128 write_zero = wa_lut[write_pattern[0]];
    SolinasFp128 q_zero = solinas_mul_wide(
        write_zero, bound_increments[0]);
    SolinasFp128 q_infinity = solinas_mul_wide(
        solinas_sub(wa_lut[write_pattern[1]], write_zero), inc_slope);
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        union_mask &= union_mask - 1ul;
        RegistersReadWriteCell source[4];
        for (uint index = 0u; index < 4u; index++) {
            source[index] = registers_read_write_cell(rows_local[index], column);
        }
        bool low_present = (output_mask[0] & (1ul << column)) != 0ul;
        bool high_present = (output_mask[1] & (1ul << column)) != 0ul;
        RegistersReadWriteIndexedStateCell low;
        RegistersReadWriteIndexedStateCell high;
        if (low_present) {
            low = registers_read_write_bind_source_cells(
                source[0], source[1], challenge);
        }
        if (high_present) {
            high = registers_read_write_bind_source_cells(
                source[2], source[3], challenge);
        }
        registers_read_write_accumulate_indexed_factored_bound_pair(
            low_present,
            low,
            high_present,
            high,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline SolinasFp128 registers_read_write_replay_four_rows_increment(
    thread const PackedRegisterCycleRow* rows,
    uint row_base,
    SolinasFp128 deferred_challenge,
    SolinasFp128 challenge)
{
    SolinasFp128 low = ram_read_write_bind(
        registers_read_write_increment(rows[row_base]),
        registers_read_write_increment(rows[row_base + 1u]),
        deferred_challenge);
    SolinasFp128 high = ram_read_write_bind(
        registers_read_write_increment(rows[row_base + 2u]),
        registers_read_write_increment(rows[row_base + 3u]),
        deferred_challenge);
    return ram_read_write_bind(low, high, challenge);
}

inline ulong registers_read_write_replay_four_rows_mask(
    thread const PackedRegisterCycleRow* rows,
    uint row_base)
{
    return registers_read_write_row_mask(rows[row_base])
        | registers_read_write_row_mask(rows[row_base + 1u])
        | registers_read_write_row_mask(rows[row_base + 2u])
        | registers_read_write_row_mask(rows[row_base + 3u]);
}

inline RegistersReadWriteIndexedStateCell
registers_read_write_replay_four_rows_cell(
    thread const PackedRegisterCycleRow* rows,
    uint row_base,
    uchar column,
    SolinasFp128 deferred_challenge,
    SolinasFp128 challenge,
    thread bool& present)
{
    RegistersReadWriteCell source_low = registers_read_write_cell(
        rows[row_base], column);
    RegistersReadWriteCell source_high = registers_read_write_cell(
        rows[row_base + 1u], column);
    bool first_low_present = source_low.present || source_high.present;
    RegistersReadWriteIndexedStateCell first_low;
    if (first_low_present) {
        first_low = registers_read_write_bind_source_cells(
            source_low, source_high, deferred_challenge);
    }

    source_low = registers_read_write_cell(rows[row_base + 2u], column);
    source_high = registers_read_write_cell(rows[row_base + 3u], column);
    bool first_high_present = source_low.present || source_high.present;
    RegistersReadWriteIndexedStateCell first_high;
    if (first_high_present) {
        first_high = registers_read_write_bind_source_cells(
            source_low, source_high, deferred_challenge);
    }

    present = first_low_present || first_high_present;
    RegistersReadWriteIndexedStateCell output;
    if (present) {
        output = registers_read_write_bind_indexed_cells(
            first_low_present,
            first_low,
            first_high_present,
            first_high,
            challenge,
            4u,
            2u);
    }
    return output;
}

inline SolinasFp128 registers_read_write_replay_eight_rows_increment(
    thread const PackedRegisterCycleRow* rows,
    uint row_base,
    SolinasFp128 first_challenge,
    SolinasFp128 second_challenge,
    SolinasFp128 challenge)
{
    SolinasFp128 low = registers_read_write_replay_four_rows_increment(
        rows, row_base, first_challenge, second_challenge);
    SolinasFp128 high = registers_read_write_replay_four_rows_increment(
        rows, row_base + 4u, first_challenge, second_challenge);
    return ram_read_write_bind(low, high, challenge);
}

inline ulong registers_read_write_replay_eight_rows_mask(
    thread const PackedRegisterCycleRow* rows,
    uint row_base)
{
    return registers_read_write_replay_four_rows_mask(rows, row_base)
        | registers_read_write_replay_four_rows_mask(rows, row_base + 4u);
}

inline RegistersReadWriteIndexedStateCell
registers_read_write_replay_eight_rows_cell(
    thread const PackedRegisterCycleRow* rows,
    uint row_base,
    uchar column,
    SolinasFp128 first_challenge,
    SolinasFp128 second_challenge,
    SolinasFp128 challenge,
    thread bool& present)
{
    bool low_present;
    RegistersReadWriteIndexedStateCell low =
        registers_read_write_replay_four_rows_cell(
            rows,
            row_base,
            column,
            first_challenge,
            second_challenge,
            low_present);
    bool high_present;
    RegistersReadWriteIndexedStateCell high =
        registers_read_write_replay_four_rows_cell(
            rows,
            row_base + 4u,
            column,
            first_challenge,
            second_challenge,
            high_present);
    present = low_present || high_present;
    RegistersReadWriteIndexedStateCell output;
    if (present) {
        output = registers_read_write_bind_indexed_cells(
            low_present,
            low,
            high_present,
            high,
            challenge,
            8u,
            4u);
    }
    return output;
}

inline RegistersReadWriteMessageTerm registers_read_write_replay_bootstrap(
    uint work,
    thread const PackedRegisterCycleRow* rows_local,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device SolinasFp128* output_increments,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 deferred_challenge,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    SolinasFp128 bound_increments[2] = {
        registers_read_write_replay_four_rows_increment(
            rows_local, 0u, deferred_challenge, challenge),
        registers_read_write_replay_four_rows_increment(
            rows_local, 4u, deferred_challenge, challenge),
    };
    output_increments[2u * work] = bound_increments[0];
    output_increments[2u * work + 1u] = bound_increments[1];
    ulong output_mask[2] = {
        registers_read_write_replay_four_rows_mask(rows_local, 0u),
        registers_read_write_replay_four_rows_mask(rows_local, 4u),
    };
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    ulong union_mask = output_mask[0] | output_mask[1];
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;
    uint output_base[2] = {
        registers_read_write_tiled_state_index(
            2u * work, output_tile_bases, output_offsets, 0u),
        registers_read_write_tiled_state_index(
            2u * work + 1u, output_tile_bases, output_offsets, 0u),
    };
    uint output_length[2] = {0u, 0u};
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        union_mask &= union_mask - 1ul;
        bool low_present;
        RegistersReadWriteIndexedStateCell low =
            registers_read_write_replay_four_rows_cell(
                rows_local,
                0u,
                (uchar)column,
                deferred_challenge,
                challenge,
                low_present);
        bool high_present;
        RegistersReadWriteIndexedStateCell high =
            registers_read_write_replay_four_rows_cell(
                rows_local,
                4u,
                (uchar)column,
                deferred_challenge,
                challenge,
                high_present);
        if (low_present) {
            registers_read_write_store_indexed_cell(
                output_base[0] + output_length[0],
                (uchar)column,
                low,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[0] += 1u;
        }
        if (high_present) {
            registers_read_write_store_indexed_cell(
                output_base[1] + output_length[1],
                (uchar)column,
                high,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[1] += 1u;
        }
        registers_read_write_accumulate_indexed_bound_pair(
            low_present,
            low,
            high_present,
            high,
            bound_increments[0],
            inc_slope,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    output_lengths[2u * work] = (uchar)output_length[0];
    output_lengths[2u * work + 1u] = (uchar)output_length[1];
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline RegistersReadWriteMessageTerm
registers_read_write_stateless_replay_bootstrap(
    uint work,
    thread const PackedRegisterCycleRow* rows_local,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 deferred_challenge,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    SolinasFp128 bound_increments[2] = {
        registers_read_write_replay_four_rows_increment(
            rows_local, 0u, deferred_challenge, challenge),
        registers_read_write_replay_four_rows_increment(
            rows_local, 4u, deferred_challenge, challenge),
    };
    ulong output_mask[2] = {
        registers_read_write_replay_four_rows_mask(rows_local, 0u),
        registers_read_write_replay_four_rows_mask(rows_local, 4u),
    };
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    ulong union_mask = output_mask[0] | output_mask[1];
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;
    uint write_pattern[2] = {
        registers_read_write_rows_write_pattern(rows_local, 0u, 4u),
        registers_read_write_rows_write_pattern(rows_local, 4u, 4u),
    };
    SolinasFp128 write_zero = wa_lut[write_pattern[0]];
    SolinasFp128 q_zero = solinas_mul_wide(
        write_zero, bound_increments[0]);
    SolinasFp128 q_infinity = solinas_mul_wide(
        solinas_sub(wa_lut[write_pattern[1]], write_zero), inc_slope);
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        union_mask &= union_mask - 1ul;
        bool low_present;
        RegistersReadWriteIndexedStateCell low =
            registers_read_write_replay_four_rows_cell(
                rows_local,
                0u,
                (uchar)column,
                deferred_challenge,
                challenge,
                low_present);
        bool high_present;
        RegistersReadWriteIndexedStateCell high =
            registers_read_write_replay_four_rows_cell(
                rows_local,
                4u,
                (uchar)column,
                deferred_challenge,
                challenge,
                high_present);
        registers_read_write_accumulate_indexed_factored_bound_pair(
            low_present,
            low,
            high_present,
            high,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline RegistersReadWriteMessageTerm registers_read_write_replay_three_bootstrap(
    uint work,
    thread const PackedRegisterCycleRow* rows_local,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device SolinasFp128* output_increments,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 first_challenge,
    SolinasFp128 second_challenge,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    SolinasFp128 bound_increments[2] = {
        registers_read_write_replay_eight_rows_increment(
            rows_local, 0u, first_challenge, second_challenge, challenge),
        registers_read_write_replay_eight_rows_increment(
            rows_local, 8u, first_challenge, second_challenge, challenge),
    };
    output_increments[2u * work] = bound_increments[0];
    output_increments[2u * work + 1u] = bound_increments[1];
    ulong output_mask[2] = {
        registers_read_write_replay_eight_rows_mask(rows_local, 0u),
        registers_read_write_replay_eight_rows_mask(rows_local, 8u),
    };
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    ulong union_mask = output_mask[0] | output_mask[1];
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;
    uint output_base[2] = {
        registers_read_write_tiled_state_index(
            2u * work, output_tile_bases, output_offsets, 0u),
        registers_read_write_tiled_state_index(
            2u * work + 1u, output_tile_bases, output_offsets, 0u),
    };
    uint output_length[2] = {0u, 0u};
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        union_mask &= union_mask - 1ul;
        bool low_present;
        RegistersReadWriteIndexedStateCell low =
            registers_read_write_replay_eight_rows_cell(
                rows_local,
                0u,
                (uchar)column,
                first_challenge,
                second_challenge,
                challenge,
                low_present);
        bool high_present;
        RegistersReadWriteIndexedStateCell high =
            registers_read_write_replay_eight_rows_cell(
                rows_local,
                8u,
                (uchar)column,
                first_challenge,
                second_challenge,
                challenge,
                high_present);
        if (low_present) {
            registers_read_write_store_indexed_cell(
                output_base[0] + output_length[0],
                (uchar)column,
                low,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[0] += 1u;
        }
        if (high_present) {
            registers_read_write_store_indexed_cell(
                output_base[1] + output_length[1],
                (uchar)column,
                high,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[1] += 1u;
        }
        registers_read_write_accumulate_indexed_bound_pair(
            low_present,
            low,
            high_present,
            high,
            bound_increments[0],
            inc_slope,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    output_lengths[2u * work] = (uchar)output_length[0];
    output_lengths[2u * work + 1u] = (uchar)output_length[1];
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline void registers_read_write_replay_three_materialize_block(
    uint block,
    thread const PackedRegisterCycleRow* rows_local,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device SolinasFp128* output_increments,
    SolinasFp128 first_challenge,
    SolinasFp128 second_challenge,
    SolinasFp128 challenge)
{
    output_increments[block] = registers_read_write_replay_eight_rows_increment(
        rows_local, 0u, first_challenge, second_challenge, challenge);
    ulong mask = registers_read_write_replay_eight_rows_mask(rows_local, 0u);
    uint output_base = registers_read_write_tiled_state_index(
        block, output_tile_bases, output_offsets, 0u);
    uint output_length = 0u;
    while (mask != 0ul) {
        uint column = registers_read_write_mask_first_column(mask);
        mask &= mask - 1ul;
        bool present;
        RegistersReadWriteIndexedStateCell cell =
            registers_read_write_replay_eight_rows_cell(
                rows_local,
                0u,
                (uchar)column,
                first_challenge,
                second_challenge,
                challenge,
                present);
        if (present) {
            registers_read_write_store_indexed_cell(
                output_base + output_length,
                (uchar)column,
                cell,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length += 1u;
        }
    }
    output_lengths[block] = (uchar)registers_read_write_rows_write_pattern(
        rows_local, 0u, 8u);
}

inline RegistersReadWriteMessageTerm registers_read_write_indexed_state_message(
    uint work,
    device const uint* input_tile_bases,
    device const ushort* input_offsets,
    device const ulong* input_masks,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const ushort* input_ra,
    device const uchar* input_wa,
    device const SolinasFp128* input_increments,
    device const uchar* input_write_patterns,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    uint input_block[2] = {2u * work, 2u * work + 1u};
    ulong input_mask[2] = {
        input_masks[input_block[0]],
        input_masks[input_block[1]],
    };
    uint input_base[2] = {
        registers_read_write_tiled_state_index(
            input_block[0], input_tile_bases, input_offsets, 0u),
        registers_read_write_tiled_state_index(
            input_block[1], input_tile_bases, input_offsets, 0u),
    };
    ulong union_mask = input_mask[0] | input_mask[1];
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;
    SolinasFp128 inc_zero = input_increments[input_block[0]];
    SolinasFp128 inc_slope = solinas_sub(
        input_increments[input_block[1]], inc_zero);
    SolinasFp128 write_zero = wa_lut[input_write_patterns[input_block[0]]];
    SolinasFp128 q_zero = solinas_mul_wide(write_zero, inc_zero);
    SolinasFp128 q_infinity = solinas_mul_wide(
        solinas_sub(
            wa_lut[input_write_patterns[input_block[1]]], write_zero),
        inc_slope);
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        ulong bit = 1ul << column;
        ulong lower = bit - 1ul;
        union_mask &= union_mask - 1ul;
        bool low_present = (input_mask[0] & bit) != 0ul;
        RegistersReadWriteIndexedStateCell low;
        if (low_present) {
            uint index = input_base[0]
                + registers_read_write_mask_popcount(input_mask[0] & lower);
            low = registers_read_write_load_indexed(
                index, input_previous, input_next, input_values, input_ra, input_wa);
        }
        bool high_present = (input_mask[1] & bit) != 0ul;
        RegistersReadWriteIndexedStateCell high;
        if (high_present) {
            uint index = input_base[1]
                + registers_read_write_mask_popcount(input_mask[1] & lower);
            high = registers_read_write_load_indexed(
                index, input_previous, input_next, input_values, input_ra, input_wa);
        }
        registers_read_write_accumulate_indexed_factored_bound_pair(
            low_present,
            low,
            high_present,
            high,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline RegistersReadWriteMessageTerm registers_read_write_indexed_fused(
    uint work,
    device const uint* input_tile_bases,
    device const ushort* input_offsets,
    device const ulong* input_masks,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const ushort* input_ra,
    device const uchar* input_wa,
    device const SolinasFp128* input_increments,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device SolinasFp128* output_increments,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    uint input_block[4] = {
        4u * work,
        4u * work + 1u,
        4u * work + 2u,
        4u * work + 3u,
    };
    ulong input_mask[4];
    uint input_base[4];
    ulong union_mask = 0ul;
    for (uint index = 0u; index < 4u; index++) {
        input_mask[index] = input_masks[input_block[index]];
        input_base[index] = registers_read_write_tiled_state_index(
            input_block[index], input_tile_bases, input_offsets, 0u);
        union_mask |= input_mask[index];
    }
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;

    SolinasFp128 bound_increments[2];
    uint output_base[2];
    for (uint pair = 0u; pair < 2u; pair++) {
        bound_increments[pair] = ram_read_write_bind(
            input_increments[input_block[2u * pair]],
            input_increments[input_block[2u * pair + 1u]],
            challenge);
        output_increments[2u * work + pair] = bound_increments[pair];
        output_base[pair] = registers_read_write_tiled_state_index(
            2u * work + pair, output_tile_bases, output_offsets, 0u);
    }
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    uint output_length[2] = {0u, 0u};
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        ulong bit = 1ul << column;
        ulong lower = bit - 1ul;
        union_mask &= union_mask - 1ul;
        bool present[4];
        RegistersReadWriteIndexedStateCell source[4];
        for (uint index = 0u; index < 4u; index++) {
            present[index] = (input_mask[index] & bit) != 0ul;
            if (present[index]) {
                uint source_index = input_base[index]
                    + registers_read_write_mask_popcount(input_mask[index] & lower);
                source[index] = registers_read_write_load_indexed(
                    source_index,
                    input_previous,
                    input_next,
                    input_values,
                    input_ra,
                    input_wa);
            }
        }
        bool low_present = present[0] || present[1];
        bool high_present = present[2] || present[3];
        RegistersReadWriteIndexedStateCell low;
        RegistersReadWriteIndexedStateCell high;
        if (low_present) {
            low = registers_read_write_bind_indexed_cells(
                present[0],
                source[0],
                present[1],
                source[1],
                challenge,
                params.ra_lut_bits,
                params.wa_lut_bits);
            registers_read_write_store_indexed_cell(
                output_base[0] + output_length[0],
                (uchar)column,
                low,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[0] += 1u;
        }
        if (high_present) {
            high = registers_read_write_bind_indexed_cells(
                present[2],
                source[2],
                present[3],
                source[3],
                challenge,
                params.ra_lut_bits,
                params.wa_lut_bits);
            registers_read_write_store_indexed_cell(
                output_base[1] + output_length[1],
                (uchar)column,
                high,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[1] += 1u;
        }
        registers_read_write_accumulate_indexed_bound_pair(
            low_present,
            low,
            high_present,
            high,
            bound_increments[0],
            inc_slope,
            ra_lut,
            wa_lut,
            q_zero,
            q_infinity);
    }
    output_lengths[2u * work] = (uchar)output_length[0];
    output_lengths[2u * work + 1u] = (uchar)output_length[1];
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline RegistersReadWriteMessageTerm registers_read_write_transition_fused(
    uint work,
    device const uint* input_tile_bases,
    device const ushort* input_offsets,
    device const ulong* input_masks,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const ushort* input_ra,
    device const uchar* input_wa,
    device const SolinasFp128* input_increments,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device SolinasFp128* output_ra,
    device SolinasFp128* output_wa,
    device SolinasFp128* output_increments,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    uint input_block[4] = {
        4u * work,
        4u * work + 1u,
        4u * work + 2u,
        4u * work + 3u,
    };
    ulong input_mask[4];
    uint input_base[4];
    ulong union_mask = 0ul;
    for (uint index = 0u; index < 4u; index++) {
        input_mask[index] = input_masks[input_block[index]];
        input_base[index] = registers_read_write_tiled_state_index(
            input_block[index], input_tile_bases, input_offsets, 0u);
        union_mask |= input_mask[index];
    }
    next_count = registers_read_write_mask_popcount(union_mask);
    next_mask = union_mask;

    SolinasFp128 bound_increments[2];
    uint output_base[2];
    for (uint pair = 0u; pair < 2u; pair++) {
        bound_increments[pair] = ram_read_write_bind(
            input_increments[input_block[2u * pair]],
            input_increments[input_block[2u * pair + 1u]],
            challenge);
        output_increments[2u * work + pair] = bound_increments[pair];
        output_base[pair] = registers_read_write_tiled_state_index(
            2u * work + pair, output_tile_bases, output_offsets, 0u);
    }
    SolinasFp128 inc_slope = solinas_sub(
        bound_increments[1], bound_increments[0]);
    uint output_length[2] = {0u, 0u};
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    while (union_mask != 0ul) {
        uint column = registers_read_write_mask_first_column(union_mask);
        ulong bit = 1ul << column;
        ulong lower = bit - 1ul;
        union_mask &= union_mask - 1ul;
        bool present[4];
        RegistersReadWriteIndexedStateCell source[4];
        for (uint index = 0u; index < 4u; index++) {
            present[index] = (input_mask[index] & bit) != 0ul;
            if (present[index]) {
                uint source_index = input_base[index]
                    + registers_read_write_mask_popcount(input_mask[index] & lower);
                source[index] = registers_read_write_load_indexed(
                    source_index,
                    input_previous,
                    input_next,
                    input_values,
                    input_ra,
                    input_wa);
            }
        }
        bool low_present = present[0] || present[1];
        bool high_present = present[2] || present[3];
        RegistersReadWriteDirectStateCell low;
        RegistersReadWriteDirectStateCell high;
        if (low_present) {
            low = registers_read_write_transition_cells(
                present[0],
                source[0],
                present[1],
                source[1],
                ra_lut,
                wa_lut,
                challenge);
            registers_read_write_store_direct_cell(
                output_base[0] + output_length[0],
                (uchar)column,
                low,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[0] += 1u;
        }
        if (high_present) {
            high = registers_read_write_transition_cells(
                present[2],
                source[2],
                present[3],
                source[3],
                ra_lut,
                wa_lut,
                challenge);
            registers_read_write_store_direct_cell(
                output_base[1] + output_length[1],
                (uchar)column,
                high,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            output_length[1] += 1u;
        }
        registers_read_write_accumulate_direct_bound_pair(
            low_present,
            low,
            high_present,
            high,
            bound_increments[0],
            inc_slope,
            q_zero,
            q_infinity);
    }
    output_lengths[2u * work] = (uchar)output_length[0];
    output_lengths[2u * work + 1u] = (uchar)output_length[1];
    SolinasFp128 head = registers_read_write_head(
        work, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline void registers_read_write_store_bound_operand(
    bool low_present,
    uint low_index,
    bool high_present,
    uint high_index,
    uint output_index,
    device const uchar* input_bytes,
    device uchar* output_bytes,
    device const SolinasFp128* weights,
    SolinasFp128 challenge);

inline uint registers_read_write_direct_terminal(
    device const uchar* input_lengths,
    device const uint* input_tile_bases,
    device const ushort* input_offsets,
    device const uchar* input_columns,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const SolinasFp128* input_ra,
    device const SolinasFp128* input_wa,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device SolinasFp128* output_ra,
    device SolinasFp128* output_wa,
    device const uchar* input_operand,
    device uchar* output_operand,
    device const SolinasFp128* operand_weights,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params)
{
    uint low_length = input_lengths[0];
    uint high_length = input_lengths[1];
    uint low_slot = 0u;
    uint high_slot = 0u;
    uint output_length = 0u;
    while (low_slot < low_length || high_slot < high_length) {
        bool low_present = low_slot < low_length;
        bool high_present = high_slot < high_length;
        uint low_index = low_present
            ? registers_read_write_tiled_state_index(
                0u, input_tile_bases, input_offsets, low_slot)
            : 0u;
        uint high_index = high_present
            ? registers_read_write_tiled_state_index(
                1u, input_tile_bases, input_offsets, high_slot)
            : 0u;
        uchar low_column = low_present
            ? input_columns[low_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uchar high_column = high_present
            ? input_columns[high_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        bool take_low = low_column <= high_column;
        bool take_high = high_column <= low_column;
        RegistersReadWriteDirectStateCell low;
        RegistersReadWriteDirectStateCell high;
        if (take_low) {
            low = registers_read_write_load_direct(
                low_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            low_slot += 1u;
        }
        if (take_high) {
            high = registers_read_write_load_direct(
                high_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            high_slot += 1u;
        }
        RegistersReadWriteDirectStateCell output_cell =
            registers_read_write_bind_direct_cells(
                take_low, low, take_high, high, challenge);
        uint output = registers_read_write_tiled_state_index(
            0u,
            output_tile_bases,
            output_offsets,
            output_length);
        registers_read_write_store_direct_cell(
            output,
            min(low_column, high_column),
            output_cell,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
        registers_read_write_store_bound_operand(
            take_low,
            low_index,
            take_high,
            high_index,
            output,
            input_operand,
            output_operand,
            operand_weights,
            challenge);
        output_length += 1u;
    }
    return output_length;
}

inline RegistersReadWriteMessageTerm registers_read_write_direct_fused(
    uint work,
    device const uchar* input_lengths,
    device const uint* input_tile_bases,
    device const ushort* input_offsets,
    device const uchar* input_columns,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const SolinasFp128* input_ra,
    device const SolinasFp128* input_wa,
    device const SolinasFp128* input_increments,
    device uchar* output_lengths,
    device const uint* output_tile_bases,
    device const ushort* output_offsets,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device SolinasFp128* output_ra,
    device SolinasFp128* output_wa,
    device const uchar* input_operand,
    device uchar* output_operand,
    device const SolinasFp128* operand_weights,
    device SolinasFp128* output_increments,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params,
    thread uint& next_count,
    thread ulong& next_mask)
{
    uint input_block[4] = {
        4u * work,
        4u * work + 1u,
        4u * work + 2u,
        4u * work + 3u,
    };
    uint input_length[4];
    uint input_slot[4] = {0u, 0u, 0u, 0u};
    for (uint index = 0u; index < 4u; index++) {
        input_length[index] = input_block[index] < params.input_blocks
            ? input_lengths[input_block[index]]
            : 0u;
    }
    SolinasFp128 bound_increments[2] = {solinas_zero(), solinas_zero()};
    uint pair_count = params.emit_message != 0u ? 2u : 1u;
    for (uint pair = 0u; pair < pair_count; pair++) {
        bound_increments[pair] = ram_read_write_bind(
            input_increments[input_block[2u * pair]],
            input_increments[input_block[2u * pair + 1u]],
            challenge);
        output_increments[2u * work + pair] = bound_increments[pair];
    }
    SolinasFp128 inc_slope = params.emit_message != 0u
        ? solinas_sub(bound_increments[1], bound_increments[0])
        : solinas_zero();
    uint output_length[2] = {0u, 0u};
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    while (true) {
        uchar column = REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uint source_index[4];
        uchar source_column[4];
        for (uint index = 0u; index < 4u; index++) {
            bool valid = input_slot[index] < input_length[index];
            source_index[index] = valid
                ? registers_read_write_tiled_state_index(
                    input_block[index],
                    input_tile_bases,
                    input_offsets,
                    input_slot[index])
                : 0u;
            source_column[index] = valid
                ? input_columns[source_index[index]]
                : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
            column = min(column, source_column[index]);
        }
        if (column == REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER) {
            break;
        }
        next_count += 1u;
        next_mask |= 1ul << uint(column);
        bool present[4];
        RegistersReadWriteDirectStateCell source[4];
        for (uint index = 0u; index < 4u; index++) {
            present[index] = source_column[index] == column;
            if (present[index]) {
                source[index] = registers_read_write_load_direct(
                    source_index[index],
                    input_previous,
                    input_next,
                    input_values,
                    input_ra,
                    input_wa);
                input_slot[index] += 1u;
            }
        }
        bool low_present = present[0] || present[1];
        bool high_present = present[2] || present[3];
        RegistersReadWriteDirectStateCell low;
        RegistersReadWriteDirectStateCell high;
        if (low_present) {
            low = registers_read_write_bind_direct_cells(
                present[0], source[0], present[1], source[1], challenge);
            uint output = registers_read_write_tiled_state_index(
                2u * work,
                output_tile_bases,
                output_offsets,
                output_length[0]);
            registers_read_write_store_direct_cell(
                output,
                column,
                low,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            registers_read_write_store_bound_operand(
                present[0],
                source_index[0],
                present[1],
                source_index[1],
                output,
                input_operand,
                output_operand,
                operand_weights,
                challenge);
            output_length[0] += 1u;
        }
        if (high_present) {
            high = registers_read_write_bind_direct_cells(
                present[2], source[2], present[3], source[3], challenge);
            uint output = registers_read_write_tiled_state_index(
                2u * work + 1u,
                output_tile_bases,
                output_offsets,
                output_length[1]);
            registers_read_write_store_direct_cell(
                output,
                column,
                high,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa);
            registers_read_write_store_bound_operand(
                present[2],
                source_index[2],
                present[3],
                source_index[3],
                output,
                input_operand,
                output_operand,
                operand_weights,
                challenge);
            output_length[1] += 1u;
        }
        if (params.emit_message != 0u) {
            registers_read_write_accumulate_direct_bound_pair(
                low_present,
                low,
                high_present,
                high,
                bound_increments[0],
                inc_slope,
                q_zero,
                q_infinity);
        }
    }
    output_lengths[2u * work] = (uchar)output_length[0];
    if (2u * work + 1u < params.output_blocks) {
        output_lengths[2u * work + 1u] = (uchar)output_length[1];
    }
    RegistersReadWriteMessageTerm result;
    if (params.emit_message != 0u) {
        SolinasFp128 head = registers_read_write_head(
            work, e_in, e_out, params.e_in_length);
        result.q_zero = solinas_mul_wide(head, q_zero);
        result.q_infinity = solinas_mul_wide(head, q_infinity);
    } else {
        result.q_zero = solinas_zero();
        result.q_infinity = solinas_zero();
    }
    return result;
}

kernel void solinas_registers_read_write_bootstrap_fused(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device uchar* output_lengths [[buffer(1)]],
    device uchar* output_columns [[buffer(2)]],
    device ulong* output_previous [[buffer(3)]],
    device ulong* output_next [[buffer(4)]],
    device SolinasFp128* output_values [[buffer(5)]],
    device ushort* output_ra [[buffer(6)]],
    device uchar* output_wa [[buffer(7)]],
    device SolinasFp128* output_increments [[buffer(8)]],
    device const SolinasFp128* ra_lut [[buffer(9)]],
    device const SolinasFp128* wa_lut [[buffer(10)]],
    device const SolinasFp128* e_in [[buffer(11)]],
    device const SolinasFp128* e_out [[buffer(12)]],
    device SolinasFp128* partials [[buffer(13)]],
    constant SolinasFp128& challenge [[buffer(14)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(15)]],
    device ushort* output_offsets [[buffer(16)]],
    device uint* geometry_counts [[buffer(17)]],
    device const uint* output_tile_bases [[buffer(18)]],
    device ushort* geometry_offsets [[buffer(19)]],
    device ulong* geometry_masks [[buffer(20)]],
    device const InstructionInputRow* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_fused_inc_source [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    PackedRegisterCycleRow rows_local[4];
    for (uint index = 0u; index < 4u; index++) {
        uint row = 4u * work + index;
        rows_local[index] = work < params.work_items && row < params.row_count
            ? registers_read_write_load_source_row(
                row,
                rows,
                stage1_instruction_input,
                stage1_fused_inc_source,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
    }
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_bootstrap_fused(
            work,
            rows_local,
            output_lengths,
            output_tile_bases,
            output_offsets,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            challenge,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_stateless_bootstrap_message(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device const SolinasFp128* ra_lut [[buffer(1)]],
    device const SolinasFp128* wa_lut [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(7)]],
    device uint* geometry_counts [[buffer(8)]],
    device ushort* geometry_offsets [[buffer(9)]],
    device ulong* geometry_masks [[buffer(10)]],
    device const InstructionInputRow* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_fused_inc_source [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    PackedRegisterCycleRow rows_local[4];
    for (uint index = 0u; index < 4u; index++) {
        uint row = 4u * work + index;
        rows_local[index] = work < params.work_items && row < params.row_count
            ? registers_read_write_load_source_row(
                row,
                rows,
                stage1_instruction_input,
                stage1_fused_inc_source,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
    }
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_stateless_bootstrap(
            work,
            rows_local,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            challenge,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_stateless_replay_bootstrap_message(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device const SolinasFp128* ra_lut [[buffer(1)]],
    device const SolinasFp128* wa_lut [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& deferred_challenge [[buffer(6)]],
    constant SolinasFp128& challenge [[buffer(7)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(8)]],
    device uint* geometry_counts [[buffer(9)]],
    device ushort* geometry_offsets [[buffer(10)]],
    device ulong* geometry_masks [[buffer(11)]],
    device const InstructionInputRow* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_fused_inc_source [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    PackedRegisterCycleRow rows_local[8];
    for (uint index = 0u; index < 8u; index++) {
        uint row = 8u * work + index;
        rows_local[index] = work < params.work_items && row < params.row_count
            ? registers_read_write_load_source_row(
                row,
                rows,
                stage1_instruction_input,
                stage1_fused_inc_source,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
    }
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_stateless_replay_bootstrap(
            work,
            rows_local,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            deferred_challenge,
            challenge,
            params,
            next_count,
            next_mask);
    }
    if (params.reserved == 0u) {
        registers_read_write_store_geometry(
            next_count,
            next_mask,
            geometry_counts,
            geometry_offsets,
            geometry_masks,
            count_sums,
            group,
            work,
            work < params.work_items,
            tid,
            lane,
            simdgroup);
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_indexed_state_geometry(
    device const ulong* input_masks [[buffer(0)]],
    device uint* geometry_counts [[buffer(1)]],
    device ushort* geometry_offsets [[buffer(2)]],
    device ulong* geometry_masks [[buffer(3)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    bool valid = work < params.work_items;
    ulong mask = valid
        ? input_masks[2u * work] | input_masks[2u * work + 1u]
        : 0ul;
    registers_read_write_store_geometry(
        registers_read_write_mask_popcount(mask),
        mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        valid,
        tid,
        lane,
        simdgroup);
}

kernel void solinas_registers_read_write_replay_bootstrap_fused(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device uchar* output_lengths [[buffer(1)]],
    device uchar* output_columns [[buffer(2)]],
    device ulong* output_previous [[buffer(3)]],
    device ulong* output_next [[buffer(4)]],
    device SolinasFp128* output_values [[buffer(5)]],
    device ushort* output_ra [[buffer(6)]],
    device uchar* output_wa [[buffer(7)]],
    device SolinasFp128* output_increments [[buffer(8)]],
    device const SolinasFp128* ra_lut [[buffer(9)]],
    device const SolinasFp128* wa_lut [[buffer(10)]],
    device const SolinasFp128* e_in [[buffer(11)]],
    device const SolinasFp128* e_out [[buffer(12)]],
    device SolinasFp128* partials [[buffer(13)]],
    constant SolinasFp128& deferred_challenge [[buffer(14)]],
    constant SolinasFp128& challenge [[buffer(15)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(16)]],
    device const ushort* output_offsets [[buffer(17)]],
    device uint* geometry_counts [[buffer(18)]],
    device const uint* output_tile_bases [[buffer(19)]],
    device ushort* geometry_offsets [[buffer(20)]],
    device ulong* geometry_masks [[buffer(21)]],
    device const InstructionInputRow* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_fused_inc_source [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    PackedRegisterCycleRow rows_local[8];
    for (uint index = 0u; index < 8u; index++) {
        uint row = 8u * work + index;
        rows_local[index] = work < params.work_items && row < params.row_count
            ? registers_read_write_load_source_row(
                row,
                rows,
                stage1_instruction_input,
                stage1_fused_inc_source,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
    }
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_replay_bootstrap(
            work,
            rows_local,
            output_lengths,
            output_tile_bases,
            output_offsets,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            deferred_challenge,
            challenge,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_replay_three_bootstrap_fused(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device uchar* output_lengths [[buffer(1)]],
    device uchar* output_columns [[buffer(2)]],
    device ulong* output_previous [[buffer(3)]],
    device ulong* output_next [[buffer(4)]],
    device SolinasFp128* output_values [[buffer(5)]],
    device ushort* output_ra [[buffer(6)]],
    device uchar* output_wa [[buffer(7)]],
    device SolinasFp128* output_increments [[buffer(8)]],
    device const SolinasFp128* ra_lut [[buffer(9)]],
    device const SolinasFp128* wa_lut [[buffer(10)]],
    device const SolinasFp128* e_in [[buffer(11)]],
    device const SolinasFp128* e_out [[buffer(12)]],
    device SolinasFp128* partials [[buffer(13)]],
    constant SolinasFp128& first_challenge [[buffer(14)]],
    constant SolinasFp128& second_challenge [[buffer(15)]],
    constant SolinasFp128& challenge [[buffer(16)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(17)]],
    device const ushort* output_offsets [[buffer(18)]],
    device uint* geometry_counts [[buffer(19)]],
    device const uint* output_tile_bases [[buffer(20)]],
    device ushort* geometry_offsets [[buffer(21)]],
    device ulong* geometry_masks [[buffer(22)]],
    device const InstructionInputRow* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_fused_inc_source [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    PackedRegisterCycleRow rows_local[16];
    for (uint index = 0u; index < 16u; index++) {
        uint row = 16u * work + index;
        rows_local[index] = work < params.work_items && row < params.row_count
            ? registers_read_write_load_source_row(
                row,
                rows,
                stage1_instruction_input,
                stage1_fused_inc_source,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
    }
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_replay_three_bootstrap(
            work,
            rows_local,
            output_lengths,
            output_tile_bases,
            output_offsets,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            first_challenge,
            second_challenge,
            challenge,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_replay_three_materialize(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device uchar* output_lengths [[buffer(1)]],
    device uchar* output_columns [[buffer(2)]],
    device ulong* output_previous [[buffer(3)]],
    device ulong* output_next [[buffer(4)]],
    device SolinasFp128* output_values [[buffer(5)]],
    device ushort* output_ra [[buffer(6)]],
    device uchar* output_wa [[buffer(7)]],
    device SolinasFp128* output_increments [[buffer(8)]],
    constant SolinasFp128& first_challenge [[buffer(9)]],
    constant SolinasFp128& second_challenge [[buffer(10)]],
    constant SolinasFp128& challenge [[buffer(11)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(12)]],
    device const ushort* output_offsets [[buffer(13)]],
    device const uint* output_tile_bases [[buffer(14)]],
    device const InstructionInputRow* stage1_instruction_input [[buffer(23)]],
    device const ulong* stage1_fused_inc_source [[buffer(24)]],
    device const ulong* stage1_rd_post [[buffer(25)]],
    device const uchar* stage1_register_map [[buffer(26)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint block = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    if (block >= params.output_blocks) {
        return;
    }
    PackedRegisterCycleRow rows_local[8];
    for (uint index = 0u; index < 8u; index++) {
        uint row = 8u * block + index;
        rows_local[index] = row < params.row_count
            ? registers_read_write_load_source_row(
                row,
                rows,
                stage1_instruction_input,
                stage1_fused_inc_source,
                stage1_rd_post,
                stage1_register_map,
                params.source_stride)
            : registers_read_write_empty_row();
    }
    registers_read_write_replay_three_materialize_block(
        block,
        rows_local,
        output_lengths,
        output_tile_bases,
        output_offsets,
        output_columns,
        output_previous,
        output_next,
        output_values,
        output_ra,
        output_wa,
        output_increments,
        first_challenge,
        second_challenge,
        challenge);
}

kernel void solinas_registers_read_write_indexed_state_message(
    device const ulong* input_masks [[buffer(0)]],
    device const ulong* input_previous [[buffer(1)]],
    device const ulong* input_next [[buffer(2)]],
    device const SolinasFp128* input_values [[buffer(3)]],
    device const ushort* input_ra [[buffer(4)]],
    device const uchar* input_wa [[buffer(5)]],
    device const SolinasFp128* input_increments [[buffer(6)]],
    device const SolinasFp128* ra_lut [[buffer(7)]],
    device const SolinasFp128* wa_lut [[buffer(8)]],
    device const SolinasFp128* e_in [[buffer(9)]],
    device const SolinasFp128* e_out [[buffer(10)]],
    device SolinasFp128* partials [[buffer(11)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(12)]],
    device const ushort* input_offsets [[buffer(13)]],
    device uint* geometry_counts [[buffer(14)]],
    device const uint* input_tile_bases [[buffer(15)]],
    device ushort* geometry_offsets [[buffer(16)]],
    device ulong* geometry_masks [[buffer(17)]],
    device const uchar* input_write_patterns [[buffer(18)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_indexed_state_message(
            work,
            input_tile_bases,
            input_offsets,
            input_masks,
            input_previous,
            input_next,
            input_values,
            input_ra,
            input_wa,
            input_increments,
            input_write_patterns,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_indexed_bind_message_fused(
    device const uchar* input_lengths [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const ushort* input_ra [[buffer(5)]],
    device const uchar* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device ushort* output_ra [[buffer(13)]],
    device uchar* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* ra_lut [[buffer(16)]],
    device const SolinasFp128* wa_lut [[buffer(17)]],
    device const SolinasFp128* e_in [[buffer(18)]],
    device const SolinasFp128* e_out [[buffer(19)]],
    device SolinasFp128* partials [[buffer(20)]],
    constant SolinasFp128& challenge [[buffer(21)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(22)]],
    device const ushort* input_offsets [[buffer(23)]],
    device ushort* output_offsets [[buffer(24)]],
    device uint* geometry_counts [[buffer(25)]],
    device const uint* input_tile_bases [[buffer(26)]],
    device const uint* output_tile_bases [[buffer(27)]],
    device ushort* geometry_offsets [[buffer(28)]],
    device const ulong* input_masks [[buffer(29)]],
    device ulong* geometry_masks [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_indexed_fused(
            work,
            input_tile_bases,
            input_offsets,
            input_masks,
            input_previous,
            input_next,
            input_values,
            input_ra,
            input_wa,
            input_increments,
            output_lengths,
            output_tile_bases,
            output_offsets,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            challenge,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_transition_bind_message_fused(
    device const uchar* input_lengths [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const ushort* input_ra [[buffer(5)]],
    device const uchar* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device SolinasFp128* output_ra [[buffer(13)]],
    device SolinasFp128* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* ra_lut [[buffer(16)]],
    device const SolinasFp128* wa_lut [[buffer(17)]],
    device const SolinasFp128* e_in [[buffer(18)]],
    device const SolinasFp128* e_out [[buffer(19)]],
    device SolinasFp128* partials [[buffer(20)]],
    constant SolinasFp128& challenge [[buffer(21)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(22)]],
    device const ushort* input_offsets [[buffer(23)]],
    device ushort* output_offsets [[buffer(24)]],
    device uint* geometry_counts [[buffer(25)]],
    device const uint* input_tile_bases [[buffer(26)]],
    device const uint* output_tile_bases [[buffer(27)]],
    device ushort* geometry_offsets [[buffer(28)]],
    device const ulong* input_masks [[buffer(29)]],
    device ulong* geometry_masks [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        term = registers_read_write_transition_fused(
            work,
            input_tile_bases,
            input_offsets,
            input_masks,
            input_previous,
            input_next,
            input_values,
            input_ra,
            input_wa,
            input_increments,
            output_lengths,
            output_tile_bases,
            output_offsets,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            ra_lut,
            wa_lut,
            e_in,
            e_out,
            challenge,
            params,
            next_count,
            next_mask);
    }
    registers_read_write_store_geometry(
        next_count,
        next_mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        work < params.work_items,
        tid,
        lane,
        simdgroup);
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        term.q_zero,
        term.q_infinity);
}

kernel void solinas_registers_read_write_direct_bind_message_fused(
    device const uchar* input_lengths [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const SolinasFp128* input_ra [[buffer(5)]],
    device const SolinasFp128* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device SolinasFp128* output_ra [[buffer(13)]],
    device SolinasFp128* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* e_in [[buffer(16)]],
    device const SolinasFp128* e_out [[buffer(17)]],
    device SolinasFp128* partials [[buffer(18)]],
    constant SolinasFp128& challenge [[buffer(19)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(20)]],
    device const ushort* input_offsets [[buffer(21)]],
    device ushort* output_offsets [[buffer(22)]],
    device uint* geometry_counts [[buffer(23)]],
    device const uint* input_tile_bases [[buffer(24)]],
    device const uint* output_tile_bases [[buffer(25)]],
    device ushort* geometry_offsets [[buffer(26)]],
    device const SolinasFp128* operand_weights [[buffer(27)]],
    device ulong* geometry_masks [[buffer(28)]],
    device const uchar* input_operand [[buffer(29)]],
    device uchar* output_operand [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    uint next_count = 0u;
    ulong next_mask = 0ul;
    if (work < params.work_items) {
        if (params.emit_message == 0u) {
            output_lengths[0] = (uchar)registers_read_write_direct_terminal(
                input_lengths,
                input_tile_bases,
                input_offsets,
                input_columns,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa,
                output_tile_bases,
                output_offsets,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa,
                input_operand,
                output_operand,
                operand_weights,
                challenge,
                params);
            output_increments[0] = ram_read_write_bind(
                input_increments[0], input_increments[1], challenge);
        } else {
            term = registers_read_write_direct_fused(
                work,
                input_lengths,
                input_tile_bases,
                input_offsets,
                input_columns,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa,
                input_increments,
                output_lengths,
                output_tile_bases,
                output_offsets,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa,
                input_operand,
                output_operand,
                operand_weights,
                output_increments,
                e_in,
                e_out,
                challenge,
                params,
                next_count,
                next_mask);
        }
    }
    if (params.emit_message != 0u) {
        registers_read_write_store_geometry(
            next_count,
            next_mask,
            geometry_counts,
            geometry_offsets,
            geometry_masks,
            count_sums,
            group,
            work,
            work < params.work_items,
            tid,
            lane,
            simdgroup);
        registers_read_write_store_partial(
            partials,
            zero_sums,
            infinity_sums,
            params.output_stride,
            group,
            lane,
            simdgroup,
            term.q_zero,
            term.q_infinity);
    }
}

inline RegistersReadWriteMessageTerm registers_read_write_direct_lane_column(
    uint column,
    thread const ulong* input_mask,
    thread const uint* input_base,
    ulong low_mask,
    ulong high_mask,
    uint low_output_base,
    uint high_output_base,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const SolinasFp128* input_ra,
    device const SolinasFp128* input_wa,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device SolinasFp128* output_ra,
    device SolinasFp128* output_wa,
    device const uchar* input_operand,
    device uchar* output_operand,
    device const SolinasFp128* operand_weights,
    SolinasFp128 challenge,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_slope)
{
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    ulong bit = 1ul << column;
    if (((low_mask | high_mask) & bit) == 0ul) {
        return term;
    }
    ulong lower = bit - 1ul;
    bool present[4];
    uint source_index[4] = {0u, 0u, 0u, 0u};
    RegistersReadWriteDirectStateCell source[4];
    for (uint index = 0u; index < 4u; index++) {
        present[index] = (input_mask[index] & bit) != 0ul;
        if (present[index]) {
            source_index[index] = input_base[index]
                + registers_read_write_mask_popcount(input_mask[index] & lower);
            source[index] = registers_read_write_load_direct(
                source_index[index],
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
        }
    }
    bool low_present = present[0] || present[1];
    bool high_present = present[2] || present[3];
    RegistersReadWriteDirectStateCell low;
    RegistersReadWriteDirectStateCell high;
    if (low_present) {
        low = registers_read_write_bind_direct_cells(
            present[0], source[0], present[1], source[1], challenge);
        uint output = low_output_base
            + registers_read_write_mask_popcount(low_mask & lower);
        registers_read_write_store_direct_cell(
            output,
            (uchar)column,
            low,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
        registers_read_write_store_bound_operand(
            present[0],
            source_index[0],
            present[1],
            source_index[1],
            output,
            input_operand,
            output_operand,
            operand_weights,
            challenge);
    }
    if (high_present) {
        high = registers_read_write_bind_direct_cells(
            present[2], source[2], present[3], source[3], challenge);
        uint output = high_output_base
            + registers_read_write_mask_popcount(high_mask & lower);
        registers_read_write_store_direct_cell(
            output,
            (uchar)column,
            high,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
        registers_read_write_store_bound_operand(
            present[2],
            source_index[2],
            present[3],
            source_index[3],
            output,
            input_operand,
            output_operand,
            operand_weights,
            challenge);
    }
    registers_read_write_accumulate_direct_bound_pair(
        low_present,
        low,
        high_present,
        high,
        inc_zero,
        inc_slope,
        term.q_zero,
        term.q_infinity);
    return term;
}

inline ulong registers_read_write_simd_shuffle_mask(
    ulong mask,
    ushort source_lane)
{
    ulong low = (ulong)simd_shuffle((uint)mask, source_lane);
    ulong high = (ulong)simd_shuffle((uint)(mask >> 32u), source_lane);
    return low | (high << 32u);
}

inline SolinasFp128 registers_read_write_simd_shuffle_field(
    SolinasFp128 value,
    ushort source_lane)
{
    SolinasFp128 result;
    result.limb = simd_shuffle(value.limb, source_lane);
    return result;
}

inline SolinasFp128 registers_read_write_simd_sum_eight(SolinasFp128 value)
{
    for (ushort offset = 4u; offset > 0u; offset >>= 1u) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

inline SolinasFp128 registers_read_write_simd_sum_four(SolinasFp128 value)
{
    for (ushort offset = 2u; offset > 0u; offset >>= 1u) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

inline RegistersReadWriteMessageTerm registers_read_write_indexed_lane_column(
    uint column,
    thread const ulong* input_mask,
    thread const uint* input_base,
    ulong low_mask,
    ulong high_mask,
    uint low_output_base,
    uint high_output_base,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const ushort* input_ra,
    device const uchar* input_wa,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 challenge,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_slope,
    uint ra_bits,
    uint wa_bits)
{
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    ulong bit = 1ul << column;
    ulong lower = bit - 1ul;
    bool present[4];
    RegistersReadWriteIndexedStateCell source[4];
    for (uint index = 0u; index < 4u; index++) {
        present[index] = (input_mask[index] & bit) != 0ul;
        if (present[index]) {
            uint source_index = input_base[index]
                + registers_read_write_mask_popcount(input_mask[index] & lower);
            source[index] = registers_read_write_load_indexed(
                source_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
        }
    }
    bool low_present = present[0] || present[1];
    bool high_present = present[2] || present[3];
    RegistersReadWriteIndexedStateCell low;
    RegistersReadWriteIndexedStateCell high;
    if (low_present) {
        low = registers_read_write_bind_indexed_cells(
            present[0],
            source[0],
            present[1],
            source[1],
            challenge,
            ra_bits,
            wa_bits);
        registers_read_write_store_indexed_cell(
            low_output_base + registers_read_write_mask_popcount(low_mask & lower),
            (uchar)column,
            low,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
    }
    if (high_present) {
        high = registers_read_write_bind_indexed_cells(
            present[2],
            source[2],
            present[3],
            source[3],
            challenge,
            ra_bits,
            wa_bits);
        registers_read_write_store_indexed_cell(
            high_output_base + registers_read_write_mask_popcount(high_mask & lower),
            (uchar)column,
            high,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
    }
    registers_read_write_accumulate_indexed_bound_pair(
        low_present,
        low,
        high_present,
        high,
        inc_zero,
        inc_slope,
        ra_lut,
        wa_lut,
        term.q_zero,
        term.q_infinity);
    return term;
}

struct RegistersReadWriteWideIndexedStateCell {
    ulong previous;
    ulong next;
    SolinasFp128 value;
    uint ra;
    ushort wa;
};

inline RegistersReadWriteWideIndexedStateCell
registers_read_write_bind_wide_indexed_cells(
    bool low_present,
    RegistersReadWriteIndexedStateCell low,
    bool high_present,
    RegistersReadWriteIndexedStateCell high,
    SolinasFp128 challenge)
{
    RegistersReadWriteWideIndexedStateCell output;
    SolinasFp128 low_value = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 high_value = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    output.value = ram_read_write_bind(low_value, high_value, challenge);
    output.previous = low_present ? low.previous : high.previous;
    output.next = high_present ? high.next : low.next;
    uint low_ra = low_present ? uint(low.ra) : 0u;
    uint high_ra = high_present ? uint(high.ra) : 0u;
    output.ra = low_ra | (high_ra << 16u);
    uint low_wa = low_present ? uint(low.wa) : 0u;
    uint high_wa = high_present ? uint(high.wa) : 0u;
    output.wa = (ushort)(low_wa | (high_wa << 8u));
    return output;
}

inline RegistersReadWriteWideIndexedStateCell
registers_read_write_load_wide_indexed(
    uint index,
    device const ulong* previous,
    device const ulong* next,
    device const SolinasFp128* values,
    device const uint* ra,
    device const ushort* wa)
{
    RegistersReadWriteWideIndexedStateCell output;
    output.previous = previous[index];
    output.next = next[index];
    output.value = values[index];
    output.ra = ra[index];
    output.wa = wa[index];
    return output;
}

inline void registers_read_write_store_wide_indexed_cell(
    uint index,
    uchar column,
    RegistersReadWriteWideIndexedStateCell cell,
    device uchar* columns,
    device ulong* previous,
    device ulong* next,
    device SolinasFp128* values,
    device uint* ra,
    device ushort* wa)
{
    columns[index] = column;
    previous[index] = cell.previous;
    next[index] = cell.next;
    values[index] = cell.value;
    ra[index] = cell.ra;
    wa[index] = cell.wa;
}

inline RegistersReadWriteDirectStateCell
registers_read_write_evaluate_wide_indexed_cell(
    RegistersReadWriteWideIndexedStateCell cell,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 deferred_challenge)
{
    RegistersReadWriteDirectStateCell output;
    output.previous = cell.previous;
    output.next = cell.next;
    output.value = cell.value;
    output.ra = ram_read_write_bind(
        ra_lut[cell.ra & 65535u],
        ra_lut[cell.ra >> 16u],
        deferred_challenge);
    uint wa = uint(cell.wa);
    output.wa = ram_read_write_bind(
        wa_lut[wa & 255u],
        wa_lut[wa >> 8u],
        deferred_challenge);
    return output;
}

inline SolinasFp128 registers_read_write_evaluate_wide_indexed_coefficient(
    RegistersReadWriteWideIndexedStateCell cell,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 challenge)
{
    uint wa = uint(cell.wa);
    SolinasFp128 low = solinas_add(
        ra_lut[cell.ra & 65535u],
        wa_lut[wa & 255u]);
    SolinasFp128 high = solinas_add(
        ra_lut[cell.ra >> 16u],
        wa_lut[wa >> 8u]);
    return ram_read_write_bind(low, high, challenge);
}

inline uint registers_read_write_compact_rs1_pattern(uint packed_ra)
{
    uint pattern = packed_ra & 0x55555555u;
    pattern = (pattern | (pattern >> 1u)) & 0x33333333u;
    pattern = (pattern | (pattern >> 2u)) & 0x0f0f0f0fu;
    pattern = (pattern | (pattern >> 4u)) & 0x00ff00ffu;
    return (pattern | (pattern >> 8u)) & 0x0000ffffu;
}

inline SolinasFp128 registers_read_write_evaluate_rs1_word(
    ulong pattern,
    device const SolinasFp128* weights,
    uint weight_base)
{
    SolinasFp128 result = solinas_zero();
    while (pattern != 0ul) {
        uint bit = registers_read_write_mask_first_column(pattern);
        result = solinas_add(result, weights[weight_base + bit]);
        pattern &= pattern - 1ul;
    }
    return result;
}

inline SolinasFp128 registers_read_write_evaluate_rs1_pattern128(
    ulong2 pattern,
    device const SolinasFp128* weights,
    uint weight_base)
{
    return solinas_add(
        registers_read_write_evaluate_rs1_word(
            pattern.x, weights, weight_base),
        registers_read_write_evaluate_rs1_word(
            pattern.y, weights, weight_base + 64u));
}

inline void registers_read_write_store_bound_operand(
    bool low_present,
    uint low_index,
    bool high_present,
    uint high_index,
    uint output_index,
    device const uchar* input_bytes,
    device uchar* output_bytes,
    device const SolinasFp128* weights,
    SolinasFp128 challenge)
{
    if (registers_read_write_operand_carry_kind == 1u) {
        device const uint* input = (device const uint*)input_bytes;
        device ulong* output = (device ulong*)output_bytes;
        ulong low = low_present ? ulong(input[low_index]) : 0ul;
        ulong high = high_present ? ulong(input[high_index]) : 0ul;
        output[output_index] = low | (high << 32u);
    } else if (registers_read_write_operand_carry_kind == 2u) {
        device const ulong* input = (device const ulong*)input_bytes;
        device ulong2* output = (device ulong2*)output_bytes;
        ulong low = low_present ? input[low_index] : 0ul;
        ulong high = high_present ? input[high_index] : 0ul;
        output[output_index] = ulong2(low, high);
    } else if (registers_read_write_operand_carry_kind == 3u) {
        device const ulong2* input = (device const ulong2*)input_bytes;
        device SolinasFp128* output = (device SolinasFp128*)output_bytes;
        SolinasFp128 low = low_present
            ? registers_read_write_evaluate_rs1_pattern128(
                input[low_index], weights, 0u)
            : solinas_zero();
        SolinasFp128 high = high_present
            ? registers_read_write_evaluate_rs1_pattern128(
                input[high_index], weights, 128u)
            : solinas_zero();
        output[output_index] = solinas_add(low, high);
    } else if (registers_read_write_operand_carry_kind == 4u) {
        device const SolinasFp128* input =
            (device const SolinasFp128*)input_bytes;
        device SolinasFp128* output = (device SolinasFp128*)output_bytes;
        output[output_index] = ram_read_write_bind(
            low_present ? input[low_index] : solinas_zero(),
            high_present ? input[high_index] : solinas_zero(),
            challenge);
    }
}

inline RegistersReadWriteMessageTerm
registers_read_write_wide_indexed_lane_column(
    uint column,
    thread const ulong* input_mask,
    thread const uint* input_base,
    ulong low_mask,
    ulong high_mask,
    uint low_output_base,
    uint high_output_base,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const ushort* input_ra,
    device const uchar* input_wa,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device uint* output_ra,
    device ushort* output_wa,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 challenge)
{
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    ulong bit = 1ul << column;
    ulong lower = bit - 1ul;
    bool present[4];
    RegistersReadWriteIndexedStateCell source[4];
    for (uint index = 0u; index < 4u; index++) {
        present[index] = (input_mask[index] & bit) != 0ul;
        if (present[index]) {
            uint source_index = input_base[index]
                + registers_read_write_mask_popcount(input_mask[index] & lower);
            source[index] = registers_read_write_load_indexed(
                source_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
        }
    }
    bool low_present = present[0] || present[1];
    bool high_present = present[2] || present[3];
    RegistersReadWriteWideIndexedStateCell low;
    RegistersReadWriteWideIndexedStateCell high;
    if (low_present) {
        low = registers_read_write_bind_wide_indexed_cells(
            present[0], source[0], present[1], source[1], challenge);
        registers_read_write_store_wide_indexed_cell(
            low_output_base + registers_read_write_mask_popcount(low_mask & lower),
            (uchar)column,
            low,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
    }
    if (high_present) {
        high = registers_read_write_bind_wide_indexed_cells(
            present[2], source[2], present[3], source[3], challenge);
        registers_read_write_store_wide_indexed_cell(
            high_output_base + registers_read_write_mask_popcount(high_mask & lower),
            (uchar)column,
            high,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
    }
    SolinasFp128 coefficient_zero = solinas_zero();
    SolinasFp128 coefficient_high = solinas_zero();
    if (low_present) {
        coefficient_zero = registers_read_write_evaluate_wide_indexed_coefficient(
            low, ra_lut, wa_lut, challenge);
    }
    if (high_present) {
        coefficient_high = registers_read_write_evaluate_wide_indexed_coefficient(
            high, ra_lut, wa_lut, challenge);
    }
    SolinasFp128 val_zero = low_present
        ? low.value
        : registers_read_write_from_u64(high.previous);
    SolinasFp128 val_high = high_present
        ? high.value
        : registers_read_write_from_u64(low.next);
    if (low_present) {
        term.q_zero = solinas_mul_wide(coefficient_zero, val_zero);
    }
    term.q_infinity = solinas_mul_wide(
        solinas_sub(coefficient_high, coefficient_zero),
        solinas_sub(val_high, val_zero));
    return term;
}

inline RegistersReadWriteMessageTerm
registers_read_write_wide_transition_lane_column(
    uint column,
    thread const ulong* input_mask,
    thread const uint* input_base,
    ulong low_mask,
    ulong high_mask,
    uint low_output_base,
    uint high_output_base,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const uint* input_ra,
    device const ushort* input_wa,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device SolinasFp128* output_ra,
    device SolinasFp128* output_wa,
    device uint* output_operand,
    bool emit_operand,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 deferred_challenge,
    SolinasFp128 challenge,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_slope)
{
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    ulong bit = 1ul << column;
    ulong lower = bit - 1ul;
    bool present[4];
    RegistersReadWriteDirectStateCell source[4];
    uint rs1_pattern[4] = {0u, 0u, 0u, 0u};
    for (uint index = 0u; index < 4u; index++) {
        present[index] = (input_mask[index] & bit) != 0ul;
        if (present[index]) {
            uint source_index = input_base[index]
                + registers_read_write_mask_popcount(input_mask[index] & lower);
            RegistersReadWriteWideIndexedStateCell packed =
                registers_read_write_load_wide_indexed(
                    source_index,
                    input_previous,
                    input_next,
                    input_values,
                    input_ra,
                    input_wa);
            source[index] = registers_read_write_evaluate_wide_indexed_cell(
                packed, ra_lut, wa_lut, deferred_challenge);
            rs1_pattern[index] = registers_read_write_compact_rs1_pattern(
                packed.ra);
        }
    }
    bool low_present = present[0] || present[1];
    bool high_present = present[2] || present[3];
    RegistersReadWriteDirectStateCell low;
    RegistersReadWriteDirectStateCell high;
    if (low_present) {
        low = registers_read_write_bind_direct_cells(
            present[0], source[0], present[1], source[1], challenge);
        uint output = low_output_base
            + registers_read_write_mask_popcount(low_mask & lower);
        registers_read_write_store_direct_cell(
            output,
            (uchar)column,
            low,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
        if (emit_operand) {
            output_operand[output] = rs1_pattern[0] | (rs1_pattern[1] << 16u);
        }
    }
    if (high_present) {
        high = registers_read_write_bind_direct_cells(
            present[2], source[2], present[3], source[3], challenge);
        uint output = high_output_base
            + registers_read_write_mask_popcount(high_mask & lower);
        registers_read_write_store_direct_cell(
            output,
            (uchar)column,
            high,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
        if (emit_operand) {
            output_operand[output] = rs1_pattern[2] | (rs1_pattern[3] << 16u);
        }
    }
    registers_read_write_accumulate_direct_bound_pair(
        low_present,
        low,
        high_present,
        high,
        inc_zero,
        inc_slope,
        term.q_zero,
        term.q_infinity);
    return term;
}

inline RegistersReadWriteMessageTerm registers_read_write_transition_lane_column(
    uint column,
    thread const ulong* input_mask,
    thread const uint* input_base,
    ulong low_mask,
    ulong high_mask,
    uint low_output_base,
    uint high_output_base,
    device const ulong* input_previous,
    device const ulong* input_next,
    device const SolinasFp128* input_values,
    device const ushort* input_ra,
    device const uchar* input_wa,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device SolinasFp128* output_ra,
    device SolinasFp128* output_wa,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    SolinasFp128 challenge,
    SolinasFp128 inc_zero,
    SolinasFp128 inc_slope)
{
    RegistersReadWriteMessageTerm term = {solinas_zero(), solinas_zero()};
    ulong bit = 1ul << column;
    ulong lower = bit - 1ul;
    bool present[4];
    RegistersReadWriteIndexedStateCell source[4];
    for (uint index = 0u; index < 4u; index++) {
        present[index] = (input_mask[index] & bit) != 0ul;
        if (present[index]) {
            uint source_index = input_base[index]
                + registers_read_write_mask_popcount(input_mask[index] & lower);
            source[index] = registers_read_write_load_indexed(
                source_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
        }
    }
    bool low_present = present[0] || present[1];
    bool high_present = present[2] || present[3];
    RegistersReadWriteDirectStateCell low;
    RegistersReadWriteDirectStateCell high;
    if (low_present) {
        low = registers_read_write_transition_cells(
            present[0],
            source[0],
            present[1],
            source[1],
            ra_lut,
            wa_lut,
            challenge);
        registers_read_write_store_direct_cell(
            low_output_base + registers_read_write_mask_popcount(low_mask & lower),
            (uchar)column,
            low,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
    }
    if (high_present) {
        high = registers_read_write_transition_cells(
            present[2],
            source[2],
            present[3],
            source[3],
            ra_lut,
            wa_lut,
            challenge);
        registers_read_write_store_direct_cell(
            high_output_base + registers_read_write_mask_popcount(high_mask & lower),
            (uchar)column,
            high,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa);
    }
    registers_read_write_accumulate_direct_bound_pair(
        low_present,
        low,
        high_present,
        high,
        inc_zero,
        inc_slope,
        term.q_zero,
        term.q_infinity);
    return term;
}

kernel void solinas_registers_read_write_direct_geometry(
    device const ulong* input_masks [[buffer(0)]],
    device uint* geometry_counts [[buffer(1)]],
    device ushort* geometry_offsets [[buffer(2)]],
    device ulong* geometry_masks [[buffer(3)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint count_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    bool valid = work < params.work_items;
    ulong mask = 0ul;
    if (valid) {
        for (uint index = 0u; index < 4u; index++) {
            mask |= input_masks[4u * work + index];
        }
    }
    registers_read_write_store_geometry(
        registers_read_write_mask_popcount(mask),
        mask,
        geometry_counts,
        geometry_offsets,
        geometry_masks,
        count_sums,
        group,
        work,
        valid,
        tid,
        lane,
        simdgroup);
}

kernel void solinas_registers_read_write_indexed_bind_message_cooperative(
    device const uchar* input_lengths [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const ushort* input_ra [[buffer(5)]],
    device const uchar* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device ushort* output_ra [[buffer(13)]],
    device uchar* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* ra_lut [[buffer(16)]],
    device const SolinasFp128* wa_lut [[buffer(17)]],
    device const SolinasFp128* e_in [[buffer(18)]],
    device const SolinasFp128* e_out [[buffer(19)]],
    device SolinasFp128* partials [[buffer(20)]],
    constant SolinasFp128& challenge [[buffer(21)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(22)]],
    device const ushort* input_offsets [[buffer(23)]],
    device const ushort* output_offsets [[buffer(24)]],
    device uint* geometry_counts [[buffer(25)]],
    device const uint* input_tile_bases [[buffer(26)]],
    device const uint* output_tile_bases [[buffer(27)]],
    device ushort* geometry_offsets [[buffer(28)]],
    device const ulong* input_masks [[buffer(29)]],
    device ulong* geometry_masks [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint local_lane = lane & 3u;
    uint work_in_simdgroup = lane >> 2u;
    uint work = group * 64u
        + simdgroup * 8u
        + work_in_simdgroup;
    SolinasFp128 lane_zero = solinas_zero();
    SolinasFp128 lane_infinity = solinas_zero();
    if (work < params.work_items) {
        ushort subgroup_base = (ushort)(lane - local_lane);
        uint owned_block = 4u * work + local_lane;
        ulong owned_mask = input_masks[owned_block];
        uint owned_base = registers_read_write_tiled_state_index(
            owned_block, input_tile_bases, input_offsets, 0u);
        ulong input_mask[4];
        uint input_base[4];
        for (uint index = 0u; index < 4u; index++) {
            ushort source_lane = subgroup_base + (ushort)index;
            input_mask[index] = registers_read_write_simd_shuffle_mask(
                owned_mask, source_lane);
            input_base[index] = simd_shuffle(owned_base, source_lane);
        }
        ulong low_mask = input_mask[0] | input_mask[1];
        ulong high_mask = input_mask[2] | input_mask[3];
        uint low_output_block = 2u * work;
        uint high_output_block = low_output_block + 1u;
        uint owned_output_base = 0u;
        if (local_lane < 2u) {
            owned_output_base = registers_read_write_tiled_state_index(
                low_output_block + local_lane,
                output_tile_bases,
                output_offsets,
                0u);
        }
        uint low_output_base = simd_shuffle(
            owned_output_base, subgroup_base);
        uint high_output_base = simd_shuffle(
            owned_output_base, subgroup_base + 1u);

        SolinasFp128 inc_zero = solinas_zero();
        SolinasFp128 inc_high = solinas_zero();
        if (local_lane == 0u) {
            inc_zero = ram_read_write_bind(
                input_increments[4u * work],
                input_increments[4u * work + 1u],
                challenge);
            inc_high = ram_read_write_bind(
                input_increments[4u * work + 2u],
                input_increments[4u * work + 3u],
                challenge);
            output_increments[low_output_block] = inc_zero;
            output_increments[high_output_block] = inc_high;
            output_lengths[low_output_block] =
                (uchar)registers_read_write_mask_popcount(low_mask);
            output_lengths[high_output_block] =
                (uchar)registers_read_write_mask_popcount(high_mask);
        }
        inc_zero = registers_read_write_simd_shuffle_field(
            inc_zero, subgroup_base);
        inc_high = registers_read_write_simd_shuffle_field(
            inc_high, subgroup_base);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_zero);

        ulong remaining_mask = low_mask | high_mask;
        while (remaining_mask != 0ul) {
            ulong lane_mask = remaining_mask;
            for (uint rank = 0u; rank < local_lane; rank++) {
                lane_mask &= lane_mask - 1ul;
            }
            if (lane_mask != 0ul) {
                uint column = registers_read_write_mask_first_column(lane_mask);
                RegistersReadWriteMessageTerm term =
                    registers_read_write_indexed_lane_column(
                        column,
                        input_mask,
                        input_base,
                        low_mask,
                        high_mask,
                        low_output_base,
                        high_output_base,
                        input_previous,
                        input_next,
                        input_values,
                        input_ra,
                        input_wa,
                        output_columns,
                        output_previous,
                        output_next,
                        output_values,
                        output_ra,
                        output_wa,
                        ra_lut,
                        wa_lut,
                        challenge,
                        inc_zero,
                        inc_slope,
                        params.ra_lut_bits,
                        params.wa_lut_bits);
                lane_zero = solinas_add(lane_zero, term.q_zero);
                lane_infinity = solinas_add(
                    lane_infinity, term.q_infinity);
            }
            for (uint index = 0u; index < 4u; index++) {
                remaining_mask &= remaining_mask - 1ul;
            }
        }
        lane_zero = registers_read_write_simd_sum_four(lane_zero);
        lane_infinity = registers_read_write_simd_sum_four(lane_infinity);
        if (local_lane == 0u) {
            SolinasFp128 head = registers_read_write_head(
                work, e_in, e_out, params.e_in_length);
            lane_zero = solinas_mul_wide(head, lane_zero);
            lane_infinity = solinas_mul_wide(head, lane_infinity);
        }
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        local_lane == 0u ? lane_zero : solinas_zero(),
        local_lane == 0u ? lane_infinity : solinas_zero());
}

kernel void solinas_registers_read_write_wide_indexed_bind_message_cooperative(
    device const uchar* input_write_patterns [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const ushort* input_ra [[buffer(5)]],
    device const uchar* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device uint* output_ra [[buffer(13)]],
    device ushort* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* ra_lut [[buffer(16)]],
    device const SolinasFp128* wa_lut [[buffer(17)]],
    device const SolinasFp128* e_in [[buffer(18)]],
    device const SolinasFp128* e_out [[buffer(19)]],
    device SolinasFp128* partials [[buffer(20)]],
    constant SolinasFp128& challenge [[buffer(21)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(22)]],
    device const ushort* input_offsets [[buffer(23)]],
    device const ushort* output_offsets [[buffer(24)]],
    device uint* geometry_counts [[buffer(25)]],
    device const uint* input_tile_bases [[buffer(26)]],
    device const uint* output_tile_bases [[buffer(27)]],
    device ushort* geometry_offsets [[buffer(28)]],
    device const ulong* input_masks [[buffer(29)]],
    device ulong* geometry_masks [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint local_lane = lane & 3u;
    uint work_in_simdgroup = lane >> 2u;
    uint work = group * 64u
        + simdgroup * 8u
        + work_in_simdgroup;
    SolinasFp128 lane_zero = solinas_zero();
    SolinasFp128 lane_infinity = solinas_zero();
    if (work < params.work_items) {
        ushort subgroup_base = (ushort)(lane - local_lane);
        uint owned_block = 4u * work + local_lane;
        ulong owned_mask = input_masks[owned_block];
        uint owned_write_pattern = uint(input_write_patterns[owned_block]);
        uint owned_base = registers_read_write_tiled_state_index(
            owned_block, input_tile_bases, input_offsets, 0u);
        ulong input_mask[4];
        uint write_pattern[4];
        uint input_base[4];
        for (uint index = 0u; index < 4u; index++) {
            ushort source_lane = subgroup_base + (ushort)index;
            input_mask[index] = registers_read_write_simd_shuffle_mask(
                owned_mask, source_lane);
            write_pattern[index] = simd_shuffle(
                owned_write_pattern, source_lane);
            input_base[index] = simd_shuffle(owned_base, source_lane);
        }
        ulong low_mask = input_mask[0] | input_mask[1];
        ulong high_mask = input_mask[2] | input_mask[3];
        uint low_output_block = 2u * work;
        uint high_output_block = low_output_block + 1u;
        uint owned_output_base = 0u;
        if (local_lane < 2u) {
            owned_output_base = registers_read_write_tiled_state_index(
                low_output_block + local_lane,
                output_tile_bases,
                output_offsets,
                0u);
        }
        uint low_output_base = simd_shuffle(
            owned_output_base, subgroup_base);
        uint high_output_base = simd_shuffle(
            owned_output_base, subgroup_base + 1u);

        SolinasFp128 inc_zero = solinas_zero();
        SolinasFp128 inc_high = solinas_zero();
        if (local_lane == 0u) {
            inc_zero = ram_read_write_bind(
                input_increments[4u * work],
                input_increments[4u * work + 1u],
                challenge);
            inc_high = ram_read_write_bind(
                input_increments[4u * work + 2u],
                input_increments[4u * work + 3u],
                challenge);
            output_increments[low_output_block] = inc_zero;
            output_increments[high_output_block] = inc_high;
            output_lengths[low_output_block] =
                (uchar)registers_read_write_mask_popcount(low_mask);
            output_lengths[high_output_block] =
                (uchar)registers_read_write_mask_popcount(high_mask);
        }
        inc_zero = registers_read_write_simd_shuffle_field(
            inc_zero, subgroup_base);
        inc_high = registers_read_write_simd_shuffle_field(
            inc_high, subgroup_base);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_zero);
        if (local_lane == 0u) {
            SolinasFp128 write_zero = ram_read_write_bind(
                wa_lut[write_pattern[0]],
                wa_lut[write_pattern[1]],
                challenge);
            SolinasFp128 write_high = ram_read_write_bind(
                wa_lut[write_pattern[2]],
                wa_lut[write_pattern[3]],
                challenge);
            lane_zero = solinas_mul_wide(write_zero, inc_zero);
            lane_infinity = solinas_mul_wide(
                solinas_sub(write_high, write_zero), inc_slope);
        }

        ulong remaining_mask = low_mask | high_mask;
        while (remaining_mask != 0ul) {
            ulong lane_mask = remaining_mask;
            for (uint rank = 0u; rank < local_lane; rank++) {
                lane_mask &= lane_mask - 1ul;
            }
            if (lane_mask != 0ul) {
                uint column = registers_read_write_mask_first_column(lane_mask);
                RegistersReadWriteMessageTerm term =
                    registers_read_write_wide_indexed_lane_column(
                        column,
                        input_mask,
                        input_base,
                        low_mask,
                        high_mask,
                        low_output_base,
                        high_output_base,
                        input_previous,
                        input_next,
                        input_values,
                        input_ra,
                        input_wa,
                        output_columns,
                        output_previous,
                        output_next,
                        output_values,
                        output_ra,
                        output_wa,
                        ra_lut,
                        wa_lut,
                        challenge);
                lane_zero = solinas_add(lane_zero, term.q_zero);
                lane_infinity = solinas_add(
                    lane_infinity, term.q_infinity);
            }
            for (uint index = 0u; index < 4u; index++) {
                remaining_mask &= remaining_mask - 1ul;
            }
        }
        lane_zero = registers_read_write_simd_sum_four(lane_zero);
        lane_infinity = registers_read_write_simd_sum_four(lane_infinity);
        if (local_lane == 0u) {
            SolinasFp128 head = registers_read_write_head(
                work, e_in, e_out, params.e_in_length);
            lane_zero = solinas_mul_wide(head, lane_zero);
            lane_infinity = solinas_mul_wide(head, lane_infinity);
        }
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        local_lane == 0u ? lane_zero : solinas_zero(),
        local_lane == 0u ? lane_infinity : solinas_zero());
}

kernel void solinas_registers_read_write_wide_transition_bind_message_cooperative(
    constant SolinasFp128& deferred_challenge [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const uint* input_ra [[buffer(5)]],
    device const ushort* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device SolinasFp128* output_ra [[buffer(13)]],
    device SolinasFp128* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* ra_lut [[buffer(16)]],
    device const SolinasFp128* wa_lut [[buffer(17)]],
    device const SolinasFp128* e_in [[buffer(18)]],
    device const SolinasFp128* e_out [[buffer(19)]],
    device SolinasFp128* partials [[buffer(20)]],
    constant SolinasFp128& challenge [[buffer(21)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(22)]],
    device const ushort* input_offsets [[buffer(23)]],
    device const ushort* output_offsets [[buffer(24)]],
    device uint* output_operand [[buffer(25)]],
    device const uint* input_tile_bases [[buffer(26)]],
    device const uint* output_tile_bases [[buffer(27)]],
    device ushort* geometry_offsets [[buffer(28)]],
    device const ulong* input_masks [[buffer(29)]],
    device ulong* geometry_masks [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint local_lane = lane & 3u;
    uint work_in_simdgroup = lane >> 2u;
    uint work = group * 64u
        + simdgroup * 8u
        + work_in_simdgroup;
    SolinasFp128 lane_zero = solinas_zero();
    SolinasFp128 lane_infinity = solinas_zero();
    if (work < params.work_items) {
        ushort subgroup_base = (ushort)(lane - local_lane);
        uint owned_block = 4u * work + local_lane;
        ulong owned_mask = input_masks[owned_block];
        uint owned_base = registers_read_write_tiled_state_index(
            owned_block, input_tile_bases, input_offsets, 0u);
        ulong input_mask[4];
        uint input_base[4];
        for (uint index = 0u; index < 4u; index++) {
            ushort source_lane = subgroup_base + (ushort)index;
            input_mask[index] = registers_read_write_simd_shuffle_mask(
                owned_mask, source_lane);
            input_base[index] = simd_shuffle(owned_base, source_lane);
        }
        ulong low_mask = input_mask[0] | input_mask[1];
        ulong high_mask = input_mask[2] | input_mask[3];
        uint low_output_block = 2u * work;
        uint high_output_block = low_output_block + 1u;
        uint owned_output_base = 0u;
        if (local_lane < 2u) {
            owned_output_base = registers_read_write_tiled_state_index(
                low_output_block + local_lane,
                output_tile_bases,
                output_offsets,
                0u);
        }
        uint low_output_base = simd_shuffle(
            owned_output_base, subgroup_base);
        uint high_output_base = simd_shuffle(
            owned_output_base, subgroup_base + 1u);

        SolinasFp128 inc_zero = solinas_zero();
        SolinasFp128 inc_high = solinas_zero();
        if (local_lane == 0u) {
            inc_zero = ram_read_write_bind(
                input_increments[4u * work],
                input_increments[4u * work + 1u],
                challenge);
            inc_high = ram_read_write_bind(
                input_increments[4u * work + 2u],
                input_increments[4u * work + 3u],
                challenge);
            output_increments[low_output_block] = inc_zero;
            output_increments[high_output_block] = inc_high;
            output_lengths[low_output_block] =
                (uchar)registers_read_write_mask_popcount(low_mask);
            output_lengths[high_output_block] =
                (uchar)registers_read_write_mask_popcount(high_mask);
        }
        inc_zero = registers_read_write_simd_shuffle_field(
            inc_zero, subgroup_base);
        inc_high = registers_read_write_simd_shuffle_field(
            inc_high, subgroup_base);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_zero);

        ulong remaining_mask = low_mask | high_mask;
        while (remaining_mask != 0ul) {
            ulong lane_mask = remaining_mask;
            for (uint rank = 0u; rank < local_lane; rank++) {
                lane_mask &= lane_mask - 1ul;
            }
            if (lane_mask != 0ul) {
                uint column = registers_read_write_mask_first_column(lane_mask);
                RegistersReadWriteMessageTerm term =
                    registers_read_write_wide_transition_lane_column(
                        column,
                        input_mask,
                        input_base,
                        low_mask,
                        high_mask,
                        low_output_base,
                        high_output_base,
                        input_previous,
                        input_next,
                        input_values,
                        input_ra,
                        input_wa,
                        output_columns,
                        output_previous,
                        output_next,
                        output_values,
                        output_ra,
                        output_wa,
                        output_operand,
                        params.reserved != 0u,
                        ra_lut,
                        wa_lut,
                        deferred_challenge,
                        challenge,
                        inc_zero,
                        inc_slope);
                lane_zero = solinas_add(lane_zero, term.q_zero);
                lane_infinity = solinas_add(
                    lane_infinity, term.q_infinity);
            }
            for (uint index = 0u; index < 4u; index++) {
                remaining_mask &= remaining_mask - 1ul;
            }
        }
        lane_zero = registers_read_write_simd_sum_four(lane_zero);
        lane_infinity = registers_read_write_simd_sum_four(lane_infinity);
        if (local_lane == 0u) {
            SolinasFp128 head = registers_read_write_head(
                work, e_in, e_out, params.e_in_length);
            lane_zero = solinas_mul_wide(head, lane_zero);
            lane_infinity = solinas_mul_wide(head, lane_infinity);
        }
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        local_lane == 0u ? lane_zero : solinas_zero(),
        local_lane == 0u ? lane_infinity : solinas_zero());
}

kernel void solinas_registers_read_write_transition_bind_message_cooperative(
    device const uchar* input_lengths [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const ushort* input_ra [[buffer(5)]],
    device const uchar* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device SolinasFp128* output_ra [[buffer(13)]],
    device SolinasFp128* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* ra_lut [[buffer(16)]],
    device const SolinasFp128* wa_lut [[buffer(17)]],
    device const SolinasFp128* e_in [[buffer(18)]],
    device const SolinasFp128* e_out [[buffer(19)]],
    device SolinasFp128* partials [[buffer(20)]],
    constant SolinasFp128& challenge [[buffer(21)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(22)]],
    device const ushort* input_offsets [[buffer(23)]],
    device const ushort* output_offsets [[buffer(24)]],
    device uint* geometry_counts [[buffer(25)]],
    device const uint* input_tile_bases [[buffer(26)]],
    device const uint* output_tile_bases [[buffer(27)]],
    device ushort* geometry_offsets [[buffer(28)]],
    device const ulong* input_masks [[buffer(29)]],
    device ulong* geometry_masks [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint local_lane = lane & 3u;
    uint work_in_simdgroup = lane >> 2u;
    uint work = group * 64u
        + simdgroup * 8u
        + work_in_simdgroup;
    SolinasFp128 lane_zero = solinas_zero();
    SolinasFp128 lane_infinity = solinas_zero();
    if (work < params.work_items) {
        ushort subgroup_base = (ushort)(lane - local_lane);
        uint owned_block = 4u * work + local_lane;
        ulong owned_mask = input_masks[owned_block];
        uint owned_base = registers_read_write_tiled_state_index(
            owned_block, input_tile_bases, input_offsets, 0u);
        ulong input_mask[4];
        uint input_base[4];
        for (uint index = 0u; index < 4u; index++) {
            ushort source_lane = subgroup_base + (ushort)index;
            input_mask[index] = registers_read_write_simd_shuffle_mask(
                owned_mask, source_lane);
            input_base[index] = simd_shuffle(owned_base, source_lane);
        }
        ulong low_mask = input_mask[0] | input_mask[1];
        ulong high_mask = input_mask[2] | input_mask[3];
        uint low_output_block = 2u * work;
        uint high_output_block = low_output_block + 1u;
        uint owned_output_base = 0u;
        if (local_lane < 2u) {
            owned_output_base = registers_read_write_tiled_state_index(
                low_output_block + local_lane,
                output_tile_bases,
                output_offsets,
                0u);
        }
        uint low_output_base = simd_shuffle(
            owned_output_base, subgroup_base);
        uint high_output_base = simd_shuffle(
            owned_output_base, subgroup_base + 1u);

        SolinasFp128 inc_zero = solinas_zero();
        SolinasFp128 inc_high = solinas_zero();
        if (local_lane == 0u) {
            inc_zero = ram_read_write_bind(
                input_increments[4u * work],
                input_increments[4u * work + 1u],
                challenge);
            inc_high = ram_read_write_bind(
                input_increments[4u * work + 2u],
                input_increments[4u * work + 3u],
                challenge);
            output_increments[low_output_block] = inc_zero;
            output_increments[high_output_block] = inc_high;
            output_lengths[low_output_block] =
                (uchar)registers_read_write_mask_popcount(low_mask);
            output_lengths[high_output_block] =
                (uchar)registers_read_write_mask_popcount(high_mask);
        }
        inc_zero = registers_read_write_simd_shuffle_field(
            inc_zero, subgroup_base);
        inc_high = registers_read_write_simd_shuffle_field(
            inc_high, subgroup_base);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_zero);

        ulong remaining_mask = low_mask | high_mask;
        while (remaining_mask != 0ul) {
            ulong lane_mask = remaining_mask;
            for (uint rank = 0u; rank < local_lane; rank++) {
                lane_mask &= lane_mask - 1ul;
            }
            if (lane_mask != 0ul) {
                uint column = registers_read_write_mask_first_column(lane_mask);
                RegistersReadWriteMessageTerm term =
                    registers_read_write_transition_lane_column(
                        column,
                        input_mask,
                        input_base,
                        low_mask,
                        high_mask,
                        low_output_base,
                        high_output_base,
                        input_previous,
                        input_next,
                        input_values,
                        input_ra,
                        input_wa,
                        output_columns,
                        output_previous,
                        output_next,
                        output_values,
                        output_ra,
                        output_wa,
                        ra_lut,
                        wa_lut,
                        challenge,
                        inc_zero,
                        inc_slope);
                lane_zero = solinas_add(lane_zero, term.q_zero);
                lane_infinity = solinas_add(
                    lane_infinity, term.q_infinity);
            }
            for (uint index = 0u; index < 4u; index++) {
                remaining_mask &= remaining_mask - 1ul;
            }
        }
        lane_zero = registers_read_write_simd_sum_four(lane_zero);
        lane_infinity = registers_read_write_simd_sum_four(lane_infinity);
        if (local_lane == 0u) {
            SolinasFp128 head = registers_read_write_head(
                work, e_in, e_out, params.e_in_length);
            lane_zero = solinas_mul_wide(head, lane_zero);
            lane_infinity = solinas_mul_wide(head, lane_infinity);
        }
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        local_lane == 0u ? lane_zero : solinas_zero(),
        local_lane == 0u ? lane_infinity : solinas_zero());
}

kernel void solinas_registers_read_write_direct_bind_message_cooperative(
    device const uchar* input_lengths [[buffer(0)]],
    device const uchar* input_columns [[buffer(1)]],
    device const ulong* input_previous [[buffer(2)]],
    device const ulong* input_next [[buffer(3)]],
    device const SolinasFp128* input_values [[buffer(4)]],
    device const SolinasFp128* input_ra [[buffer(5)]],
    device const SolinasFp128* input_wa [[buffer(6)]],
    device const SolinasFp128* input_increments [[buffer(7)]],
    device uchar* output_lengths [[buffer(8)]],
    device uchar* output_columns [[buffer(9)]],
    device ulong* output_previous [[buffer(10)]],
    device ulong* output_next [[buffer(11)]],
    device SolinasFp128* output_values [[buffer(12)]],
    device SolinasFp128* output_ra [[buffer(13)]],
    device SolinasFp128* output_wa [[buffer(14)]],
    device SolinasFp128* output_increments [[buffer(15)]],
    device const SolinasFp128* e_in [[buffer(16)]],
    device const SolinasFp128* e_out [[buffer(17)]],
    device SolinasFp128* partials [[buffer(18)]],
    constant SolinasFp128& challenge [[buffer(19)]],
    constant RegistersReadWriteSequenceParams& params [[buffer(20)]],
    device const ushort* input_offsets [[buffer(21)]],
    device const ushort* output_offsets [[buffer(22)]],
    device const SolinasFp128* operand_weights [[buffer(23)]],
    device const uint* input_tile_bases [[buffer(24)]],
    device const uint* output_tile_bases [[buffer(25)]],
    device ushort* geometry_offsets [[buffer(26)]],
    device const ulong* input_masks [[buffer(27)]],
    device ulong* geometry_masks [[buffer(28)]],
    device const uchar* input_operand [[buffer(29)]],
    device uchar* output_operand [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint local_lane = lane & 7u;
    uint work_in_simdgroup = lane >> 3u;
    uint work = group * 32u
        + simdgroup * 4u
        + work_in_simdgroup;
    SolinasFp128 lane_zero = solinas_zero();
    SolinasFp128 lane_infinity = solinas_zero();
    if (work < params.work_items) {
        ushort subgroup_base = (ushort)(lane - local_lane);
        ulong owned_mask = 0ul;
        uint owned_base = 0u;
        if (local_lane < 4u) {
            uint block = 4u * work + local_lane;
            owned_mask = input_masks[block];
            owned_base = registers_read_write_tiled_state_index(
                block, input_tile_bases, input_offsets, 0u);
        }
        ulong input_mask[4];
        uint input_base[4];
        for (uint index = 0u; index < 4u; index++) {
            ushort source_lane = subgroup_base + (ushort)index;
            input_mask[index] = registers_read_write_simd_shuffle_mask(
                owned_mask, source_lane);
            input_base[index] = simd_shuffle(owned_base, source_lane);
        }
        ulong low_mask = input_mask[0] | input_mask[1];
        ulong high_mask = input_mask[2] | input_mask[3];
        uint low_output_block = 2u * work;
        uint high_output_block = low_output_block + 1u;
        uint owned_output_base = 0u;
        if (local_lane < 2u) {
            owned_output_base = registers_read_write_tiled_state_index(
                low_output_block + local_lane,
                output_tile_bases,
                output_offsets,
                0u);
        }
        uint low_output_base = simd_shuffle(
            owned_output_base, subgroup_base);
        uint high_output_base = simd_shuffle(
            owned_output_base, subgroup_base + 1u);

        SolinasFp128 inc_zero = solinas_zero();
        SolinasFp128 inc_high = solinas_zero();
        if (local_lane == 0u) {
            inc_zero = ram_read_write_bind(
                input_increments[4u * work],
                input_increments[4u * work + 1u],
                challenge);
            inc_high = ram_read_write_bind(
                input_increments[4u * work + 2u],
                input_increments[4u * work + 3u],
                challenge);
            output_increments[low_output_block] = inc_zero;
            output_increments[high_output_block] = inc_high;
            output_lengths[low_output_block] =
                (uchar)registers_read_write_mask_popcount(low_mask);
            output_lengths[high_output_block] =
                (uchar)registers_read_write_mask_popcount(high_mask);
        }
        inc_zero = registers_read_write_simd_shuffle_field(
            inc_zero, subgroup_base);
        inc_high = registers_read_write_simd_shuffle_field(
            inc_high, subgroup_base);
        SolinasFp128 inc_slope = solinas_sub(inc_high, inc_zero);

        for (uint pass = 0u; pass < 8u; pass++) {
            uint column = local_lane + 8u * pass;
            RegistersReadWriteMessageTerm term =
                registers_read_write_direct_lane_column(
                    column,
                    input_mask,
                    input_base,
                    low_mask,
                    high_mask,
                    low_output_base,
                    high_output_base,
                    input_previous,
                    input_next,
                    input_values,
                    input_ra,
                    input_wa,
                    output_columns,
                    output_previous,
                    output_next,
                    output_values,
                    output_ra,
                    output_wa,
                    input_operand,
                    output_operand,
                    operand_weights,
                    challenge,
                    inc_zero,
                    inc_slope);
            lane_zero = solinas_add(lane_zero, term.q_zero);
            lane_infinity = solinas_add(
                lane_infinity, term.q_infinity);
        }
        lane_zero = registers_read_write_simd_sum_eight(lane_zero);
        lane_infinity = registers_read_write_simd_sum_eight(lane_infinity);
        if (local_lane == 0u) {
            SolinasFp128 head = registers_read_write_head(
                work, e_in, e_out, params.e_in_length);
            lane_zero = solinas_mul_wide(head, lane_zero);
            lane_infinity = solinas_mul_wide(head, lane_infinity);
        }
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        local_lane == 0u ? lane_zero : solinas_zero(),
        local_lane == 0u ? lane_infinity : solinas_zero());
}

struct RegistersReadWriteOperandClaimsParams {
    uint row_count;
    uint cycles_per_high_block;
    uint address_bits;
    uint output_stride;
    uint remap_indices;
};

kernel void solinas_registers_read_write_operand_claims(
    device const PackedRegisterCycleRow* rows [[buffer(0)]],
    device const SolinasFp128* e_hi [[buffer(1)]],
    device const SolinasFp128* e_lo [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant RegistersReadWriteOperandClaimsParams& params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 rs1_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 rs2_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint block_start = group * params.cycles_per_high_block;
    uint block_end = min(
        block_start + params.cycles_per_high_block,
        params.row_count);
    SolinasFp128 rs1 = solinas_zero();
    SolinasFp128 rs2 = solinas_zero();
    for (uint row_index = block_start + tid;
         row_index < block_end;
         row_index += REGISTERS_READ_WRITE_SEQUENCE_THREADS) {
        PackedRegisterCycleRow row = rows[row_index];
        uint eq_base = (row_index - block_start) << params.address_bits;
        if (row.rs1_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            rs1 = solinas_add(rs1, e_lo[eq_base + uint(row.rs1_index)]);
        }
        if (row.rs2_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            rs2 = solinas_add(rs2, e_lo[eq_base + uint(row.rs2_index)]);
        }
    }
    rs1 = solinas_simd_sum_32(rs1);
    rs2 = solinas_simd_sum_32(rs2);
    if (lane == 0u) {
        rs1_sums[simdgroup] = rs1;
        rs2_sums[simdgroup] = rs2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        rs1 = lane < REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS
            ? rs1_sums[lane]
            : solinas_zero();
        rs2 = lane < REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS
            ? rs2_sums[lane]
            : solinas_zero();
        rs1 = solinas_simd_sum_32(rs1);
        rs2 = solinas_simd_sum_32(rs2);
        if (lane == 0u) {
            SolinasFp128 high = e_hi[group];
            partials[group] = solinas_mul_wide(high, rs1);
            partials[params.output_stride + group] =
                solinas_mul_wide(high, rs2);
        }
    }
}

kernel void solinas_registers_read_write_compact_rs1_claim(
    device const uchar* rs1_indices [[buffer(0)]],
    device const SolinasFp128* e_hi [[buffer(1)]],
    device const SolinasFp128* e_lo [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant RegistersReadWriteOperandClaimsParams& params [[buffer(4)]],
    device const uchar* register_map [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 rs1_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint block_start = group * params.cycles_per_high_block;
    uint block_end = min(
        block_start + params.cycles_per_high_block,
        params.row_count);
    SolinasFp128 rs1 = solinas_zero();
    for (uint row_index = block_start + tid;
         row_index < block_end;
         row_index += REGISTERS_READ_WRITE_SEQUENCE_THREADS) {
        uint rs1_index = uint(rs1_indices[row_index]);
        if (params.remap_indices != 0u
            && rs1_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            rs1_index = uint(register_map[rs1_index]);
        }
        if (rs1_index != REGISTERS_READ_WRITE_NO_REGISTER) {
            uint eq_base = (row_index - block_start) << params.address_bits;
            rs1 = solinas_add(rs1, e_lo[eq_base + rs1_index]);
        }
    }
    rs1 = solinas_simd_sum_32(rs1);
    if (lane == 0u) {
        rs1_sums[simdgroup] = rs1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        rs1 = lane < REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS
            ? rs1_sums[lane]
            : solinas_zero();
        rs1 = solinas_simd_sum_32(rs1);
        if (lane == 0u) {
            partials[group] = solinas_mul_wide(e_hi[group], rs1);
        }
    }
}
