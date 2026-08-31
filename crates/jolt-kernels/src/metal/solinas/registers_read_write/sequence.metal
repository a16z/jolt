#define REGISTERS_READ_WRITE_SEQUENCE_THREADS 256u
#define REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS 8u
#define REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER 255u

struct RegistersReadWriteSequenceParams {
    uint row_count;
    uint input_blocks;
    uint output_blocks;
    uint input_capacity;
    uint output_capacity;
    uint work_items;
    uint output_stride;
    uint e_in_length;
    uint ra_lut_bits;
    uint wa_lut_bits;
    uint emit_message;
    uint reserved;
    uint source_stride;
};

struct RegistersReadWriteIndexedStateCell {
    SolinasFp128 value;
    ulong previous;
    ulong next;
    ushort ra;
    uchar wa;
};

struct RegistersReadWriteDirectStateCell {
    SolinasFp128 value;
    ulong previous;
    ulong next;
    SolinasFp128 ra;
    SolinasFp128 wa;
};

struct RegistersReadWriteMessageTerm {
    SolinasFp128 q_zero;
    SolinasFp128 q_infinity;
};

inline uint registers_read_write_state_index(
    uint block,
    uint capacity,
    uint slot)
{
    return block * capacity + slot;
}

inline void registers_read_write_sort_columns(
    thread uchar* columns)
{
    for (uint end = 6u; end > 1u; end--) {
        for (uint index = 1u; index < end; index++) {
            if (columns[index - 1u] > columns[index]) {
                uchar temporary = columns[index - 1u];
                columns[index - 1u] = columns[index];
                columns[index] = temporary;
            }
        }
    }
}

inline RegistersReadWriteIndexedStateCell registers_read_write_load_indexed(
    uint index,
    device const ulong* previous,
    device const ulong* next,
    device const SolinasFp128* values,
    device const ushort* ra,
    device const uchar* wa)
{
    RegistersReadWriteIndexedStateCell cell;
    cell.value = values[index];
    cell.previous = previous[index];
    cell.next = next[index];
    cell.ra = ra[index];
    cell.wa = wa[index];
    return cell;
}

inline RegistersReadWriteDirectStateCell registers_read_write_load_direct(
    uint index,
    device const ulong* previous,
    device const ulong* next,
    device const SolinasFp128* values,
    device const SolinasFp128* ra,
    device const SolinasFp128* wa)
{
    RegistersReadWriteDirectStateCell cell;
    cell.value = values[index];
    cell.previous = previous[index];
    cell.next = next[index];
    cell.ra = ra[index];
    cell.wa = wa[index];
    return cell;
}

inline uint registers_read_write_bind_source_block(
    uint block,
    device const PackedRegisterCycleRow* rows,
    uint row_count,
    device uchar* output_columns,
    device ulong* output_previous,
    device ulong* output_next,
    device SolinasFp128* output_values,
    device ushort* output_ra,
    device uchar* output_wa,
    device SolinasFp128* output_increments,
    SolinasFp128 challenge,
    uint output_capacity)
{
    uint low_index = 2u * block;
    uint high_index = low_index + 1u;
    PackedRegisterCycleRow low = low_index < row_count
        ? rows[low_index]
        : registers_read_write_empty_row();
    PackedRegisterCycleRow high = high_index < row_count
        ? rows[high_index]
        : registers_read_write_empty_row();
    SolinasFp128 low_increment = registers_read_write_increment(low);
    SolinasFp128 high_increment = registers_read_write_increment(high);
    output_increments[block] = ram_read_write_bind(
        low_increment, high_increment, challenge);

    uchar columns[6] = {
        low.rs1_index,
        low.rs2_index,
        low.rd_index,
        high.rs1_index,
        high.rs2_index,
        high.rd_index,
    };
    registers_read_write_sort_columns(columns);
    uint output_length = 0u;
    uchar prior = REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
    for (uint candidate = 0u; candidate < 6u; candidate++) {
        uchar column = columns[candidate];
        if (column == REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER || column == prior) {
            continue;
        }
        prior = column;
        RegistersReadWriteCell low_cell = registers_read_write_cell(low, column);
        RegistersReadWriteCell high_cell = registers_read_write_cell(high, column);
        SolinasFp128 low_value = registers_read_write_from_u64(
            low_cell.present ? low_cell.value : high_cell.previous);
        SolinasFp128 high_value = registers_read_write_from_u64(
            high_cell.present ? high_cell.value : low_cell.next);
        uint output = registers_read_write_state_index(
            block, output_capacity, output_length);
        output_columns[output] = column;
        output_previous[output] = low_cell.present
            ? low_cell.previous
            : high_cell.previous;
        output_next[output] = high_cell.present
            ? high_cell.next
            : low_cell.next;
        output_values[output] = ram_read_write_bind(
            low_value, high_value, challenge);
        output_ra[output] = (ushort)(
            (high_cell.read_mask << 2u) | low_cell.read_mask);
        output_wa[output] = (uchar)(
            (uint(high_cell.write) << 1u) | uint(low_cell.write));
        output_length += 1u;
    }
    return output_length;
}

inline uint registers_read_write_bind_indexed_block(
    uint output_block,
    device const uchar* input_lengths,
    device const uchar* input_columns,
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
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params)
{
    uint low_block = 2u * output_block;
    uint high_block = low_block + 1u;
    uint low_length = input_lengths[low_block];
    uint high_length = input_lengths[high_block];
    uint low_slot = 0u;
    uint high_slot = 0u;
    uint output_length = 0u;
    while (low_slot < low_length || high_slot < high_length) {
        uint low_index = registers_read_write_state_index(
            low_block, params.input_capacity, low_slot);
        uint high_index = registers_read_write_state_index(
            high_block, params.input_capacity, high_slot);
        uchar low_column = low_slot < low_length
            ? input_columns[low_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uchar high_column = high_slot < high_length
            ? input_columns[high_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        bool take_low = low_column <= high_column;
        bool take_high = high_column <= low_column;
        RegistersReadWriteIndexedStateCell low_cell;
        RegistersReadWriteIndexedStateCell high_cell;
        if (take_low) {
            low_cell = registers_read_write_load_indexed(
                low_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            low_slot += 1u;
        }
        if (take_high) {
            high_cell = registers_read_write_load_indexed(
                high_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            high_slot += 1u;
        }
        SolinasFp128 low_value = take_low
            ? low_cell.value
            : registers_read_write_from_u64(high_cell.previous);
        SolinasFp128 high_value = take_high
            ? high_cell.value
            : registers_read_write_from_u64(low_cell.next);
        uint output = registers_read_write_state_index(
            output_block, params.output_capacity, output_length);
        output_columns[output] = min(low_column, high_column);
        output_previous[output] = take_low
            ? low_cell.previous
            : high_cell.previous;
        output_next[output] = take_high
            ? high_cell.next
            : low_cell.next;
        output_values[output] = ram_read_write_bind(
            low_value, high_value, challenge);
        uint low_ra = take_low ? uint(low_cell.ra) : 0u;
        uint high_ra = take_high ? uint(high_cell.ra) : 0u;
        output_ra[output] = (ushort)((high_ra << params.ra_lut_bits) | low_ra);
        uint low_wa = take_low ? uint(low_cell.wa) : 0u;
        uint high_wa = take_high ? uint(high_cell.wa) : 0u;
        output_wa[output] = (uchar)((high_wa << params.wa_lut_bits) | low_wa);
        output_length += 1u;
    }
    return output_length;
}

inline uint registers_read_write_bind_transition_block(
    uint output_block,
    device const uchar* input_lengths,
    device const uchar* input_columns,
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
    constant RegistersReadWriteSequenceParams& params)
{
    uint low_block = 2u * output_block;
    uint high_block = low_block + 1u;
    uint low_length = input_lengths[low_block];
    uint high_length = input_lengths[high_block];
    uint low_slot = 0u;
    uint high_slot = 0u;
    uint output_length = 0u;
    while (low_slot < low_length || high_slot < high_length) {
        uint low_index = registers_read_write_state_index(
            low_block, params.input_capacity, low_slot);
        uint high_index = registers_read_write_state_index(
            high_block, params.input_capacity, high_slot);
        uchar low_column = low_slot < low_length
            ? input_columns[low_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uchar high_column = high_slot < high_length
            ? input_columns[high_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        bool take_low = low_column <= high_column;
        bool take_high = high_column <= low_column;
        RegistersReadWriteIndexedStateCell low_cell;
        RegistersReadWriteIndexedStateCell high_cell;
        if (take_low) {
            low_cell = registers_read_write_load_indexed(
                low_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            low_slot += 1u;
        }
        if (take_high) {
            high_cell = registers_read_write_load_indexed(
                high_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            high_slot += 1u;
        }
        SolinasFp128 low_value = take_low
            ? low_cell.value
            : registers_read_write_from_u64(high_cell.previous);
        SolinasFp128 high_value = take_high
            ? high_cell.value
            : registers_read_write_from_u64(low_cell.next);
        uint output = registers_read_write_state_index(
            output_block, params.output_capacity, output_length);
        output_columns[output] = min(low_column, high_column);
        output_previous[output] = take_low
            ? low_cell.previous
            : high_cell.previous;
        output_next[output] = take_high
            ? high_cell.next
            : low_cell.next;
        output_values[output] = ram_read_write_bind(
            low_value, high_value, challenge);
        output_ra[output] = ram_read_write_bind(
            take_low ? ra_lut[low_cell.ra] : solinas_zero(),
            take_high ? ra_lut[high_cell.ra] : solinas_zero(),
            challenge);
        output_wa[output] = ram_read_write_bind(
            take_low ? wa_lut[low_cell.wa] : solinas_zero(),
            take_high ? wa_lut[high_cell.wa] : solinas_zero(),
            challenge);
        output_length += 1u;
    }
    return output_length;
}

inline uint registers_read_write_bind_direct_block(
    uint output_block,
    device const uchar* input_lengths,
    device const uchar* input_columns,
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
    SolinasFp128 challenge,
    constant RegistersReadWriteSequenceParams& params)
{
    uint low_block = 2u * output_block;
    uint high_block = low_block + 1u;
    uint low_length = input_lengths[low_block];
    uint high_length = input_lengths[high_block];
    uint low_slot = 0u;
    uint high_slot = 0u;
    uint output_length = 0u;
    while (low_slot < low_length || high_slot < high_length) {
        uint low_index = registers_read_write_state_index(
            low_block, params.input_capacity, low_slot);
        uint high_index = registers_read_write_state_index(
            high_block, params.input_capacity, high_slot);
        uchar low_column = low_slot < low_length
            ? input_columns[low_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uchar high_column = high_slot < high_length
            ? input_columns[high_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        bool take_low = low_column <= high_column;
        bool take_high = high_column <= low_column;
        RegistersReadWriteDirectStateCell low_cell;
        RegistersReadWriteDirectStateCell high_cell;
        if (take_low) {
            low_cell = registers_read_write_load_direct(
                low_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            low_slot += 1u;
        }
        if (take_high) {
            high_cell = registers_read_write_load_direct(
                high_index,
                input_previous,
                input_next,
                input_values,
                input_ra,
                input_wa);
            high_slot += 1u;
        }
        SolinasFp128 low_value = take_low
            ? low_cell.value
            : registers_read_write_from_u64(high_cell.previous);
        SolinasFp128 high_value = take_high
            ? high_cell.value
            : registers_read_write_from_u64(low_cell.next);
        uint output = registers_read_write_state_index(
            output_block, params.output_capacity, output_length);
        output_columns[output] = min(low_column, high_column);
        output_previous[output] = take_low
            ? low_cell.previous
            : high_cell.previous;
        output_next[output] = take_high
            ? high_cell.next
            : low_cell.next;
        output_values[output] = ram_read_write_bind(
            low_value, high_value, challenge);
        output_ra[output] = ram_read_write_bind(
            take_low ? low_cell.ra : solinas_zero(),
            take_high ? high_cell.ra : solinas_zero(),
            challenge);
        output_wa[output] = ram_read_write_bind(
            take_low ? low_cell.wa : solinas_zero(),
            take_high ? high_cell.wa : solinas_zero(),
            challenge);
        output_length += 1u;
    }
    return output_length;
}

inline RegistersReadWriteMessageTerm registers_read_write_indexed_message(
    uint parent,
    device const uchar* lengths,
    device const uchar* columns,
    device const ulong* previous,
    device const ulong* next,
    device const SolinasFp128* values,
    device const ushort* ra,
    device const uchar* wa,
    device const SolinasFp128* increments,
    device const SolinasFp128* ra_lut,
    device const SolinasFp128* wa_lut,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    constant RegistersReadWriteSequenceParams& params)
{
    uint low_block = 2u * parent;
    uint high_block = low_block + 1u;
    uint low_length = lengths[low_block];
    uint high_length = lengths[high_block];
    uint low_slot = 0u;
    uint high_slot = 0u;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    SolinasFp128 inc_zero = increments[low_block];
    SolinasFp128 inc_slope = solinas_sub(increments[high_block], inc_zero);
    while (low_slot < low_length || high_slot < high_length) {
        uint low_index = registers_read_write_state_index(
            low_block, params.output_capacity, low_slot);
        uint high_index = registers_read_write_state_index(
            high_block, params.output_capacity, high_slot);
        uchar low_column = low_slot < low_length
            ? columns[low_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uchar high_column = high_slot < high_length
            ? columns[high_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        bool take_low = low_column <= high_column;
        bool take_high = high_column <= low_column;
        RegistersReadWriteIndexedStateCell low_cell;
        RegistersReadWriteIndexedStateCell high_cell;
        if (take_low) {
            low_cell = registers_read_write_load_indexed(
                low_index, previous, next, values, ra, wa);
            low_slot += 1u;
        }
        if (take_high) {
            high_cell = registers_read_write_load_indexed(
                high_index, previous, next, values, ra, wa);
            high_slot += 1u;
        }
        SolinasFp128 ra_zero = take_low ? ra_lut[low_cell.ra] : solinas_zero();
        SolinasFp128 ra_slope = take_high
            ? solinas_sub(ra_lut[high_cell.ra], ra_zero)
            : solinas_sub(solinas_zero(), ra_zero);
        SolinasFp128 wa_zero = take_low ? wa_lut[low_cell.wa] : solinas_zero();
        SolinasFp128 wa_slope = take_high
            ? solinas_sub(wa_lut[high_cell.wa], wa_zero)
            : solinas_sub(solinas_zero(), wa_zero);
        SolinasFp128 val_zero = take_low
            ? low_cell.value
            : registers_read_write_from_u64(high_cell.previous);
        SolinasFp128 val_high = take_high
            ? high_cell.value
            : registers_read_write_from_u64(low_cell.next);
        SolinasFp128 val_slope = solinas_sub(val_high, val_zero);
        q_zero = solinas_add(
            q_zero,
            solinas_add(
                solinas_mul_wide(ra_zero, val_zero),
                solinas_mul_wide(wa_zero, solinas_add(val_zero, inc_zero))));
        q_infinity = solinas_add(
            q_infinity,
            solinas_add(
                solinas_mul_wide(ra_slope, val_slope),
                solinas_mul_wide(
                    wa_slope,
                    solinas_add(val_slope, inc_slope))));
    }
    SolinasFp128 head = registers_read_write_head(
        parent, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline RegistersReadWriteMessageTerm registers_read_write_direct_message(
    uint parent,
    device const uchar* lengths,
    device const uchar* columns,
    device const ulong* previous,
    device const ulong* next,
    device const SolinasFp128* values,
    device const SolinasFp128* ra,
    device const SolinasFp128* wa,
    device const SolinasFp128* increments,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    constant RegistersReadWriteSequenceParams& params)
{
    uint low_block = 2u * parent;
    uint high_block = low_block + 1u;
    uint low_length = lengths[low_block];
    uint high_length = lengths[high_block];
    uint low_slot = 0u;
    uint high_slot = 0u;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    SolinasFp128 inc_zero = increments[low_block];
    SolinasFp128 inc_slope = solinas_sub(increments[high_block], inc_zero);
    while (low_slot < low_length || high_slot < high_length) {
        uint low_index = registers_read_write_state_index(
            low_block, params.output_capacity, low_slot);
        uint high_index = registers_read_write_state_index(
            high_block, params.output_capacity, high_slot);
        uchar low_column = low_slot < low_length
            ? columns[low_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        uchar high_column = high_slot < high_length
            ? columns[high_index]
            : REGISTERS_READ_WRITE_SEQUENCE_NO_REGISTER;
        bool take_low = low_column <= high_column;
        bool take_high = high_column <= low_column;
        RegistersReadWriteDirectStateCell low_cell;
        RegistersReadWriteDirectStateCell high_cell;
        if (take_low) {
            low_cell = registers_read_write_load_direct(
                low_index, previous, next, values, ra, wa);
            low_slot += 1u;
        }
        if (take_high) {
            high_cell = registers_read_write_load_direct(
                high_index, previous, next, values, ra, wa);
            high_slot += 1u;
        }
        SolinasFp128 ra_zero = take_low ? low_cell.ra : solinas_zero();
        SolinasFp128 ra_slope = take_high
            ? solinas_sub(high_cell.ra, ra_zero)
            : solinas_sub(solinas_zero(), ra_zero);
        SolinasFp128 wa_zero = take_low ? low_cell.wa : solinas_zero();
        SolinasFp128 wa_slope = take_high
            ? solinas_sub(high_cell.wa, wa_zero)
            : solinas_sub(solinas_zero(), wa_zero);
        SolinasFp128 val_zero = take_low
            ? low_cell.value
            : registers_read_write_from_u64(high_cell.previous);
        SolinasFp128 val_high = take_high
            ? high_cell.value
            : registers_read_write_from_u64(low_cell.next);
        SolinasFp128 val_slope = solinas_sub(val_high, val_zero);
        q_zero = solinas_add(
            q_zero,
            solinas_add(
                solinas_mul_wide(ra_zero, val_zero),
                solinas_mul_wide(wa_zero, solinas_add(val_zero, inc_zero))));
        q_infinity = solinas_add(
            q_infinity,
            solinas_add(
                solinas_mul_wide(ra_slope, val_slope),
                solinas_mul_wide(
                    wa_slope,
                    solinas_add(val_slope, inc_slope))));
    }
    SolinasFp128 head = registers_read_write_head(
        parent, e_in, e_out, params.e_in_length);
    RegistersReadWriteMessageTerm result;
    result.q_zero = solinas_mul_wide(head, q_zero);
    result.q_infinity = solinas_mul_wide(head, q_infinity);
    return result;
}

inline void registers_read_write_store_partial(
    device SolinasFp128* partials,
    threadgroup SolinasFp128* zero_sums,
    threadgroup SolinasFp128* infinity_sums,
    uint output_stride,
    uint group,
    uint lane,
    uint simdgroup,
    SolinasFp128 q_zero,
    SolinasFp128 q_infinity)
{
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[group] = q_zero;
            partials[output_stride + group] = q_infinity;
        }
    }
}

kernel void solinas_registers_read_write_bootstrap(
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
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    if (work < params.work_items) {
        uint low_block = 2u * work;
        uint high_block = low_block + 1u;
        output_lengths[low_block] = (uchar)registers_read_write_bind_source_block(
            low_block,
            rows,
            params.row_count,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            challenge,
            params.output_capacity);
        output_lengths[high_block] = (uchar)registers_read_write_bind_source_block(
            high_block,
            rows,
            params.row_count,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            challenge,
            params.output_capacity);
        RegistersReadWriteMessageTerm term = registers_read_write_indexed_message(
            work,
            output_lengths,
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
            params);
        q_zero = term.q_zero;
        q_infinity = term.q_infinity;
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        q_zero,
        q_infinity);
}

kernel void solinas_registers_read_write_indexed_bind_message(
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
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    if (work < params.work_items) {
        uint low_block = 2u * work;
        uint high_block = low_block + 1u;
        output_lengths[low_block] = (uchar)registers_read_write_bind_indexed_block(
            low_block,
            input_lengths,
            input_columns,
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
            challenge,
            params);
        output_lengths[high_block] = (uchar)registers_read_write_bind_indexed_block(
            high_block,
            input_lengths,
            input_columns,
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
            challenge,
            params);
        output_increments[low_block] = ram_read_write_bind(
            input_increments[2u * low_block],
            input_increments[2u * low_block + 1u],
            challenge);
        output_increments[high_block] = ram_read_write_bind(
            input_increments[2u * high_block],
            input_increments[2u * high_block + 1u],
            challenge);
        RegistersReadWriteMessageTerm term = registers_read_write_indexed_message(
            work,
            output_lengths,
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
            params);
        q_zero = term.q_zero;
        q_infinity = term.q_infinity;
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        q_zero,
        q_infinity);
}

kernel void solinas_registers_read_write_transition_bind_message(
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
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    if (work < params.work_items) {
        uint low_block = 2u * work;
        uint high_block = low_block + 1u;
        output_lengths[low_block] = (uchar)registers_read_write_bind_transition_block(
            low_block,
            input_lengths,
            input_columns,
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
            params);
        output_lengths[high_block] = (uchar)registers_read_write_bind_transition_block(
            high_block,
            input_lengths,
            input_columns,
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
            params);
        output_increments[low_block] = ram_read_write_bind(
            input_increments[2u * low_block],
            input_increments[2u * low_block + 1u],
            challenge);
        output_increments[high_block] = ram_read_write_bind(
            input_increments[2u * high_block],
            input_increments[2u * high_block + 1u],
            challenge);
        RegistersReadWriteMessageTerm term = registers_read_write_direct_message(
            work,
            output_lengths,
            output_columns,
            output_previous,
            output_next,
            output_values,
            output_ra,
            output_wa,
            output_increments,
            e_in,
            e_out,
            params);
        q_zero = term.q_zero;
        q_infinity = term.q_infinity;
    }
    registers_read_write_store_partial(
        partials,
        zero_sums,
        infinity_sums,
        params.output_stride,
        group,
        lane,
        simdgroup,
        q_zero,
        q_infinity);
}

kernel void solinas_registers_read_write_direct_bind_message(
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
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 zero_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[REGISTERS_READ_WRITE_SEQUENCE_SIMDGROUPS];
    uint work = group * REGISTERS_READ_WRITE_SEQUENCE_THREADS + tid;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    if (work < params.work_items) {
        uint low_block = 2u * work;
        uint high_block = min(low_block + 1u, params.output_blocks - 1u);
        output_lengths[low_block] = (uchar)registers_read_write_bind_direct_block(
            low_block,
            input_lengths,
            input_columns,
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
            challenge,
            params);
        output_increments[low_block] = ram_read_write_bind(
            input_increments[2u * low_block],
            input_increments[2u * low_block + 1u],
            challenge);
        if (high_block != low_block) {
            output_lengths[high_block] = (uchar)registers_read_write_bind_direct_block(
                high_block,
                input_lengths,
                input_columns,
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
                challenge,
                params);
            output_increments[high_block] = ram_read_write_bind(
                input_increments[2u * high_block],
                input_increments[2u * high_block + 1u],
                challenge);
        }
        if (params.emit_message != 0u) {
            RegistersReadWriteMessageTerm term = registers_read_write_direct_message(
                work,
                output_lengths,
                output_columns,
                output_previous,
                output_next,
                output_values,
                output_ra,
                output_wa,
                output_increments,
                e_in,
                e_out,
                params);
            q_zero = term.q_zero;
            q_infinity = term.q_infinity;
        }
    }
    if (params.emit_message != 0u) {
        registers_read_write_store_partial(
            partials,
            zero_sums,
            infinity_sums,
            params.output_stride,
            group,
            lane,
            simdgroup,
            q_zero,
            q_infinity);
    }
}
