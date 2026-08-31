// Concatenate after offset-specialized fp128.metal and simd_reduce.metal.

#define RAM_READ_WRITE_HOT_CHUNK_SIZE 4096u
#define RAM_READ_WRITE_SIMD_WIDTH 32u
#define RAM_READ_WRITE_HOT_THREADS 256u
#define RAM_READ_WRITE_CYCLE_THREADS 256u
#define RAM_READ_WRITE_CYCLE_SIMDGROUPS \
    (RAM_READ_WRITE_CYCLE_THREADS / RAM_READ_WRITE_SIMD_WIDTH)
#define RAM_READ_WRITE_BOUNDED_SIMDGROUPS \
    (RAM_READ_WRITE_HOT_THREADS / RAM_READ_WRITE_SIMD_WIDTH)
#define RAM_READ_WRITE_HOT_SIMDGROUPS \
    (RAM_READ_WRITE_HOT_THREADS / RAM_READ_WRITE_SIMD_WIDTH)

struct RamReadWriteSegment {
    uint offset;
    uint length;
    uint capacity;
    uint aux_offset;
};

struct RamReadWritePhaseParams {
    uint work_items;
    uint output_stride;
    uint e_in_length;
    uint bind;
    uint emit_message;
    uint hot_source_aux;
    uint hot_threshold;
    uint source_initial;
};

struct RamReadWriteHotChunk {
    uint hot_index;
    uint local_offset;
};

struct RamReadWriteHotSegment {
    uint segment_index;
    uint first_chunk;
    uint chunk_count;
    uint aux_offset;
};

struct RamReadWriteMessageTerm {
    SolinasFp128 q_zero;
    SolinasFp128 q_infinity;
};

struct RamReadWriteReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

struct RamReadWriteAccessRecord {
    uint cycle;
    uint address;
    ulong pre_value;
    ulong post_value;
};

struct RamReadWriteInitialScatterParams {
    uint records;
    uint address_count;
    uint cycle_offset;
    uint reserved;
};

struct RamReadWritePrefixParams {
    uint records;
    uint output_stride;
    uint e_in_length;
    uint rounds_bound;
    uint bind;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

#define RAM_READ_WRITE_PREFIX_START 0x80000000u
#define RAM_READ_WRITE_PREFIX_BLOCK_MASK 0x7fffffffu

inline uint ram_read_write_prefix_table_index(uint cycle, uint rounds_bound) {
    return cycle & ((1u << rounds_bound) - 1u);
}

inline SolinasFp128 ram_read_write_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 ram_read_write_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline SolinasFp128 ram_read_write_head(
    uint parent,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    uint e_in_length)
{
    uint in_bits = ctz(e_in_length);
    return solinas_mul_wide(
        e_out[parent >> in_bits],
        e_in[parent & (e_in_length - 1u)]);
}

kernel void solinas_ram_read_write_initial_scatter(
    device const RamReadWriteAccessRecord* records [[buffer(0)]],
    device const uint* address_ranks [[buffer(1)]],
    device const uint* address_prefixes [[buffer(2)]],
    device uint* address_blocks [[buffer(3)]],
    device ulong* address_previous [[buffer(4)]],
    device ulong* address_next [[buffer(5)]],
    device uint* cycle_blocks [[buffer(6)]],
    device SolinasFp128* cycle_increments [[buffer(7)]],
    constant RamReadWriteInitialScatterParams& params [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.records) {
        return;
    }
    RamReadWriteAccessRecord record = records[index];
    if (record.address >= params.address_count) {
        return;
    }
    uint address_index = address_prefixes[record.address] + address_ranks[index];
    uint cycle_index = params.cycle_offset + index;
    address_blocks[address_index] = record.cycle;
    address_previous[address_index] = record.pre_value;
    address_next[address_index] = record.post_value;
    cycle_blocks[cycle_index] = record.cycle;
    cycle_increments[cycle_index] = solinas_sub(
        ram_read_write_from_u64(record.post_value),
        ram_read_write_from_u64(record.pre_value));
}

inline RamReadWriteMessageTerm ram_read_write_address_message_term(
    uint index,
    uint segment_begin,
    uint segment_end,
    device const uint* blocks,
    device const ulong* previous,
    device const ulong* next,
    device const SolinasFp128* values,
    device const SolinasFp128* ra,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    uint e_in_length,
    bool source_initial)
{
    RamReadWriteMessageTerm result = {
        solinas_zero(),
        solinas_zero(),
    };
    uint first_block = blocks[index];
    uint parent = first_block >> 1;
    bool leader = index == segment_begin
        || (blocks[index - 1u] >> 1) != parent;
    if (!leader) {
        return result;
    }
    SolinasFp128 first_value;
    SolinasFp128 first_ra;
    if (source_initial) {
        first_value = ram_read_write_from_u64(previous[index]);
        first_ra = ram_read_write_from_u64(1u);
    } else {
        first_value = values[index];
        first_ra = ra[index];
    }
    bool paired = index + 1u < segment_end
        && (blocks[index + 1u] >> 1) == parent;
    SolinasFp128 local_zero;
    SolinasFp128 local_infinity;
    if (paired) {
        SolinasFp128 second_value;
        SolinasFp128 second_ra;
        if (source_initial) {
            second_value = ram_read_write_from_u64(previous[index + 1u]);
            second_ra = ram_read_write_from_u64(1u);
        } else {
            second_value = values[index + 1u];
            second_ra = ra[index + 1u];
        }
        SolinasFp128 value_slope = solinas_sub(second_value, first_value);
        SolinasFp128 ra_slope = solinas_sub(second_ra, first_ra);
        local_zero = solinas_mul_wide(first_ra, first_value);
        local_infinity = solinas_mul_wide(ra_slope, value_slope);
    } else if ((first_block & 1u) == 0u) {
        SolinasFp128 value_slope = solinas_sub(
            ram_read_write_from_u64(next[index]),
            first_value);
        local_zero = solinas_mul_wide(first_ra, first_value);
        local_infinity = solinas_mul_wide(
            solinas_sub(solinas_zero(), first_ra),
            value_slope);
    } else {
        SolinasFp128 value_slope = solinas_sub(
            first_value,
            ram_read_write_from_u64(previous[index]));
        local_zero = solinas_zero();
        local_infinity = solinas_mul_wide(first_ra, value_slope);
    }
    SolinasFp128 head = ram_read_write_head(
        parent, e_in, e_out, e_in_length);
    result.q_zero = solinas_mul_wide(head, local_zero);
    result.q_infinity = solinas_mul_wide(head, local_infinity);
    return result;
}

kernel void solinas_ram_read_write_address(
    device RamReadWriteSegment* segments [[buffer(0)]],
    device uint* blocks [[buffer(1)]],
    device ulong* previous [[buffer(2)]],
    device ulong* next [[buffer(3)]],
    device SolinasFp128* values [[buffer(4)]],
    device SolinasFp128* ra [[buffer(5)]],
    device const SolinasFp128* e_in [[buffer(6)]],
    device const SolinasFp128* e_out [[buffer(7)]],
    device SolinasFp128* partials [[buffer(8)]],
    constant SolinasFp128& challenge [[buffer(9)]],
    constant RamReadWritePhaseParams& params [[buffer(10)]],
    uint segment_index [[thread_position_in_grid]])
{
    if (segment_index >= params.work_items) {
        return;
    }
    RamReadWriteSegment segment = segments[segment_index];
    if (segment.capacity > params.hot_threshold) {
        return;
    }
    uint begin = segment.offset;
    uint end = begin + segment.length;

    if (params.bind != 0u) {
        uint read = begin;
        uint write = begin;
        while (read < end) {
            uint first_block = blocks[read];
            uint parent = first_block >> 1;
            ulong first_previous = previous[read];
            ulong first_next = next[read];
            SolinasFp128 first_value;
            SolinasFp128 first_ra;
            if (params.source_initial != 0u) {
                first_value = ram_read_write_from_u64(first_previous);
                first_ra = ram_read_write_from_u64(1u);
            } else {
                first_value = values[read];
                first_ra = ra[read];
            }
            bool paired = read + 1u < end && (blocks[read + 1u] >> 1) == parent;

            ulong output_previous;
            ulong output_next;
            SolinasFp128 output_value;
            SolinasFp128 output_ra;
            if (paired) {
                ulong second_next = next[read + 1u];
                SolinasFp128 second_value;
                SolinasFp128 second_ra;
                if (params.source_initial != 0u) {
                    second_value = ram_read_write_from_u64(previous[read + 1u]);
                    second_ra = ram_read_write_from_u64(1u);
                } else {
                    second_value = values[read + 1u];
                    second_ra = ra[read + 1u];
                }
                output_previous = first_previous;
                output_next = second_next;
                output_value = ram_read_write_bind(
                    first_value, second_value, challenge);
                output_ra = ram_read_write_bind(first_ra, second_ra, challenge);
                read += 2u;
            } else if ((first_block & 1u) == 0u) {
                output_previous = first_previous;
                output_next = first_next;
                output_value = ram_read_write_bind(
                    first_value,
                    ram_read_write_from_u64(first_next),
                    challenge);
                output_ra = solinas_mul_wide(
                    solinas_sub(solinas_zero(), challenge),
                    first_ra);
                output_ra = solinas_add(first_ra, output_ra);
                read += 1u;
            } else {
                output_previous = first_previous;
                output_next = first_next;
                output_value = ram_read_write_bind(
                    ram_read_write_from_u64(first_previous),
                    first_value,
                    challenge);
                output_ra = solinas_mul_wide(challenge, first_ra);
                read += 1u;
            }

            blocks[write] = parent;
            previous[write] = output_previous;
            next[write] = output_next;
            values[write] = output_value;
            ra[write] = output_ra;
            write += 1u;
        }
        segment.length = write - begin;
        segments[segment_index] = segment;
        end = write;
    }

    if (params.emit_message == 0u) {
        return;
    }
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    uint read = begin;
    bool message_initial = params.source_initial != 0u && params.bind == 0u;
    while (read < end) {
        uint first_block = blocks[read];
        uint parent = first_block >> 1;
        SolinasFp128 first_value;
        SolinasFp128 first_ra;
        if (message_initial) {
            first_value = ram_read_write_from_u64(previous[read]);
            first_ra = ram_read_write_from_u64(1u);
        } else {
            first_value = values[read];
            first_ra = ra[read];
        }
        bool paired = read + 1u < end && (blocks[read + 1u] >> 1) == parent;
        SolinasFp128 local_zero;
        SolinasFp128 local_infinity;
        if (paired) {
            SolinasFp128 second_value;
            SolinasFp128 second_ra;
            if (message_initial) {
                second_value = ram_read_write_from_u64(previous[read + 1u]);
                second_ra = ram_read_write_from_u64(1u);
            } else {
                second_value = values[read + 1u];
                second_ra = ra[read + 1u];
            }
            SolinasFp128 value_slope = solinas_sub(second_value, first_value);
            SolinasFp128 ra_slope = solinas_sub(second_ra, first_ra);
            local_zero = solinas_mul_wide(first_ra, first_value);
            local_infinity = solinas_mul_wide(ra_slope, value_slope);
            read += 2u;
        } else if ((first_block & 1u) == 0u) {
            SolinasFp128 value_slope = solinas_sub(
                ram_read_write_from_u64(next[read]),
                first_value);
            local_zero = solinas_mul_wide(first_ra, first_value);
            local_infinity = solinas_mul_wide(
                solinas_sub(solinas_zero(), first_ra),
                value_slope);
            read += 1u;
        } else {
            SolinasFp128 value_slope = solinas_sub(
                first_value,
                ram_read_write_from_u64(previous[read]));
            local_zero = solinas_zero();
            local_infinity = solinas_mul_wide(first_ra, value_slope);
            read += 1u;
        }
        SolinasFp128 head = ram_read_write_head(
            parent, e_in, e_out, params.e_in_length);
        q_zero = solinas_add(q_zero, solinas_mul_wide(head, local_zero));
        q_infinity = solinas_add(
            q_infinity,
            solinas_mul_wide(head, local_infinity));
    }
    partials[segment_index] = q_zero;
    partials[params.output_stride + segment_index] = q_infinity;
}

kernel void solinas_ram_read_write_address_bounded(
    device const uint* bounded_segments [[buffer(0)]],
    device RamReadWriteSegment* segments [[buffer(1)]],
    device uint* blocks [[buffer(2)]],
    device ulong* previous [[buffer(3)]],
    device ulong* next [[buffer(4)]],
    device SolinasFp128* values [[buffer(5)]],
    device SolinasFp128* ra [[buffer(6)]],
    device const SolinasFp128* e_in [[buffer(7)]],
    device const SolinasFp128* e_out [[buffer(8)]],
    device SolinasFp128* partials [[buffer(9)]],
    constant SolinasFp128& challenge [[buffer(10)]],
    constant RamReadWritePhaseParams& params [[buffer(11)]],
    uint bounded_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (bounded_index >= params.work_items) {
        return;
    }
    uint segment_index = bounded_segments[bounded_index];
    RamReadWriteSegment segment = segments[segment_index];
    uint begin = segment.offset;
    uint end = begin + segment.length;
    threadgroup uint group_offsets[RAM_READ_WRITE_BOUNDED_SIMDGROUPS];
    threadgroup uint running;
    threadgroup uint prior_block;
    threadgroup uint batch_last_block;
    threadgroup SolinasFp128 zero_sums[RAM_READ_WRITE_BOUNDED_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[RAM_READ_WRITE_BOUNDED_SIMDGROUPS];

    if (tid == 0u) {
        running = 0u;
        prior_block = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.bind != 0u) {
        uint source_length = segment.length;
        bool source_initial = params.source_initial != 0u;
        for (uint batch = 0u; batch < source_length; batch += threads) {
            uint local_index = batch + tid;
            uint index = begin + local_index;
            bool valid = local_index < source_length;
            uint first_block = valid ? blocks[index] : 0u;
            uint parent = first_block >> 1;
            uint previous_block = 0u;
            if (valid && local_index != 0u) {
                previous_block = tid == 0u ? prior_block : blocks[index - 1u];
            }
            bool leader = valid
                && (local_index == 0u || (previous_block >> 1) != parent);
            ulong output_previous = 0u;
            ulong output_next = 0u;
            SolinasFp128 output_value = solinas_zero();
            SolinasFp128 output_ra = solinas_zero();
            if (leader) {
                ulong first_previous = previous[index];
                ulong first_next = next[index];
                SolinasFp128 first_value = source_initial
                    ? ram_read_write_from_u64(first_previous)
                    : values[index];
                SolinasFp128 first_ra = source_initial
                    ? ram_read_write_from_u64(1u)
                    : ra[index];
                bool paired = local_index + 1u < source_length
                    && (blocks[index + 1u] >> 1) == parent;
                if (paired) {
                    SolinasFp128 second_value = source_initial
                        ? ram_read_write_from_u64(previous[index + 1u])
                        : values[index + 1u];
                    SolinasFp128 second_ra = source_initial
                        ? ram_read_write_from_u64(1u)
                        : ra[index + 1u];
                    output_previous = first_previous;
                    output_next = next[index + 1u];
                    output_value = ram_read_write_bind(
                        first_value, second_value, challenge);
                    output_ra = ram_read_write_bind(
                        first_ra, second_ra, challenge);
                } else if ((first_block & 1u) == 0u) {
                    output_previous = first_previous;
                    output_next = first_next;
                    output_value = ram_read_write_bind(
                        first_value,
                        ram_read_write_from_u64(first_next),
                        challenge);
                    output_ra = solinas_mul_wide(
                        solinas_sub(solinas_zero(), challenge),
                        first_ra);
                    output_ra = solinas_add(first_ra, output_ra);
                } else {
                    output_previous = first_previous;
                    output_next = first_next;
                    output_value = ram_read_write_bind(
                        ram_read_write_from_u64(first_previous),
                        first_value,
                        challenge);
                    output_ra = solinas_mul_wide(challenge, first_ra);
                }
            }

            uint batch_length = min(threads, source_length - batch);
            if (tid + 1u == batch_length) {
                batch_last_block = first_block;
            }
            uint flag = uint(leader);
            uint local_offset = simd_prefix_exclusive_sum(flag);
            uint simd_count = simd_sum(flag);
            if (lane == 0u) {
                group_offsets[simdgroup] = simd_count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint cursor = running;
                for (uint group = 0u; group < RAM_READ_WRITE_BOUNDED_SIMDGROUPS; group++) {
                    uint count = group_offsets[group];
                    group_offsets[group] = cursor;
                    cursor += count;
                }
                running = cursor;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (leader) {
                uint output = begin + group_offsets[simdgroup] + local_offset;
                blocks[output] = parent;
                previous[output] = output_previous;
                next[output] = output_next;
                values[output] = output_value;
                ra[output] = output_ra;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (tid == 0u) {
                prior_block = batch_last_block;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            segment.length = running;
            segments[segment_index] = segment;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        end = begin + running;
    }

    if (params.emit_message == 0u) {
        return;
    }
    bool message_initial = params.source_initial != 0u && params.bind == 0u;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    for (uint index = begin + tid; index < end; index += threads) {
        RamReadWriteMessageTerm term = ram_read_write_address_message_term(
            index,
            begin,
            end,
            blocks,
            previous,
            next,
            values,
            ra,
            e_in,
            e_out,
            params.e_in_length,
            message_initial);
        q_zero = solinas_add(q_zero, term.q_zero);
        q_infinity = solinas_add(q_infinity, term.q_infinity);
    }
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < RAM_READ_WRITE_BOUNDED_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < RAM_READ_WRITE_BOUNDED_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[segment_index] = q_zero;
            partials[params.output_stride + segment_index] = q_infinity;
        }
    }
}

kernel void solinas_ram_read_write_address_hot_count(
    device const RamReadWriteHotChunk* chunks [[buffer(0)]],
    device const RamReadWriteHotSegment* hot_segments [[buffer(1)]],
    device const RamReadWriteSegment* segments [[buffer(2)]],
    device const uint* primary_blocks [[buffer(3)]],
    device const uint* auxiliary_blocks [[buffer(4)]],
    device uint* chunk_counts [[buffer(5)]],
    constant RamReadWritePhaseParams& params [[buffer(6)]],
    uint chunk_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (chunk_index >= params.work_items) {
        return;
    }
    RamReadWriteHotChunk chunk = chunks[chunk_index];
    RamReadWriteHotSegment hot = hot_segments[chunk.hot_index];
    RamReadWriteSegment segment = segments[hot.segment_index];
    uint source_length = segment.length;
    if (chunk.local_offset >= source_length) {
        if (tid == 0u) {
            chunk_counts[chunk_index] = 0u;
        }
        return;
    }
    device const uint* blocks = params.hot_source_aux != 0u
        ? auxiliary_blocks
        : primary_blocks;
    uint source_begin = params.hot_source_aux != 0u
        ? hot.aux_offset
        : segment.offset;
    uint chunk_begin = source_begin + chunk.local_offset;
    uint chunk_end = min(
        chunk_begin + RAM_READ_WRITE_HOT_CHUNK_SIZE,
        source_begin + source_length);
    uint count = 0u;
    for (uint index = chunk_begin + tid; index < chunk_end; index += threads) {
        uint parent = blocks[index] >> 1;
        count += uint(index == source_begin || (blocks[index - 1u] >> 1) != parent);
    }
    count = simd_sum(count);
    threadgroup uint group_counts[RAM_READ_WRITE_HOT_SIMDGROUPS];
    if (lane == 0u) {
        group_counts[simdgroup] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint total = 0u;
        for (uint group = 0u; group < threads / RAM_READ_WRITE_SIMD_WIDTH; group++) {
            total += group_counts[group];
        }
        chunk_counts[chunk_index] = total;
    }
}

kernel void solinas_ram_read_write_address_hot_prefix(
    device const RamReadWriteHotSegment* hot_segments [[buffer(0)]],
    device RamReadWriteSegment* segments [[buffer(1)]],
    device const uint* chunk_counts [[buffer(2)]],
    device uint* chunk_offsets [[buffer(3)]],
    device uint* source_lengths [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant RamReadWritePhaseParams& params [[buffer(6)]],
    uint hot_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (hot_index >= params.work_items) {
        return;
    }
    RamReadWriteHotSegment hot = hot_segments[hot_index];
    RamReadWriteSegment segment = segments[hot.segment_index];
    threadgroup uint group_offsets[RAM_READ_WRITE_HOT_SIMDGROUPS];
    threadgroup uint running;
    if (tid == 0u) {
        running = 0u;
        source_lengths[hot_index] = segment.length;
        partials[hot.segment_index] = solinas_zero();
        partials[params.output_stride + hot.segment_index] = solinas_zero();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint batch = 0u; batch < hot.chunk_count; batch += threads) {
        uint local_chunk = batch + tid;
        bool valid = local_chunk < hot.chunk_count;
        uint count = valid ? chunk_counts[hot.first_chunk + local_chunk] : 0u;
        uint local_offset = simd_prefix_exclusive_sum(count);
        uint simd_count = simd_sum(count);
        if (lane == 0u) {
            group_offsets[simdgroup] = simd_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint cursor = running;
            uint simdgroups = threads / RAM_READ_WRITE_SIMD_WIDTH;
            for (uint group = 0u; group < simdgroups; group++) {
                uint count = group_offsets[group];
                group_offsets[group] = cursor;
                cursor += count;
            }
            running = cursor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            chunk_offsets[hot.first_chunk + local_chunk] =
                group_offsets[simdgroup] + local_offset;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        segment.length = running;
        segments[hot.segment_index] = segment;
    }
}

kernel void solinas_ram_read_write_address_hot_scatter(
    device const RamReadWriteHotChunk* chunks [[buffer(0)]],
    device const RamReadWriteHotSegment* hot_segments [[buffer(1)]],
    device const RamReadWriteSegment* segments [[buffer(2)]],
    device const uint* chunk_offsets [[buffer(3)]],
    device const uint* source_lengths [[buffer(4)]],
    device uint* primary_blocks [[buffer(5)]],
    device ulong* primary_previous [[buffer(6)]],
    device ulong* primary_next [[buffer(7)]],
    device SolinasFp128* primary_values [[buffer(8)]],
    device SolinasFp128* primary_ra [[buffer(9)]],
    device uint* auxiliary_blocks [[buffer(10)]],
    device ulong* auxiliary_previous [[buffer(11)]],
    device ulong* auxiliary_next [[buffer(12)]],
    device SolinasFp128* auxiliary_values [[buffer(13)]],
    device SolinasFp128* auxiliary_ra [[buffer(14)]],
    constant SolinasFp128& challenge [[buffer(15)]],
    constant RamReadWritePhaseParams& params [[buffer(16)]],
    uint chunk_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (chunk_index >= params.work_items) {
        return;
    }
    RamReadWriteHotChunk chunk = chunks[chunk_index];
    RamReadWriteHotSegment hot = hot_segments[chunk.hot_index];
    RamReadWriteSegment segment = segments[hot.segment_index];
    uint source_length = source_lengths[chunk.hot_index];
    if (chunk.local_offset >= source_length) {
        return;
    }
    device uint* source_blocks = params.hot_source_aux != 0u
        ? auxiliary_blocks
        : primary_blocks;
    device ulong* source_previous = params.hot_source_aux != 0u
        ? auxiliary_previous
        : primary_previous;
    device ulong* source_next = params.hot_source_aux != 0u
        ? auxiliary_next
        : primary_next;
    device SolinasFp128* source_values = params.hot_source_aux != 0u
        ? auxiliary_values
        : primary_values;
    device SolinasFp128* source_ra = params.hot_source_aux != 0u
        ? auxiliary_ra
        : primary_ra;
    device uint* destination_blocks = params.hot_source_aux != 0u
        ? primary_blocks
        : auxiliary_blocks;
    device ulong* destination_previous = params.hot_source_aux != 0u
        ? primary_previous
        : auxiliary_previous;
    device ulong* destination_next = params.hot_source_aux != 0u
        ? primary_next
        : auxiliary_next;
    device SolinasFp128* destination_values = params.hot_source_aux != 0u
        ? primary_values
        : auxiliary_values;
    device SolinasFp128* destination_ra = params.hot_source_aux != 0u
        ? primary_ra
        : auxiliary_ra;
    uint source_begin = params.hot_source_aux != 0u
        ? hot.aux_offset
        : segment.offset;
    uint destination_begin = params.hot_source_aux != 0u
        ? segment.offset
        : hot.aux_offset;
    uint source_end = source_begin + source_length;
    uint chunk_begin = source_begin + chunk.local_offset;
    uint chunk_end = min(
        chunk_begin + RAM_READ_WRITE_HOT_CHUNK_SIZE,
        source_end);
    threadgroup uint group_offsets[RAM_READ_WRITE_HOT_SIMDGROUPS];
    threadgroup uint chunk_cursor;
    if (tid == 0u) {
        chunk_cursor = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint batch = 0u; batch < chunk_end - chunk_begin; batch += threads) {
        uint index = chunk_begin + batch + tid;
        bool valid = index < chunk_end;
        uint first_block = valid ? source_blocks[index] : 0u;
        uint parent = first_block >> 1;
        bool leader = valid
            && (index == source_begin
                || (source_blocks[index - 1u] >> 1) != parent);
        ulong output_previous = 0u;
        ulong output_next = 0u;
        SolinasFp128 output_value = solinas_zero();
        SolinasFp128 output_ra = solinas_zero();

        if (leader) {
            ulong first_previous = source_previous[index];
            ulong first_next = source_next[index];
            SolinasFp128 first_value;
            SolinasFp128 first_ra;
            if (params.source_initial != 0u) {
                first_value = ram_read_write_from_u64(first_previous);
                first_ra = ram_read_write_from_u64(1u);
            } else {
                first_value = source_values[index];
                first_ra = source_ra[index];
            }
            bool paired = index + 1u < source_end
                && (source_blocks[index + 1u] >> 1) == parent;
            output_previous = first_previous;
            output_next = first_next;
            if (paired) {
                output_next = source_next[index + 1u];
                SolinasFp128 second_value;
                SolinasFp128 second_ra;
                if (params.source_initial != 0u) {
                    second_value = ram_read_write_from_u64(source_previous[index + 1u]);
                    second_ra = ram_read_write_from_u64(1u);
                } else {
                    second_value = source_values[index + 1u];
                    second_ra = source_ra[index + 1u];
                }
                output_value = ram_read_write_bind(
                    first_value, second_value, challenge);
                output_ra = ram_read_write_bind(
                    first_ra, second_ra, challenge);
            } else if ((first_block & 1u) == 0u) {
                output_value = ram_read_write_bind(
                    first_value,
                    ram_read_write_from_u64(first_next),
                    challenge);
                output_ra = solinas_add(
                    first_ra,
                    solinas_mul_wide(
                        solinas_sub(solinas_zero(), challenge),
                        first_ra));
            } else {
                output_value = ram_read_write_bind(
                    ram_read_write_from_u64(first_previous),
                    first_value,
                    challenge);
                output_ra = solinas_mul_wide(challenge, first_ra);
            }
        }

        uint flag = leader ? 1u : 0u;
        uint local_offset = simd_prefix_exclusive_sum(flag);
        uint simd_count = simd_sum(flag);
        if (lane == 0u) {
            group_offsets[simdgroup] = simd_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint cursor = chunk_cursor;
            for (uint group = 0u; group < threads / RAM_READ_WRITE_SIMD_WIDTH; group++) {
                uint count = group_offsets[group];
                group_offsets[group] = cursor;
                cursor += count;
            }
            chunk_cursor = cursor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (leader) {
            uint output = destination_begin
                + chunk_offsets[chunk_index]
                + group_offsets[simdgroup]
                + local_offset;
            destination_blocks[output] = parent;
            destination_previous[output] = output_previous;
            destination_next[output] = output_next;
            destination_values[output] = output_value;
            destination_ra[output] = output_ra;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void solinas_ram_read_write_address_hot_message(
    device const RamReadWriteHotChunk* chunks [[buffer(0)]],
    device const RamReadWriteHotSegment* hot_segments [[buffer(1)]],
    device const RamReadWriteSegment* segments [[buffer(2)]],
    device const uint* primary_blocks [[buffer(3)]],
    device const ulong* primary_previous [[buffer(4)]],
    device const ulong* primary_next [[buffer(5)]],
    device const SolinasFp128* primary_values [[buffer(6)]],
    device const SolinasFp128* primary_ra [[buffer(7)]],
    device const uint* auxiliary_blocks [[buffer(8)]],
    device const ulong* auxiliary_previous [[buffer(9)]],
    device const ulong* auxiliary_next [[buffer(10)]],
    device const SolinasFp128* auxiliary_values [[buffer(11)]],
    device const SolinasFp128* auxiliary_ra [[buffer(12)]],
    device const SolinasFp128* e_in [[buffer(13)]],
    device const SolinasFp128* e_out [[buffer(14)]],
    device SolinasFp128* partials [[buffer(15)]],
    constant RamReadWritePhaseParams& params [[buffer(16)]],
    uint chunk_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (chunk_index >= params.work_items) {
        return;
    }
    RamReadWriteHotChunk chunk = chunks[chunk_index];
    RamReadWriteHotSegment hot = hot_segments[chunk.hot_index];
    RamReadWriteSegment segment = segments[hot.segment_index];
    device const uint* blocks = params.hot_source_aux != 0u
        ? auxiliary_blocks
        : primary_blocks;
    device const ulong* previous = params.hot_source_aux != 0u
        ? auxiliary_previous
        : primary_previous;
    device const ulong* next = params.hot_source_aux != 0u
        ? auxiliary_next
        : primary_next;
    device const SolinasFp128* values = params.hot_source_aux != 0u
        ? auxiliary_values
        : primary_values;
    device const SolinasFp128* ra = params.hot_source_aux != 0u
        ? auxiliary_ra
        : primary_ra;
    uint segment_begin = params.hot_source_aux != 0u
        ? hot.aux_offset
        : segment.offset;
    uint segment_end = segment_begin + segment.length;
    uint chunk_begin = segment_begin + chunk.local_offset;
    uint chunk_end = min(
        chunk_begin + RAM_READ_WRITE_HOT_CHUNK_SIZE,
        segment_end);

    SolinasFp128 zero_a = solinas_zero();
    SolinasFp128 zero_b = solinas_zero();
    SolinasFp128 zero_c = solinas_zero();
    SolinasFp128 zero_d = solinas_zero();
    SolinasFp128 infinity_a = solinas_zero();
    SolinasFp128 infinity_b = solinas_zero();
    SolinasFp128 infinity_c = solinas_zero();
    SolinasFp128 infinity_d = solinas_zero();
    uint step = 4u * threads;
    for (uint base = chunk_begin + tid; base < chunk_end; base += step) {
        RamReadWriteMessageTerm term = ram_read_write_address_message_term(
            base,
            segment_begin,
            segment_end,
            blocks,
            previous,
            next,
            values,
            ra,
            e_in,
            e_out,
            params.e_in_length,
            params.source_initial != 0u);
        zero_a = solinas_add(zero_a, term.q_zero);
        infinity_a = solinas_add(infinity_a, term.q_infinity);

        uint index = base + threads;
        if (index < chunk_end) {
            term = ram_read_write_address_message_term(
                index,
                segment_begin,
                segment_end,
                blocks,
                previous,
                next,
                values,
                ra,
                e_in,
                e_out,
                params.e_in_length,
                params.source_initial != 0u);
            zero_b = solinas_add(zero_b, term.q_zero);
            infinity_b = solinas_add(infinity_b, term.q_infinity);
        }

        index += threads;
        if (index < chunk_end) {
            term = ram_read_write_address_message_term(
                index,
                segment_begin,
                segment_end,
                blocks,
                previous,
                next,
                values,
                ra,
                e_in,
                e_out,
                params.e_in_length,
                params.source_initial != 0u);
            zero_c = solinas_add(zero_c, term.q_zero);
            infinity_c = solinas_add(infinity_c, term.q_infinity);
        }

        index += threads;
        if (index < chunk_end) {
            term = ram_read_write_address_message_term(
                index,
                segment_begin,
                segment_end,
                blocks,
                previous,
                next,
                values,
                ra,
                e_in,
                e_out,
                params.e_in_length,
                params.source_initial != 0u);
            zero_d = solinas_add(zero_d, term.q_zero);
            infinity_d = solinas_add(infinity_d, term.q_infinity);
        }
    }
    SolinasFp128 q_zero = solinas_add(
        solinas_add(zero_a, zero_b),
        solinas_add(zero_c, zero_d));
    SolinasFp128 q_infinity = solinas_add(
        solinas_add(infinity_a, infinity_b),
        solinas_add(infinity_c, infinity_d));

    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    threadgroup SolinasFp128 zero_sums[RAM_READ_WRITE_HOT_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[RAM_READ_WRITE_HOT_SIMDGROUPS];
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < RAM_READ_WRITE_HOT_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < RAM_READ_WRITE_HOT_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[chunk_index] = q_zero;
            partials[params.output_stride + chunk_index] = q_infinity;
        }
    }
}

kernel void solinas_ram_read_write_cycle(
    device RamReadWriteSegment* segments [[buffer(0)]],
    device uint* blocks [[buffer(1)]],
    device SolinasFp128* hamming [[buffer(2)]],
    device SolinasFp128* increments [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant SolinasFp128& challenge [[buffer(7)]],
    constant RamReadWritePhaseParams& params [[buffer(8)]],
    uint segment_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (segment_index >= params.work_items) {
        return;
    }
    RamReadWriteSegment segment = segments[segment_index];
    uint begin = segment.offset;
    uint end = begin + segment.length;
    threadgroup uint group_offsets[RAM_READ_WRITE_CYCLE_SIMDGROUPS];
    threadgroup uint running;
    threadgroup uint prior_block;
    threadgroup uint batch_last_block;
    threadgroup SolinasFp128 zero_sums[RAM_READ_WRITE_CYCLE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[RAM_READ_WRITE_CYCLE_SIMDGROUPS];

    if (tid == 0u) {
        running = 0u;
        prior_block = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params.bind != 0u) {
        uint source_length = segment.length;
        bool source_initial = params.source_initial != 0u;
        for (uint batch = 0u; batch < source_length; batch += threads) {
            uint local_index = batch + tid;
            uint index = begin + local_index;
            bool valid = local_index < source_length;
            uint first_block = valid ? blocks[index] : 0u;
            uint parent = first_block >> 1;
            uint previous_block = 0u;
            if (valid && local_index != 0u) {
                if (tid == 0u) {
                    previous_block = prior_block;
                } else {
                    previous_block = blocks[index - 1u];
                }
            }
            bool leader = valid
                && (local_index == 0u || (previous_block >> 1) != parent);
            SolinasFp128 output_hamming = solinas_zero();
            SolinasFp128 output_increment = solinas_zero();
            if (leader) {
                SolinasFp128 first_hamming = source_initial
                    ? ram_read_write_from_u64(1u)
                    : hamming[index];
                SolinasFp128 first_increment = increments[index];
                uint second_block = local_index + 1u < source_length
                    ? blocks[index + 1u]
                    : 0u;
                bool paired = local_index + 1u < source_length
                    && (second_block >> 1) == parent;
                if (paired) {
                    SolinasFp128 second_hamming = source_initial
                        ? ram_read_write_from_u64(1u)
                        : hamming[index + 1u];
                    output_hamming = ram_read_write_bind(
                        first_hamming, second_hamming, challenge);
                    output_increment = ram_read_write_bind(
                        first_increment, increments[index + 1u], challenge);
                } else if ((first_block & 1u) == 0u) {
                    output_hamming = ram_read_write_bind(
                        first_hamming, solinas_zero(), challenge);
                    output_increment = ram_read_write_bind(
                        first_increment, solinas_zero(), challenge);
                } else {
                    output_hamming = solinas_mul_wide(challenge, first_hamming);
                    output_increment = solinas_mul_wide(challenge, first_increment);
                }
            }

            uint batch_length = min(threads, source_length - batch);
            if (tid + 1u == batch_length) {
                batch_last_block = first_block;
            }
            uint flag = uint(leader);
            uint local_offset = simd_prefix_exclusive_sum(flag);
            uint simd_count = simd_sum(flag);
            if (lane == 0u) {
                group_offsets[simdgroup] = simd_count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint cursor = running;
                for (uint group = 0u; group < RAM_READ_WRITE_CYCLE_SIMDGROUPS; group++) {
                    uint count = group_offsets[group];
                    group_offsets[group] = cursor;
                    cursor += count;
                }
                running = cursor;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (leader) {
                uint output = begin + group_offsets[simdgroup] + local_offset;
                blocks[output] = parent;
                hamming[output] = output_hamming;
                increments[output] = output_increment;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (tid == 0u) {
                prior_block = batch_last_block;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            segment.length = running;
            segments[segment_index] = segment;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        end = begin + running;
    }

    if (params.emit_message == 0u) {
        return;
    }
    bool message_initial = params.source_initial != 0u && params.bind == 0u;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    for (uint index = begin + tid; index < end; index += threads) {
        uint first_block = blocks[index];
        uint parent = first_block >> 1;
        uint previous_block = index == begin ? 0u : blocks[index - 1u];
        bool leader = index == begin || (previous_block >> 1) != parent;
        if (leader) {
            SolinasFp128 first_hamming = message_initial
                ? ram_read_write_from_u64(1u)
                : hamming[index];
            SolinasFp128 first_increment = increments[index];
            uint second_block = index + 1u < end ? blocks[index + 1u] : 0u;
            bool paired = index + 1u < end
                && (second_block >> 1) == parent;
            SolinasFp128 hamming_slope;
            SolinasFp128 increment_slope;
            SolinasFp128 local_zero;
            if (paired) {
                SolinasFp128 second_hamming = message_initial
                    ? ram_read_write_from_u64(1u)
                    : hamming[index + 1u];
                hamming_slope = solinas_sub(second_hamming, first_hamming);
                increment_slope = solinas_sub(increments[index + 1u], first_increment);
                local_zero = solinas_mul_wide(first_hamming, first_increment);
            } else if ((first_block & 1u) == 0u) {
                hamming_slope = solinas_sub(solinas_zero(), first_hamming);
                increment_slope = solinas_sub(solinas_zero(), first_increment);
                local_zero = solinas_mul_wide(first_hamming, first_increment);
            } else {
                hamming_slope = first_hamming;
                increment_slope = first_increment;
                local_zero = solinas_zero();
            }
            SolinasFp128 head = ram_read_write_head(
                parent, e_in, e_out, params.e_in_length);
            q_zero = solinas_add(q_zero, solinas_mul_wide(head, local_zero));
            q_infinity = solinas_add(
                q_infinity,
                solinas_mul_wide(
                    head,
                    solinas_mul_wide(hamming_slope, increment_slope)));
        }
    }
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < RAM_READ_WRITE_CYCLE_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < RAM_READ_WRITE_CYCLE_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[segment_index] = q_zero;
            partials[params.output_stride + segment_index] = q_infinity;
        }
    }
}

kernel void solinas_ram_read_write_prefix_address(
    device const uint* blocks [[buffer(0)]],
    device const ulong* previous [[buffer(1)]],
    device const ulong* next [[buffer(2)]],
    device const SolinasFp128* weights [[buffer(3)]],
    device const SolinasFp128* suffixes [[buffer(4)]],
    device const SolinasFp128* e_in [[buffer(5)]],
    device const SolinasFp128* e_out [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant RamReadWritePrefixParams& params [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    if (gid < params.records) {
        uint raw_block = blocks[gid];
        uint cycle = raw_block & RAM_READ_WRITE_PREFIX_BLOCK_MASK;
        uint parent = cycle >> (params.rounds_bound + 1u);
        bool leader = (raw_block & RAM_READ_WRITE_PREFIX_START) != 0u
            || ((blocks[gid - 1u] & RAM_READ_WRITE_PREFIX_BLOCK_MASK)
                >> (params.rounds_bound + 1u)) != parent;
        if (leader) {
            SolinasFp128 low_value = solinas_zero();
            SolinasFp128 low_ra = solinas_zero();
            SolinasFp128 high_value = solinas_zero();
            SolinasFp128 high_ra = solinas_zero();
            ulong low_previous = 0u;
            ulong low_next = 0u;
            ulong high_previous = 0u;
            ulong high_next = 0u;
            bool low_present = false;
            bool high_present = false;
            uint cursor = gid;
            while (cursor < params.records) {
                uint cursor_raw = blocks[cursor];
                if (cursor != gid
                    && (cursor_raw & RAM_READ_WRITE_PREFIX_START) != 0u) {
                    break;
                }
                uint cursor_cycle = cursor_raw & RAM_READ_WRITE_PREFIX_BLOCK_MASK;
                if (cursor_cycle >> (params.rounds_bound + 1u) != parent) {
                    break;
                }
                uint table_index = ram_read_write_prefix_table_index(
                    cursor_cycle, params.rounds_bound);
                SolinasFp128 weight = weights[table_index];
                SolinasFp128 suffix = suffixes[table_index];
                SolinasFp128 delta = solinas_sub(
                    ram_read_write_from_u64(next[cursor]),
                    ram_read_write_from_u64(previous[cursor]));
                bool high_child =
                    ((cursor_cycle >> params.rounds_bound) & 1u) != 0u;
                if (high_child) {
                    if (!high_present) {
                        high_present = true;
                        high_previous = previous[cursor];
                        high_value = ram_read_write_from_u64(high_previous);
                    }
                    high_next = next[cursor];
                    high_value = solinas_add(
                        high_value,
                        solinas_mul_wide(delta, suffix));
                    high_ra = solinas_add(high_ra, weight);
                } else {
                    if (!low_present) {
                        low_present = true;
                        low_previous = previous[cursor];
                        low_value = ram_read_write_from_u64(low_previous);
                    }
                    low_next = next[cursor];
                    low_value = solinas_add(
                        low_value,
                        solinas_mul_wide(delta, suffix));
                    low_ra = solinas_add(low_ra, weight);
                }
                cursor++;
            }
            if (!low_present) {
                low_value = ram_read_write_from_u64(high_previous);
            }
            if (!high_present) {
                high_value = ram_read_write_from_u64(low_next);
            }
            SolinasFp128 head = ram_read_write_head(
                parent, e_in, e_out, params.e_in_length);
            q_zero = solinas_mul_wide(
                head,
                solinas_mul_wide(low_ra, low_value));
            q_infinity = solinas_mul_wide(
                head,
                solinas_mul_wide(
                    solinas_sub(high_ra, low_ra),
                    solinas_sub(high_value, low_value)));
        }
    }
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    threadgroup SolinasFp128 zero_sums[RAM_READ_WRITE_HOT_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[RAM_READ_WRITE_HOT_SIMDGROUPS];
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < RAM_READ_WRITE_HOT_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < RAM_READ_WRITE_HOT_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[group] = q_zero;
            partials[params.output_stride + group] = q_infinity;
        }
    }
}

kernel void solinas_ram_read_write_prefix_cycle(
    device const RamReadWriteSegment* segments [[buffer(0)]],
    device const uint* blocks [[buffer(1)]],
    device const uint* address_indices [[buffer(2)]],
    device const ulong* address_previous [[buffer(3)]],
    device const ulong* address_next [[buffer(4)]],
    device const SolinasFp128* address_weights [[buffer(5)]],
    device const SolinasFp128* e_in [[buffer(6)]],
    device const SolinasFp128* e_out [[buffer(7)]],
    device SolinasFp128* partials [[buffer(8)]],
    constant RamReadWritePrefixParams& params [[buffer(9)]],
    uint segment_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    RamReadWriteSegment segment = segments[segment_index];
    uint begin = segment.offset;
    uint end = begin + segment.capacity;
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    for (uint index = begin + tid; index < end; index += threads) {
        uint cycle = blocks[index];
        uint parent = cycle >> (params.rounds_bound + 1u);
        bool leader = index == begin
            || blocks[index - 1u] >> (params.rounds_bound + 1u) != parent;
        if (!leader) {
            continue;
        }
        SolinasFp128 low_hamming = solinas_zero();
        SolinasFp128 low_increment = solinas_zero();
        SolinasFp128 high_hamming = solinas_zero();
        SolinasFp128 high_increment = solinas_zero();
        uint cursor = index;
        while (cursor < end
            && blocks[cursor] >> (params.rounds_bound + 1u) == parent) {
            uint address_index = address_indices[cursor];
            uint table_index = ram_read_write_prefix_table_index(
                blocks[cursor], params.rounds_bound);
            SolinasFp128 weight = address_weights[table_index];
            SolinasFp128 delta = solinas_sub(
                ram_read_write_from_u64(address_next[address_index]),
                ram_read_write_from_u64(address_previous[address_index]));
            SolinasFp128 increment = solinas_mul_wide(delta, weight);
            if (((blocks[cursor] >> params.rounds_bound) & 1u) == 0u) {
                low_hamming = solinas_add(low_hamming, weight);
                low_increment = solinas_add(low_increment, increment);
            } else {
                high_hamming = solinas_add(high_hamming, weight);
                high_increment = solinas_add(high_increment, increment);
            }
            cursor++;
        }
        SolinasFp128 head = ram_read_write_head(
            parent, e_in, e_out, params.e_in_length);
        q_zero = solinas_add(
            q_zero,
            solinas_mul_wide(
                head,
                solinas_mul_wide(low_hamming, low_increment)));
        q_infinity = solinas_add(
            q_infinity,
            solinas_mul_wide(
                head,
                solinas_mul_wide(
                    solinas_sub(high_hamming, low_hamming),
                    solinas_sub(high_increment, low_increment))));
    }
    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
    threadgroup SolinasFp128 zero_sums[RAM_READ_WRITE_CYCLE_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[RAM_READ_WRITE_CYCLE_SIMDGROUPS];
    if (lane == 0u) {
        zero_sums[simdgroup] = q_zero;
        infinity_sums[simdgroup] = q_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        q_zero = lane < RAM_READ_WRITE_CYCLE_SIMDGROUPS
            ? zero_sums[lane]
            : solinas_zero();
        q_infinity = lane < RAM_READ_WRITE_CYCLE_SIMDGROUPS
            ? infinity_sums[lane]
            : solinas_zero();
        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        if (lane == 0u) {
            partials[segment_index] = q_zero;
            partials[params.output_stride + segment_index] = q_infinity;
        }
    }
}

kernel void solinas_ram_read_write_prefix_address_transition(
    device const RamReadWriteSegment* segments [[buffer(0)]],
    device uint* blocks [[buffer(1)]],
    device ulong* previous [[buffer(2)]],
    device ulong* next [[buffer(3)]],
    device SolinasFp128* values [[buffer(4)]],
    device SolinasFp128* ra [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    device const SolinasFp128* weights [[buffer(7)]],
    device const SolinasFp128* suffixes [[buffer(8)]],
    constant RamReadWritePrefixParams& params [[buffer(9)]],
    uint segment_index [[thread_position_in_grid]])
{
    if (segment_index >= params.reserved_0) {
        return;
    }
    partials[segment_index] = solinas_zero();
    partials[params.reserved_0 + segment_index] = solinas_zero();
    RamReadWriteSegment segment = segments[segment_index];
    if (segment.capacity > RAM_READ_WRITE_HOT_CHUNK_SIZE) {
        return;
    }
    uint begin = segment.offset;
    uint end = begin + segment.capacity;
    uint read = begin;
    uint write = begin;
    while (read < end) {
        uint cycle = blocks[read] & RAM_READ_WRITE_PREFIX_BLOCK_MASK;
        uint block = cycle >> params.rounds_bound;
        ulong output_previous = previous[read];
        ulong output_next = next[read];
        SolinasFp128 output_value = ram_read_write_from_u64(output_previous);
        SolinasFp128 output_ra = solinas_zero();
        do {
            uint local_cycle = blocks[read] & RAM_READ_WRITE_PREFIX_BLOCK_MASK;
            uint table_index = ram_read_write_prefix_table_index(
                local_cycle, params.rounds_bound);
            SolinasFp128 weight = weights[table_index];
            SolinasFp128 suffix = suffixes[table_index];
            SolinasFp128 delta = solinas_sub(
                ram_read_write_from_u64(next[read]),
                ram_read_write_from_u64(previous[read]));
            output_value = solinas_add(
                output_value,
                solinas_mul_wide(delta, suffix));
            output_ra = solinas_add(output_ra, weight);
            output_next = next[read];
            read++;
        } while (read < end
            && ((blocks[read] & RAM_READ_WRITE_PREFIX_BLOCK_MASK)
                >> params.rounds_bound) == block);
        blocks[write] = block;
        previous[write] = output_previous;
        next[write] = output_next;
        values[write] = output_value;
        ra[write] = output_ra;
        write++;
    }
}

kernel void solinas_ram_read_write_prefix_hot_transition(
    device const uint* destinations [[buffer(0)]],
    device const uint* source_blocks [[buffer(1)]],
    device const ulong* source_previous [[buffer(2)]],
    device const ulong* source_next [[buffer(3)]],
    device const SolinasFp128* weights [[buffer(4)]],
    device const SolinasFp128* suffixes [[buffer(5)]],
    device uint* output_blocks [[buffer(6)]],
    device ulong* output_previous [[buffer(7)]],
    device ulong* output_next [[buffer(8)]],
    device SolinasFp128* output_values [[buffer(9)]],
    device SolinasFp128* output_ra [[buffer(10)]],
    constant RamReadWritePrefixParams& params [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.records || destinations[gid] == 0xffffffffu) {
        return;
    }
    uint cycle = source_blocks[gid] & RAM_READ_WRITE_PREFIX_BLOCK_MASK;
    uint block = cycle >> params.rounds_bound;
    ulong previous = source_previous[gid];
    ulong next = source_next[gid];
    SolinasFp128 value = ram_read_write_from_u64(previous);
    SolinasFp128 ra = solinas_zero();
    uint cursor = gid;
    do {
        uint local_cycle = source_blocks[cursor] & RAM_READ_WRITE_PREFIX_BLOCK_MASK;
        uint table_index = ram_read_write_prefix_table_index(
            local_cycle, params.rounds_bound);
        SolinasFp128 weight = weights[table_index];
        SolinasFp128 suffix = suffixes[table_index];
        SolinasFp128 delta = solinas_sub(
            ram_read_write_from_u64(source_next[cursor]),
            ram_read_write_from_u64(source_previous[cursor]));
        value = solinas_add(value, solinas_mul_wide(delta, suffix));
        ra = solinas_add(ra, weight);
        next = source_next[cursor];
        cursor++;
    } while (cursor < params.records
        && (source_blocks[cursor] & RAM_READ_WRITE_PREFIX_START) == 0u
        && ((source_blocks[cursor] & RAM_READ_WRITE_PREFIX_BLOCK_MASK)
            >> params.rounds_bound) == block);
    uint output = destinations[gid];
    output_blocks[output] = block;
    output_previous[output] = previous;
    output_next[output] = next;
    output_values[output] = value;
    output_ra[output] = ra;
}

kernel void solinas_ram_read_write_prefix_cycle_transition(
    device const RamReadWriteSegment* segments [[buffer(0)]],
    device const uint* source_blocks [[buffer(1)]],
    device const uint* address_indices [[buffer(2)]],
    device uint* destination_blocks [[buffer(3)]],
    device SolinasFp128* destination_hamming [[buffer(4)]],
    device SolinasFp128* destination_increments [[buffer(5)]],
    device const ulong* address_previous [[buffer(6)]],
    device const ulong* address_next [[buffer(7)]],
    device const SolinasFp128* weights [[buffer(8)]],
    constant RamReadWritePrefixParams& params [[buffer(9)]],
    uint segment_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    RamReadWriteSegment segment = segments[segment_index];
    uint begin = segment.offset;
    uint source_length = segment.capacity;
    threadgroup uint group_offsets[RAM_READ_WRITE_CYCLE_SIMDGROUPS];
    threadgroup uint running;
    threadgroup uint prior_block;
    threadgroup uint batch_last_block;
    if (tid == 0u) {
        running = 0u;
        prior_block = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint batch = 0u; batch < source_length; batch += threads) {
        uint local_index = batch + tid;
        uint index = begin + local_index;
        bool valid = local_index < source_length;
        uint source_block = valid ? source_blocks[index] : 0u;
        uint block = source_block >> params.rounds_bound;
        uint previous_block = 0u;
        if (valid && local_index != 0u) {
            previous_block = tid == 0u ? prior_block : source_blocks[index - 1u];
        }
        bool leader = valid
            && (local_index == 0u
                || (previous_block >> params.rounds_bound) != block);
        SolinasFp128 output_hamming = solinas_zero();
        SolinasFp128 output_increment = solinas_zero();
        if (leader) {
            uint cursor = index;
            uint end = begin + source_length;
            while (cursor < end
                && source_blocks[cursor] >> params.rounds_bound == block) {
                uint address_index = address_indices[cursor];
                uint table_index = ram_read_write_prefix_table_index(
                    source_blocks[cursor], params.rounds_bound);
                SolinasFp128 weight = weights[table_index];
                SolinasFp128 delta = solinas_sub(
                    ram_read_write_from_u64(address_next[address_index]),
                    ram_read_write_from_u64(address_previous[address_index]));
                output_hamming = solinas_add(output_hamming, weight);
                output_increment = solinas_add(
                    output_increment,
                    solinas_mul_wide(delta, weight));
                cursor++;
            }
        }
        uint batch_length = min(threads, source_length - batch);
        if (tid + 1u == batch_length) {
            batch_last_block = source_block;
        }
        uint flag = uint(leader);
        uint local_offset = simd_prefix_exclusive_sum(flag);
        uint simd_count = simd_sum(flag);
        if (lane == 0u) {
            group_offsets[simdgroup] = simd_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint cursor = running;
            for (uint group = 0u; group < RAM_READ_WRITE_CYCLE_SIMDGROUPS; group++) {
                uint count = group_offsets[group];
                group_offsets[group] = cursor;
                cursor += count;
            }
            running = cursor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (leader) {
            uint output = segment.aux_offset
                + group_offsets[simdgroup] + local_offset;
            destination_blocks[output] = block;
            destination_hamming[output] = output_hamming;
            destination_increments[output] = output_increment;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0u) {
            prior_block = batch_last_block;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void solinas_ram_read_write_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamReadWriteReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint column = 0u; column < params.columns; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u && gid / 32u < params.output_count) {
            output[column * params.output_count + gid / 32u] = value;
        }
    }
}
