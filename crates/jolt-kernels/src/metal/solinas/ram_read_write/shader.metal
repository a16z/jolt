// Concatenate after offset-specialized fp128.metal and simd_reduce.metal.

#define RAM_READ_WRITE_HOT_THRESHOLD 4096u
#define RAM_READ_WRITE_SIMD_WIDTH 32u
#define RAM_READ_WRITE_HOT_THREADS 256u
#define RAM_READ_WRITE_HOT_SIMDGROUPS \
    (RAM_READ_WRITE_HOT_THREADS / RAM_READ_WRITE_SIMD_WIDTH)

struct RamReadWriteSegment {
    uint offset;
    uint length;
    uint capacity;
    uint reserved;
};

struct RamReadWritePhaseParams {
    uint work_items;
    uint output_stride;
    uint e_in_length;
    uint bind;
    uint emit_message;
};

struct RamReadWriteReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

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
    if (segment.capacity > RAM_READ_WRITE_HOT_THRESHOLD) {
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
            SolinasFp128 first_value = values[read];
            SolinasFp128 first_ra = ra[read];
            bool paired = read + 1u < end && (blocks[read + 1u] >> 1) == parent;

            ulong output_previous;
            ulong output_next;
            SolinasFp128 output_value;
            SolinasFp128 output_ra;
            if (paired) {
                ulong second_next = next[read + 1u];
                SolinasFp128 second_value = values[read + 1u];
                SolinasFp128 second_ra = ra[read + 1u];
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
    while (read < end) {
        uint first_block = blocks[read];
        uint parent = first_block >> 1;
        SolinasFp128 first_value = values[read];
        SolinasFp128 first_ra = ra[read];
        bool paired = read + 1u < end && (blocks[read + 1u] >> 1) == parent;
        SolinasFp128 local_zero;
        SolinasFp128 local_infinity;
        if (paired) {
            SolinasFp128 value_slope = solinas_sub(values[read + 1u], first_value);
            SolinasFp128 ra_slope = solinas_sub(ra[read + 1u], first_ra);
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

kernel void solinas_ram_read_write_address_hot(
    device const uint* hot_addresses [[buffer(0)]],
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
    uint hot_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (hot_index >= params.work_items) {
        return;
    }
    uint segment_index = hot_addresses[hot_index];
    RamReadWriteSegment segment = segments[segment_index];
    uint begin = segment.offset;
    uint end = begin + segment.length;
    threadgroup uint group_offsets[RAM_READ_WRITE_HOT_SIMDGROUPS];
    threadgroup uint compacted_length;
    threadgroup uint previous_chunk_parent;
    threadgroup uint current_chunk_last_parent;
    threadgroup SolinasFp128 zero_sums[RAM_READ_WRITE_HOT_SIMDGROUPS];
    threadgroup SolinasFp128 infinity_sums[RAM_READ_WRITE_HOT_SIMDGROUPS];

    if (params.bind != 0u) {
        if (tid == 0u) {
            compacted_length = 0u;
            previous_chunk_parent = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint chunk = 0u; chunk < segment.length; chunk += threads) {
            uint index = begin + chunk + tid;
            bool valid = index < end;
            uint first_block = valid ? blocks[index] : 0u;
            uint parent = first_block >> 1;
            uint boundary_parent = previous_chunk_parent;
            if (tid == 0u) {
                uint chunk_length = min(threads, segment.length - chunk);
                current_chunk_last_parent =
                    blocks[begin + chunk + chunk_length - 1u] >> 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool leader = valid
                && (index == begin
                    || (tid == 0u
                        ? boundary_parent
                        : blocks[index - 1u] >> 1) != parent);
            ulong output_previous = 0u;
            ulong output_next = 0u;
            SolinasFp128 output_value = solinas_zero();
            SolinasFp128 output_ra = solinas_zero();

            if (leader) {
                ulong first_previous = previous[index];
                ulong first_next = next[index];
                SolinasFp128 first_value = values[index];
                SolinasFp128 first_ra = ra[index];
                bool paired = index + 1u < end
                    && (blocks[index + 1u] >> 1) == parent;
                output_previous = first_previous;
                output_next = first_next;
                if (paired) {
                    output_next = next[index + 1u];
                    output_value = ram_read_write_bind(
                        first_value, values[index + 1u], challenge);
                    output_ra = ram_read_write_bind(
                        first_ra, ra[index + 1u], challenge);
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
                uint cursor = compacted_length;
                for (uint group = 0u; group < RAM_READ_WRITE_HOT_SIMDGROUPS; group++) {
                    uint count = group_offsets[group];
                    group_offsets[group] = cursor;
                    cursor += count;
                }
                compacted_length = cursor;
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
            threadgroup_barrier(mem_flags::mem_device);
            if (tid == 0u) {
                previous_chunk_parent = current_chunk_last_parent;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            segment.length = compacted_length;
            segments[segment_index] = segment;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        end = begin + compacted_length;
    }

    if (params.emit_message == 0u) {
        return;
    }
    SolinasFp128 q_zero = solinas_zero();
    SolinasFp128 q_infinity = solinas_zero();
    for (uint index = begin + tid; index < end; index += threads) {
        uint first_block = blocks[index];
        uint parent = first_block >> 1;
        bool leader = index == begin || (blocks[index - 1u] >> 1) != parent;
        if (!leader) {
            continue;
        }
        SolinasFp128 first_value = values[index];
        SolinasFp128 first_ra = ra[index];
        bool paired = index + 1u < end && (blocks[index + 1u] >> 1) == parent;
        SolinasFp128 local_zero;
        SolinasFp128 local_infinity;
        if (paired) {
            SolinasFp128 value_slope = solinas_sub(values[index + 1u], first_value);
            SolinasFp128 ra_slope = solinas_sub(ra[index + 1u], first_ra);
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
            parent, e_in, e_out, params.e_in_length);
        q_zero = solinas_add(q_zero, solinas_mul_wide(head, local_zero));
        q_infinity = solinas_add(
            q_infinity,
            solinas_mul_wide(head, local_infinity));
    }

    q_zero = solinas_simd_sum_32(q_zero);
    q_infinity = solinas_simd_sum_32(q_infinity);
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
            partials[segment_index] = q_zero;
            partials[params.output_stride + segment_index] = q_infinity;
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
    uint segment_index [[thread_position_in_grid]])
{
    if (segment_index >= params.work_items) {
        return;
    }
    RamReadWriteSegment segment = segments[segment_index];
    uint begin = segment.offset;
    uint end = begin + segment.length;

    if (params.bind != 0u) {
        uint read = begin;
        uint write = begin;
        while (read < end) {
            uint first_block = blocks[read];
            uint parent = first_block >> 1;
            SolinasFp128 first_hamming = hamming[read];
            SolinasFp128 first_increment = increments[read];
            bool paired = read + 1u < end && (blocks[read + 1u] >> 1) == parent;
            SolinasFp128 output_hamming;
            SolinasFp128 output_increment;
            if (paired) {
                output_hamming = ram_read_write_bind(
                    first_hamming, hamming[read + 1u], challenge);
                output_increment = ram_read_write_bind(
                    first_increment, increments[read + 1u], challenge);
                read += 2u;
            } else if ((first_block & 1u) == 0u) {
                output_hamming = ram_read_write_bind(
                    first_hamming, solinas_zero(), challenge);
                output_increment = ram_read_write_bind(
                    first_increment, solinas_zero(), challenge);
                read += 1u;
            } else {
                output_hamming = solinas_mul_wide(challenge, first_hamming);
                output_increment = solinas_mul_wide(challenge, first_increment);
                read += 1u;
            }
            blocks[write] = parent;
            hamming[write] = output_hamming;
            increments[write] = output_increment;
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
    while (read < end) {
        uint first_block = blocks[read];
        uint parent = first_block >> 1;
        SolinasFp128 first_hamming = hamming[read];
        SolinasFp128 first_increment = increments[read];
        bool paired = read + 1u < end && (blocks[read + 1u] >> 1) == parent;
        SolinasFp128 hamming_slope;
        SolinasFp128 increment_slope;
        SolinasFp128 local_zero;
        if (paired) {
            hamming_slope = solinas_sub(hamming[read + 1u], first_hamming);
            increment_slope = solinas_sub(increments[read + 1u], first_increment);
            local_zero = solinas_mul_wide(first_hamming, first_increment);
            read += 2u;
        } else if ((first_block & 1u) == 0u) {
            hamming_slope = solinas_sub(solinas_zero(), first_hamming);
            increment_slope = solinas_sub(solinas_zero(), first_increment);
            local_zero = solinas_mul_wide(first_hamming, first_increment);
            read += 1u;
        } else {
            hamming_slope = first_hamming;
            increment_slope = first_increment;
            local_zero = solinas_zero();
            read += 1u;
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
    partials[segment_index] = q_zero;
    partials[params.output_stride + segment_index] = q_infinity;
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
