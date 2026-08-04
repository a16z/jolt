#define ADDRESS_RAF_BINS 256u
#define ADDRESS_RAF_LANES 6u
#define ADDRESS_RAF_FLAG_SHIFT 62u

struct AddressRafScanRow {
    ulong words[5];
};

struct AddressRafParams {
    uint rows;
    uint suffix_len;
    uint rows_per_threadgroup;
    uint threadgroup_count;
};

inline ulong address_compact_even_bits(ulong value) {
    value &= 0x5555555555555555ul;
    value = (value | (value >> 1)) & 0x3333333333333333ul;
    value = (value | (value >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    value = (value | (value >> 4)) & 0x00ff00ff00ff00fful;
    value = (value | (value >> 8)) & 0x0000ffff0000fffful;
    value = (value | (value >> 16)) & 0x00000000fffffffful;
    return value;
}

inline SolinasFp128 address_field_from_u128(ulong lo, ulong hi) {
    SolinasFp128 value;
    value.limb = uint4((uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32));
    return value;
}

inline SolinasFp128 address_simd_sum(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

kernel void solinas_address_raf_histogram(
    device const uchar* keys [[buffer(0)]],
    device uint* group_counts [[buffer(1)]],
    constant AddressRafParams& params [[buffer(2)]],
    threadgroup atomic_uint* histogram [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    for (uint bin = tid; bin < ADDRESS_RAF_BINS; bin += threads) {
        atomic_store_explicit(&histogram[bin], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint start = group * params.rows_per_threadgroup;
    uint end = min(start + params.rows_per_threadgroup, params.rows);
    for (uint row = start + tid; row < end; row += threads) {
        uint bin = keys[row];
        atomic_fetch_add_explicit(&histogram[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bin = tid; bin < ADDRESS_RAF_BINS; bin += threads) {
        group_counts[group * ADDRESS_RAF_BINS + bin] =
            atomic_load_explicit(&histogram[bin], memory_order_relaxed);
    }
}

kernel void solinas_address_raf_offsets(
    device const uint* group_counts [[buffer(0)]],
    device uint* group_offsets [[buffer(1)]],
    device uint* bin_offsets [[buffer(2)]],
    constant AddressRafParams& params [[buffer(3)]],
    threadgroup uint* totals [[threadgroup(0)]],
    uint bin [[thread_index_in_threadgroup]])
{
    uint total = 0;
    for (uint group = 0; group < params.threadgroup_count; group++) {
        total += group_counts[group * ADDRESS_RAF_BINS + bin];
    }
    totals[bin] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (bin == 0) {
        uint running = 0;
        for (uint current = 0; current < ADDRESS_RAF_BINS; current++) {
            bin_offsets[current] = running;
            running += totals[current];
        }
        bin_offsets[ADDRESS_RAF_BINS] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    uint running = bin_offsets[bin];
    for (uint group = 0; group < params.threadgroup_count; group++) {
        uint index = group * ADDRESS_RAF_BINS + bin;
        group_offsets[index] = running;
        running += group_counts[index];
    }
}

kernel void solinas_address_raf_scatter(
    device const uchar* keys [[buffer(0)]],
    device const uint* group_offsets [[buffer(1)]],
    device uint* bucketed_indices [[buffer(2)]],
    constant AddressRafParams& params [[buffer(3)]],
    threadgroup atomic_uint* positions [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    for (uint bin = tid; bin < ADDRESS_RAF_BINS; bin += threads) {
        atomic_store_explicit(&positions[bin], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint start = group * params.rows_per_threadgroup;
    uint end = min(start + params.rows_per_threadgroup, params.rows);
    for (uint row = start + tid; row < end; row += threads) {
        uint bin = keys[row];
        uint local = atomic_fetch_add_explicit(&positions[bin], 1u, memory_order_relaxed);
        uint destination = group_offsets[group * ADDRESS_RAF_BINS + bin] + local;
        bucketed_indices[destination] = row;
    }
}

kernel void solinas_address_raf_reduce(
    device const AddressRafScanRow* rows [[buffer(0)]],
    device const SolinasFp128* weights [[buffer(1)]],
    device const uint* bucketed_indices [[buffer(2)]],
    device const uint* bin_offsets [[buffer(3)]],
    device SolinasFp128* output [[buffer(4)]],
    constant AddressRafParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint bin [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[ADDRESS_RAF_LANES];
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_LANES; output_lane++) {
        sums[output_lane] = solinas_zero();
    }

    uint start = bin_offsets[bin];
    uint end = bin_offsets[bin + 1];
    for (uint position = start + tid; position < end; position += threads) {
        uint row_index = bucketed_indices[position];
        AddressRafScanRow row = rows[row_index];
        ulong lookup_lo = row.words[0];
        ulong lookup_hi = row.words[1];
        bool raf = ((row.words[4] >> ADDRESS_RAF_FLAG_SHIFT) & 1ul) != 0;

        ulong suffix_lo = 0;
        ulong suffix_hi = 0;
        if (params.suffix_len == 64) {
            suffix_lo = lookup_lo;
        } else if (params.suffix_len > 64) {
            suffix_lo = lookup_lo;
            uint high_bits = params.suffix_len - 64;
            suffix_hi = lookup_hi & ((1ul << high_bits) - 1ul);
        } else if (params.suffix_len != 0) {
            suffix_lo = lookup_lo & ((1ul << params.suffix_len) - 1ul);
        }

        SolinasFp128 weight = weights[row_index];
        if (raf) {
            sums[3] = solinas_add(sums[3], weight);
            if (suffix_lo != 0 || suffix_hi != 0) {
                SolinasFp128 identity = solinas_mul_wide(
                    weight,
                    address_field_from_u128(suffix_lo, suffix_hi));
                sums[4] = solinas_add(sums[4], identity);
            }
            uint upper_bits = params.suffix_len > 64 ? params.suffix_len - 64 : 0;
            bool upper_all_ones = upper_bits == 0
                || suffix_hi == ((1ul << upper_bits) - 1ul);
            if (upper_all_ones) {
                sums[5] = solinas_add(sums[5], weight);
            }
        } else {
            sums[0] = solinas_add(sums[0], weight);
            ulong left = address_compact_even_bits(suffix_lo >> 1)
                | (address_compact_even_bits(suffix_hi >> 1) << 32);
            ulong right = address_compact_even_bits(suffix_lo)
                | (address_compact_even_bits(suffix_hi) << 32);
            if (left != 0) {
                SolinasFp128 product = solinas_mul_wide(
                    weight,
                    address_field_from_u128(left, 0));
                sums[1] = solinas_add(sums[1], product);
            }
            if (right != 0) {
                SolinasFp128 product = solinas_mul_wide(
                    weight,
                    address_field_from_u128(right, 0));
                sums[2] = solinas_add(sums[2], product);
            }
        }
    }

    uint simdgroups = threads / 32;
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_LANES; output_lane++) {
        SolinasFp128 sum = address_simd_sum(sums[output_lane]);
        if (lane == 0) {
            shared[output_lane * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_LANES; output_lane++) {
            SolinasFp128 sum = lane < simdgroups
                ? shared[output_lane * simdgroups + lane]
                : solinas_zero();
            sum = address_simd_sum(sum);
            if (lane == 0) {
                output[output_lane * ADDRESS_RAF_BINS + bin] = sum;
            }
        }
    }
}
