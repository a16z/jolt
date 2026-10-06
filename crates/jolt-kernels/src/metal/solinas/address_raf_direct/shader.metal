#define ADDRESS_RAF_DIRECT_BINS 256u
#define ADDRESS_RAF_DIRECT_KEYS (2u * ADDRESS_RAF_DIRECT_BINS)
#define ADDRESS_RAF_DIRECT_LANES 3u
#define ADDRESS_RAF_DIRECT_WORDS 5u
#define ADDRESS_RAF_DIRECT_FIELDS (ADDRESS_RAF_DIRECT_KEYS * ADDRESS_RAF_DIRECT_LANES)

struct AddressRafDirectParams {
    uint rows;
    uint suffix_len;
    uint rows_per_threadgroup;
    uint threadgroup_count;
    uint condense;
    uint packed_rows;
};

struct AddressRafDirectLookup {
    ulong2 limbs;
};

inline ulong address_direct_compact_even_bits(ulong value) {
    value &= 0x5555555555555555ul;
    value = (value | (value >> 1)) & 0x3333333333333333ul;
    value = (value | (value >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    value = (value | (value >> 4)) & 0x00ff00ff00ff00fful;
    value = (value | (value >> 8)) & 0x0000ffff0000fffful;
    value = (value | (value >> 16)) & 0x00000000fffffffful;
    return value;
}

inline SolinasFp128 address_direct_field_from_u128(ulong lo, ulong hi) {
    SolinasFp128 value;
    value.limb = uint4((uint)lo, (uint)(lo >> 32), (uint)hi, (uint)(hi >> 32));
    return value;
}

inline uint address_direct_lookup_byte(AddressRafDirectLookup lookup, uint shift) {
    return shift < 64
        ? (uint)(lookup.limbs[0] >> shift) & 0xffu
        : (uint)(lookup.limbs[1] >> (shift - 64)) & 0xffu;
}

inline SolinasFp128 address_direct_simd_sum(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

kernel void solinas_address_raf_direct_tile(
    device const uchar* packed_rows [[buffer(0)]],
    device const AddressRafDirectLookup* lookups [[buffer(1)]],
    device SolinasFp128* weights [[buffer(2)]],
    device const SolinasFp128* previous_phase_table [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant AddressRafDirectParams& params [[buffer(5)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint counters = ADDRESS_RAF_DIRECT_FIELDS * ADDRESS_RAF_DIRECT_WORDS;
    for (uint counter = tid; counter < counters; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Chunk 0 dominates (padding rows, the high chunks of 64-bit lookups);
    // its six fields are summed per thread instead of contended atomics.
    SolinasLazySum chunk_zero[2 * ADDRESS_RAF_DIRECT_LANES];
    for (uint slot = 0; slot < 2 * ADDRESS_RAF_DIRECT_LANES; slot++) {
        chunk_zero[slot] = solinas_lazy_zero();
    }
    uint start = group * params.rows_per_threadgroup;
    uint end = min(start + params.rows_per_threadgroup, params.rows);
    for (uint row = start + tid; row < end; row += threads) {
        AddressRafDirectLookup lookup = lookups[row];
        uint raf_flag = params.packed_rows == 0
            ? (uint)packed_rows[row]
            : (uint)packed_rows[row] >> 7;
        uint key = address_direct_lookup_byte(lookup, params.suffix_len)
            | (raf_flag << 8);
        SolinasFp128 weight = weights[row];
        if (params.condense != 0) {
            uint previous_chunk = address_direct_lookup_byte(lookup, params.suffix_len + 8);
            weight = solinas_mul_wide(weight, previous_phase_table[previous_chunk]);
            weights[row] = weight;
        }

        ulong lookup_lo = lookup.limbs[0];
        ulong lookup_hi = lookup.limbs[1];
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

        uint first_field = key * ADDRESS_RAF_DIRECT_LANES;
        bool local = (key & (ADDRESS_RAF_DIRECT_BINS - 1u)) == 0u;
        uint first_slot = raf_flag * ADDRESS_RAF_DIRECT_LANES;
        SolinasFp128 contributions[ADDRESS_RAF_DIRECT_LANES];
        bool present[ADDRESS_RAF_DIRECT_LANES];
        contributions[0] = weight;
        present[0] = true;
        if (key < ADDRESS_RAF_DIRECT_BINS) {
            ulong left = address_direct_compact_even_bits(suffix_lo >> 1)
                | (address_direct_compact_even_bits(suffix_hi >> 1) << 32);
            ulong right = address_direct_compact_even_bits(suffix_lo)
                | (address_direct_compact_even_bits(suffix_hi) << 32);
            present[1] = left != 0;
            present[2] = right != 0;
            contributions[1] = present[1]
                ? solinas_mul_wide(weight, address_direct_field_from_u128(left, 0))
                : solinas_zero();
            contributions[2] = present[2]
                ? solinas_mul_wide(weight, address_direct_field_from_u128(right, 0))
                : solinas_zero();
        } else {
            present[1] = suffix_lo != 0 || suffix_hi != 0;
            contributions[1] = present[1]
                ? solinas_mul_wide(weight, address_direct_field_from_u128(suffix_lo, suffix_hi))
                : solinas_zero();
            uint upper_bits = params.suffix_len > 64 ? params.suffix_len - 64 : 0;
            present[2] = upper_bits == 0 || suffix_hi == ((1ul << upper_bits) - 1ul);
            contributions[2] = weight;
        }
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            if (!present[output_lane]) {
                continue;
            }
            if (local) {
                solinas_lazy_add(chunk_zero[first_slot + output_lane], contributions[output_lane]);
            } else {
                solinas_deferred_atomic_add_5(
                    sums, first_field + output_lane, contributions[output_lane]);
            }
        }
    }
    for (uint flag = 0; flag < 2; flag++) {
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            solinas_deferred_atomic_flush_simd(
                sums,
                flag * ADDRESS_RAF_DIRECT_BINS * ADDRESS_RAF_DIRECT_LANES + output_lane,
                chunk_zero[flag * ADDRESS_RAF_DIRECT_LANES + output_lane],
                lane);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_base = group * ADDRESS_RAF_DIRECT_FIELDS;
    for (uint field = tid; field < ADDRESS_RAF_DIRECT_FIELDS; field += threads) {
        partials[output_base + field] = solinas_deferred_atomic_reduce_5(sums, field);
    }
}

kernel void solinas_address_raf_direct_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant AddressRafDirectParams& params [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint key [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[ADDRESS_RAF_DIRECT_LANES];
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
        sums[output_lane] = solinas_zero();
    }
    uint field = key * ADDRESS_RAF_DIRECT_LANES;
    for (uint group = tid; group < params.threadgroup_count; group += threads) {
        uint base = group * ADDRESS_RAF_DIRECT_FIELDS + field;
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            sums[output_lane] = solinas_add(sums[output_lane], partials[base + output_lane]);
        }
    }

    uint simdgroups = threads / 32;
    for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
        SolinasFp128 sum = address_direct_simd_sum(sums[output_lane]);
        if (lane == 0) {
            shared[output_lane * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint output_lane = 0; output_lane < ADDRESS_RAF_DIRECT_LANES; output_lane++) {
            SolinasFp128 sum = lane < simdgroups
                ? shared[output_lane * simdgroups + lane]
                : solinas_zero();
            sum = address_direct_simd_sum(sum);
            if (lane == 0) {
                uint chunk = key & (ADDRESS_RAF_DIRECT_BINS - 1);
                uint first_lane = key >= ADDRESS_RAF_DIRECT_BINS ? 3 : 0;
                output[(first_lane + output_lane) * ADDRESS_RAF_DIRECT_BINS + chunk] = sum;
            }
        }
    }
}
