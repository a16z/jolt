#define ADDRESS_RAF_BINS 256u
#define ADDRESS_RAF_LANES 6u
#define ADDRESS_RAF_OUTPUTS (ADDRESS_RAF_BINS * ADDRESS_RAF_LANES)
#define ADDRESS_RAF_FLAG_SHIFT 62u

struct AddressRafScanRow {
    ulong words[5];
};

struct AddressRafParams {
    uint rows;
    uint suffix_len;
    uint rows_per_simdgroup;
    uint simdgroup_count;
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

inline void address_sort_pair(
    thread uint& key,
    thread SolinasFp128& a,
    thread SolinasFp128& b,
    thread SolinasFp128& c,
    uint lane,
    uint sequence,
    uint distance)
{
    uint other_key = simd_shuffle_xor(key, distance);
    SolinasFp128 other_a;
    SolinasFp128 other_b;
    SolinasFp128 other_c;
    other_a.limb = simd_shuffle_xor(a.limb, distance);
    other_b.limb = simd_shuffle_xor(b.limb, distance);
    other_c.limb = simd_shuffle_xor(c.limb, distance);
    bool ascending = (lane & sequence) == 0;
    bool lower = (lane & distance) == 0;
    bool wants_minimum = ascending == lower;
    bool take_other = wants_minimum ? other_key < key : other_key > key;
    if (take_other) {
        key = other_key;
        a = other_a;
        b = other_b;
        c = other_c;
    }
}

inline void address_sort_simd(
    thread uint& key,
    thread SolinasFp128& a,
    thread SolinasFp128& b,
    thread SolinasFp128& c,
    uint lane)
{
    for (uint sequence = 2; sequence <= 32; sequence <<= 1) {
        for (uint distance = sequence >> 1; distance > 0; distance >>= 1) {
            address_sort_pair(key, a, b, c, lane, sequence, distance);
        }
    }
}

inline void address_segmented_sum(
    uint key,
    thread SolinasFp128& a,
    thread SolinasFp128& b,
    thread SolinasFp128& c,
    uint lane)
{
    for (ushort offset = 1; offset < 32; offset <<= 1) {
        uint previous_key = simd_shuffle_up(key, offset);
        SolinasFp128 previous_a;
        SolinasFp128 previous_b;
        SolinasFp128 previous_c;
        previous_a.limb = simd_shuffle_up(a.limb, offset);
        previous_b.limb = simd_shuffle_up(b.limb, offset);
        previous_c.limb = simd_shuffle_up(c.limb, offset);
        if (lane >= offset && previous_key == key) {
            a = solinas_add(a, previous_a);
            b = solinas_add(b, previous_b);
            c = solinas_add(c, previous_c);
        }
    }
}

kernel void solinas_address_raf_scan(
    device const AddressRafScanRow* rows [[buffer(0)]],
    device const SolinasFp128* weights [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant AddressRafParams& params [[buffer(3)]],
    uint threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint simdgroups_per_threadgroup = threads / 32;
    uint group = threadgroup_position * simdgroups_per_threadgroup + simdgroup;
    if (group >= params.simdgroup_count) {
        return;
    }

    uint partial_base = group * ADDRESS_RAF_OUTPUTS;
    for (uint output = lane; output < ADDRESS_RAF_OUTPUTS; output += 32) {
        partials[partial_base + output] = solinas_zero();
    }

    uint row_start = group * params.rows_per_simdgroup;
    uint row_end = min(row_start + params.rows_per_simdgroup, params.rows);
    for (uint batch = row_start; batch < row_end; batch += 32) {
        uint row_index = batch + lane;
        uint key = 0xffffffffu;
        SolinasFp128 a = solinas_zero();
        SolinasFp128 b = solinas_zero();
        SolinasFp128 c = solinas_zero();
        if (row_index < row_end) {
            AddressRafScanRow row = rows[row_index];
            ulong lookup_lo = row.words[0];
            ulong lookup_hi = row.words[1];
            bool raf = ((row.words[4] >> ADDRESS_RAF_FLAG_SHIFT) & 1ul) != 0;
            uint chunk;
            if (params.suffix_len < 64) {
                chunk = (uint)((lookup_lo >> params.suffix_len) & 0xfful);
            } else {
                chunk = (uint)((lookup_hi >> (params.suffix_len - 64)) & 0xfful);
            }
            key = chunk + (raf ? ADDRESS_RAF_BINS : 0u);

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
            a = weight;
            if (raf) {
                if (suffix_lo != 0 || suffix_hi != 0) {
                    b = solinas_mul_wide(weight, address_field_from_u128(suffix_lo, suffix_hi));
                }
                uint upper_bits = params.suffix_len > 64 ? params.suffix_len - 64 : 0;
                bool upper_all_ones = upper_bits == 0
                    || suffix_hi == ((1ul << upper_bits) - 1ul);
                c = upper_all_ones ? weight : solinas_zero();
            } else {
                ulong left = address_compact_even_bits(suffix_lo >> 1)
                    | (address_compact_even_bits(suffix_hi >> 1) << 32);
                ulong right = address_compact_even_bits(suffix_lo)
                    | (address_compact_even_bits(suffix_hi) << 32);
                if (left != 0) {
                    b = solinas_mul_wide(weight, address_field_from_u128(left, 0));
                }
                if (right != 0) {
                    c = solinas_mul_wide(weight, address_field_from_u128(right, 0));
                }
            }
        }

        address_sort_simd(key, a, b, c, lane);
        address_segmented_sum(key, a, b, c, lane);
        uint next_key = simd_shuffle_down(key, 1);
        bool is_tail = lane == 31 || next_key != key;
        if (is_tail && key < 2 * ADDRESS_RAF_BINS) {
            bool raf = key >= ADDRESS_RAF_BINS;
            uint chunk = key & (ADDRESS_RAF_BINS - 1);
            uint first_lane = raf ? 3 : 0;
            uint indices[3] = {
                partial_base + (first_lane + 0) * ADDRESS_RAF_BINS + chunk,
                partial_base + (first_lane + 1) * ADDRESS_RAF_BINS + chunk,
                partial_base + (first_lane + 2) * ADDRESS_RAF_BINS + chunk,
            };
            partials[indices[0]] = solinas_add(partials[indices[0]], a);
            partials[indices[1]] = solinas_add(partials[indices[1]], b);
            partials[indices[2]] = solinas_add(partials[indices[2]], c);
        }
    }
}

kernel void solinas_address_raf_reduce(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant AddressRafParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= ADDRESS_RAF_OUTPUTS) {
        return;
    }
    SolinasFp128 sum = solinas_zero();
    for (uint group = 0; group < params.simdgroup_count; group++) {
        sum = solinas_add(sum, partials[group * ADDRESS_RAF_OUTPUTS + gid]);
    }
    output[gid] = sum;
}
