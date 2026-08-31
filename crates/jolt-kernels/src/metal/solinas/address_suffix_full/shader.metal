#define ADDRESS_SUFFIX_FULL_BINS 256u
#define ADDRESS_SUFFIX_FULL_MAX_SUFFIXES 5u
#define ADDRESS_SUFFIX_FULL_WORDS 5u
#define ADDRESS_SUFFIX_FULL_FIELDS (ADDRESS_SUFFIX_FULL_BINS * ADDRESS_SUFFIX_FULL_MAX_SUFFIXES)

struct AddressSuffixFullParams {
    uint suffix_len;
    uint job_count;
    uint output_elements;
    uint reserved;
};

struct AddressSuffixFullJob {
    uint start;
    uint end;
    uint table;
    uint reserved;
};

struct AddressSuffixFullTable {
    uint job_start;
    uint job_end;
    uint output_start;
    uint suffix_count;
};

struct AddressSuffixFullLookup {
    ulong2 limbs;
};

struct AddressSuffixFullBits {
    ulong lo;
    ulong hi;
    ulong x;
    ulong y;
    uint len;
    uint operand_len;
};

inline ulong address_suffix_full_compact_even_bits(ulong value) {
    value &= 0x5555555555555555ul;
    value = (value | (value >> 1)) & 0x3333333333333333ul;
    value = (value | (value >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    value = (value | (value >> 4)) & 0x00ff00ff00ff00fful;
    value = (value | (value >> 8)) & 0x0000ffff0000fffful;
    value = (value | (value >> 16)) & 0x00000000fffffffful;
    return value;
}

inline ulong address_suffix_full_mask(uint bits) {
    return bits == 0 ? 0ul : ((1ul << bits) - 1ul);
}

inline AddressSuffixFullBits address_suffix_full_bits(
    AddressSuffixFullLookup lookup,
    uint suffix_len)
{
    AddressSuffixFullBits bits;
    bits.len = suffix_len;
    bits.operand_len = suffix_len / 2;
    bits.lo = 0;
    bits.hi = 0;
    if (suffix_len == 64) {
        bits.lo = lookup.limbs[0];
    } else if (suffix_len > 64) {
        bits.lo = lookup.limbs[0];
        bits.hi = lookup.limbs[1] & address_suffix_full_mask(suffix_len - 64);
    } else if (suffix_len != 0) {
        bits.lo = lookup.limbs[0] & address_suffix_full_mask(suffix_len);
    }
    bits.x = address_suffix_full_compact_even_bits(bits.lo >> 1)
        | (address_suffix_full_compact_even_bits(bits.hi >> 1) << 32);
    bits.y = address_suffix_full_compact_even_bits(bits.lo)
        | (address_suffix_full_compact_even_bits(bits.hi) << 32);
    return bits;
}

inline uint address_suffix_full_lookup_byte(AddressSuffixFullLookup lookup, uint shift) {
    return shift < 64
        ? (uint)(lookup.limbs[0] >> shift) & 0xffu
        : (uint)(lookup.limbs[1] >> (shift - 64)) & 0xffu;
}

inline uint address_suffix_full_trailing_zeros(ulong value, uint len) {
    return value == 0 ? len : min((uint)ctz(value), len);
}

inline uint address_suffix_full_leading_ones(ulong value, uint len) {
    if (len == 0) {
        return 0;
    }
    ulong inverse = (~value) & address_suffix_full_mask(len);
    return inverse == 0 ? len : (uint)clz(inverse << (64 - len));
}

inline ulong address_suffix_full_unbounded_shl(ulong value, uint shift) {
    return shift >= 64 ? 0ul : value << shift;
}

inline ulong address_suffix_full_unbounded_shr(ulong value, uint shift) {
    return shift >= 64 ? 0ul : value >> shift;
}

inline ulong address_suffix_full_rotate_right(ulong value, uint shift) {
    return (value >> shift) | (value << (64 - shift));
}

inline uint address_suffix_full_rotate_right_32(uint value, uint shift) {
    return (value >> shift) | (value << (32 - shift));
}

inline uint address_suffix_full_swap_bytes_32(uint value) {
    return ((value & 0x000000ffu) << 24)
        | ((value & 0x0000ff00u) << 8)
        | ((value & 0x00ff0000u) >> 8)
        | ((value & 0xff000000u) >> 24);
}

inline ulong address_suffix_full_pext(ulong x, ulong y) {
    ulong output = 0;
    uint destination = 0;
    while (y != 0) {
        uint source = ctz(y);
        output |= ((x >> source) & 1ul) << destination;
        destination += 1;
        y &= y - 1;
    }
    return output;
}

inline ulong address_suffix_full_window_sign(ulong x, ulong y) {
    return y == 0 ? 0ul : ((x >> (63u - (uint)clz(y))) & 1ul);
}

inline ulong address_suffix_full_sign_extension_w(AddressSuffixFullBits bits) {
    if (bits.len == 0) {
        return 0ul;
    }

    constexpr uint word_half = 32u;
    uint count = min(bits.operand_len, word_half);
    ulong fill = 0;
    if (bits.len >= 64) {
        if (((bits.x >> (word_half - 1)) & 1ul) == 0) {
            return 0ul;
        }
        fill = 0xffffffff00000000ul;
    }
    uint first_position = word_half - count;
    for (uint offset = 0; offset < count; offset++) {
        uint position = first_position + offset;
        if (position != 0) {
            ulong y_bit = (bits.y >> (count - 1 - offset)) & 1ul;
            fill += (1ul - y_bit) << position;
        }
    }
    return fill;
}

inline ulong address_suffix_full_evaluate(uchar kind, AddressSuffixFullBits bits) {
    ulong operand_mask = address_suffix_full_mask(bits.operand_len);
    switch (kind) {
        case 0: return 1ul;
        case 1: return bits.x & bits.y;
        case 2: return bits.x & ~bits.y;
        case 3: return bits.x ^ bits.y;
        case 4: return bits.x | bits.y;
        case 5: return bits.y;
        case 6: return (ulong)(uint)bits.y;
        case 7: return (ulong)(bits.x == 0 && bits.y == operand_mask);
        case 8: {
            uint len = min(bits.operand_len, 32u);
            return (ulong)((uint)bits.x == 0 && (uint)bits.y == (uint)address_suffix_full_mask(len));
        }
        case 9: return bits.hi;
        case 10: return bits.lo;
        case 11: return (ulong)(uint)bits.lo;
        case 12: return (ulong)(bits.x < bits.y);
        case 13: return (ulong)(bits.x > bits.y);
        case 14: return (ulong)(bits.x == bits.y);
        case 15: return (ulong)(bits.x == 0);
        case 16: return (ulong)(bits.y == 0);
        case 17: return bits.len == 0 ? 1ul : bits.lo & 1ul;
        case 18: return (ulong)(bits.x == 0 && bits.y == operand_mask);
        case 19: return bits.len == 0 ? 1ul : 1ul << (bits.lo & 63ul);
        case 20: return bits.len == 0 ? 1ul : 1ul << (bits.lo & 31ul);
        case 21: {
            ulong lo = (ulong)address_suffix_full_swap_bytes_32((uint)bits.lo);
            ulong hi = (ulong)address_suffix_full_swap_bytes_32((uint)(bits.lo >> 32));
            return lo | (hi << 32);
        }
        case 22: return bits.len == 0 ? 1ul : 1ul << (63u - (uint)(bits.lo & 63ul));
        case 23: return address_suffix_full_unbounded_shr(
            bits.x,
            address_suffix_full_trailing_zeros(bits.y, bits.operand_len));
        case 24: return 1ul << address_suffix_full_leading_ones(bits.y, bits.operand_len);
        case 25: {
            uint padding = address_suffix_full_trailing_zeros(bits.y, bits.operand_len);
            return padding == 0 ? 0ul : (~0ul << (64 - padding));
        }
        case 26: return address_suffix_full_unbounded_shl(
            bits.x & ~bits.y,
            address_suffix_full_leading_ones(bits.y, bits.operand_len));
        case 27: return (ulong)(bits.len == 0 || (bits.lo & 3ul) == 0);
        case 28: {
            if (bits.len < 32) return 1ul;
            return ((bits.lo >> 31) & 1ul) != 0 ? 0xffffffff00000000ul : 0ul;
        }
        case 29: {
            if (bits.len < 64) return 1ul;
            return ((bits.lo >> 62) & 1ul) != 0 ? 0xffffffff00000000ul : 0ul;
        }
        case 30: {
            uint shift = min(address_suffix_full_trailing_zeros(bits.y, bits.operand_len), 32u);
            return shift == 32 ? 0ul : (ulong)((uint)bits.x >> shift);
        }
        case 31: {
            uint len = min(bits.operand_len, 32u);
            return 1ul << address_suffix_full_leading_ones((uint)bits.y, len);
        }
        case 32: {
            uint leading = address_suffix_full_leading_ones((uint)bits.y, bits.operand_len);
            return leading >= 32 ? 0ul : (ulong)(1u << leading);
        }
        case 33: {
            uint len = min(bits.operand_len, 32u);
            uint leading = address_suffix_full_leading_ones((uint)bits.y, len);
            uint value = (uint)bits.x & ~(uint)bits.y;
            return leading >= 32 ? 0ul : (ulong)(value << leading);
        }
        case 34: return (ulong)(bits.hi == 0);
        case 35: return address_suffix_full_rotate_right(bits.x ^ bits.y, 16);
        case 36: return address_suffix_full_rotate_right(bits.x ^ bits.y, 24);
        case 37: return address_suffix_full_rotate_right(bits.x ^ bits.y, 32);
        case 38: return address_suffix_full_rotate_right(bits.x ^ bits.y, 63);
        case 39: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 16);
        case 40: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 12);
        case 41: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 8);
        case 42: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 7);
        case 43: {
            if (bits.len < 3) return 1ul;
            return 1ul << (((bits.lo >> 2) & 1ul) * 32);
        }
        case 44: return address_suffix_full_pext(bits.x, bits.y);
        case 45: {
            uint count = popcount(bits.y);
            return count < 64 ? 1ul << count : 0ul;
        }
        case 46: return address_suffix_full_window_sign(bits.x, bits.y);
        case 47: {
            ulong sign = address_suffix_full_window_sign(bits.x, bits.y);
            uint count = popcount(bits.y);
            return sign != 0 && count < 64 ? 1ul << count : 0ul;
        }
        case 48: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 22);
        case 49: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 19);
        case 50: return (ulong)address_suffix_full_rotate_right_32((uint)bits.x ^ (uint)bits.y, 6);
        case 51: return address_suffix_full_sign_extension_w(bits);
        case 52: return bits.len < 64 ? 0ul : ((bits.x >> 31) & 1ul) * (bits.y & 1ul);
        case 53: return 1ul << (8 * (uint)(bits.lo & 7ul));
        case 54: return 1ul << (8 * (uint)(bits.lo & 6ul));
        case 55: return bits.lo & ~7ul;
        default: return 0ul;
    }
}

inline SolinasFp128 address_suffix_full_field_from_u64(ulong scalar) {
    SolinasFp128 value = solinas_zero();
    value.limb[0] = (uint)scalar;
    value.limb[1] = (uint)(scalar >> 32);
    return value;
}

inline void address_suffix_full_atomic_add(
    threadgroup atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * ADDRESS_SUFFIX_FULL_WORDS;
    uint carry = 0;
    for (uint limb = 0; limb < 4; limb++) {
        ulong addend = (ulong)value.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(
            &sums[base + limb],
            low,
            memory_order_relaxed);
        carry = (uint)(addend >> 32) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry != 0) {
        atomic_fetch_add_explicit(&sums[base + 4], carry, memory_order_relaxed);
    }
}

inline SolinasFp128 address_suffix_full_reduce_atomic_sum(
    threadgroup atomic_uint* sums,
    uint field)
{
    uint base = field * ADDRESS_SUFFIX_FULL_WORDS;
    SolinasFp128 low;
    for (uint limb = 0; limb < 4; limb++) {
        low.limb[limb] = atomic_load_explicit(&sums[base + limb], memory_order_relaxed);
    }
    uint overflow = atomic_load_explicit(&sums[base + 4], memory_order_relaxed);
    SolinasCorrection canonical = solinas_add_offset(low);
    low = solinas_select(canonical.carry != 0, canonical.value, low);

    ulong correction_word = (ulong)overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32);
    return solinas_add(low, correction);
}

kernel void solinas_address_suffix_full_tile(
    device const AddressSuffixFullLookup* lookups [[buffer(0)]],
    device const SolinasFp128* weights [[buffer(1)]],
    device const AddressSuffixFullJob* jobs [[buffer(2)]],
    device const uchar* suffix_kinds [[buffer(3)]],
    device const uchar* suffix_counts [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant AddressSuffixFullParams& params [[buffer(6)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint job_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    for (uint counter = tid; counter < ADDRESS_SUFFIX_FULL_FIELDS * ADDRESS_SUFFIX_FULL_WORDS; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AddressSuffixFullJob job = jobs[job_index];
    uint suffix_count = suffix_counts[job.table];
    for (uint row = job.start + tid; row < job.end; row += threads) {
        AddressSuffixFullLookup lookup = lookups[row];
        AddressSuffixFullBits bits = address_suffix_full_bits(lookup, params.suffix_len);
        uint chunk = address_suffix_full_lookup_byte(lookup, params.suffix_len);
        SolinasFp128 weight = weights[row];
        for (uint suffix = 0; suffix < suffix_count; suffix++) {
            uchar kind = suffix_kinds[job.table * ADDRESS_SUFFIX_FULL_MAX_SUFFIXES + suffix];
            ulong scalar = address_suffix_full_evaluate(kind, bits);
            if (scalar != 0) {
                SolinasFp128 contribution = scalar == 1
                    ? weight
                    : solinas_mul_wide(weight, address_suffix_full_field_from_u64(scalar));
                address_suffix_full_atomic_add(
                    sums,
                    suffix * ADDRESS_SUFFIX_FULL_BINS + chunk,
                    contribution);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_base = job_index * ADDRESS_SUFFIX_FULL_FIELDS;
    for (uint field = tid; field < ADDRESS_SUFFIX_FULL_FIELDS; field += threads) {
        partials[output_base + field] = address_suffix_full_reduce_atomic_sum(sums, field);
    }
}

kernel void solinas_address_suffix_full_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device const AddressSuffixFullTable* tables [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    uint table [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    AddressSuffixFullTable descriptor = tables[table];
    uint fields = descriptor.suffix_count * ADDRESS_SUFFIX_FULL_BINS;
    for (uint field = tid; field < fields; field += threads) {
        SolinasFp128 sum = solinas_zero();
        for (uint job = descriptor.job_start; job < descriptor.job_end; job++) {
            sum = solinas_add(sum, partials[job * ADDRESS_SUFFIX_FULL_FIELDS + field]);
        }
        uint suffix = field / ADDRESS_SUFFIX_FULL_BINS;
        uint chunk = field & (ADDRESS_SUFFIX_FULL_BINS - 1);
        output[(descriptor.output_start + suffix) * ADDRESS_SUFFIX_FULL_BINS + chunk] = sum;
    }
}
