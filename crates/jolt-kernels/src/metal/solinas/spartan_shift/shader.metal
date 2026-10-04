// Concatenate after fp128.metal, simd_reduce.metal and spartan_outer_common.metal.

#define SPARTAN_SHIFT_PREFIX_PAIRS 4u
#define SPARTAN_SHIFT_OUTPUT_COLUMNS 5u
// Compact-row flag bits; FLAG_* in spartan_outer_uniskip/mod.rs (keep in sync).
#define SPARTAN_SHIFT_FLAG_VIRTUAL 9u
#define SPARTAN_SHIFT_FLAG_IS_FIRST 24u
#define SPARTAN_SHIFT_FLAG_IS_NOOP 27u

struct SpartanShiftParams {
    uint prefix_elements;
    uint suffix_elements;
    uint high_tile_elements;
    uint high_tiles;
};

struct SpartanShiftWide192 {
    uint limb[6];
};

struct SpartanShiftNativeValue {
    ulong unexpanded_pc;
    ulong pc;
    bool is_virtual;
    bool is_first_in_sequence;
    bool is_noop;
};

inline SolinasFp128 spartan_shift_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SpartanShiftWide192 spartan_shift_product_u64(
    SolinasFp128 lhs,
    ulong rhs)
{
    uint rhs_limb[2] = {(uint)rhs, (uint)(rhs >> 32)};
    SpartanShiftWide192 product;
    for (uint i = 0u; i < 6u; i++) {
        product.limb[i] = 0u;
    }
    for (uint i = 0u; i < 4u; i++) {
        ulong carry = 0ul;
        for (uint j = 0u; j < 2u; j++) {
            uint k = i + j;
            ulong word = (ulong)lhs.limb[i] * (ulong)rhs_limb[j]
                + (ulong)product.limb[k]
                + carry;
            product.limb[k] = (uint)word;
            carry = word >> 32;
        }
        product.limb[i + 2u] = (uint)carry;
    }
    return product;
}

inline SolinasFp128 spartan_shift_reduce_u192(SpartanShiftWide192 product) {
    SolinasFp128 folded;
    ulong carry = 0ul;
    for (uint i = 0u; i < 2u; i++) {
        ulong word = (ulong)product.limb[i + 4u] * (ulong)SOLINAS_OFFSET
            + (ulong)product.limb[i]
            + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }
    for (uint i = 2u; i < 4u; i++) {
        ulong word = (ulong)product.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }

    ulong word = (ulong)folded.limb[0] + carry * (ulong)SOLINAS_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32;
    for (uint i = 1u; i < 4u; i++) {
        word = (ulong)folded.limb[i] + carry;
        folded.limb[i] = (uint)word;
        carry = word >> 32;
    }

    SolinasCorrection corrected = solinas_add_offset(folded);
    return solinas_select(
        carry != 0ul || corrected.carry != 0u,
        corrected.value,
        folded);
}

inline SolinasFp128 spartan_shift_mul_u64(SolinasFp128 lhs, ulong rhs) {
    return spartan_shift_reduce_u192(spartan_shift_product_u64(lhs, rhs));
}

inline SpartanShiftNativeValue spartan_shift_native_value(
    device const InstructionInputRow* compact,
    device const SpartanRawRow* raw,
    uint row)
{
    ulong flags = instruction_input_row_word(compact[row], 5u);
    SpartanShiftNativeValue value;
    value.unexpanded_pc = instruction_input_row_word(compact[row], 1u);
    value.pc = spartan_raw_row_word(raw[row], 0u);
    value.is_virtual = ((flags >> SPARTAN_SHIFT_FLAG_VIRTUAL) & 1ul) != 0ul;
    value.is_first_in_sequence = ((flags >> SPARTAN_SHIFT_FLAG_IS_FIRST) & 1ul) != 0ul;
    value.is_noop = ((flags >> SPARTAN_SHIFT_FLAG_IS_NOOP) & 1ul) != 0ul;
    return value;
}

inline SolinasFp128 spartan_shift_outer_mixed(
    SpartanShiftNativeValue native,
    device const SolinasFp128* gamma_powers)
{
    SolinasFp128 value = spartan_shift_from_u64(native.unexpanded_pc);
    value = solinas_add(value, spartan_shift_mul_u64(gamma_powers[0], native.pc));
    if (native.is_virtual) {
        value = solinas_add(value, gamma_powers[1]);
    }
    if (native.is_first_in_sequence) {
        value = solinas_add(value, gamma_powers[2]);
    }
    return value;
}

kernel void solinas_spartan_shift_build_mixed_partials(
    device const InstructionInputRow* compact [[buffer(0)]],
    device const SpartanRawRow* raw [[buffer(1)]],
    device const SolinasFp128* gamma_powers [[buffer(2)]],
    device const SolinasFp128* high_weights [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SpartanShiftParams& params [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint low_groups = (params.prefix_elements + threads - 1u) / threads;
    uint high_tile = group / low_groups;
    uint low_group = group - high_tile * low_groups;
    uint x_lo = low_group * threads + tid;
    if (high_tile >= params.high_tiles || x_lo >= params.prefix_elements) {
        return;
    }

    uint high_start = high_tile * params.high_tile_elements;
    uint high_end = min(
        high_start + params.high_tile_elements,
        params.suffix_elements);
    uint row = high_start * params.prefix_elements + x_lo;
    SpartanShiftNativeValue current_native = spartan_shift_native_value(compact, raw, row);
    SolinasFp128 current = spartan_shift_outer_mixed(current_native, gamma_powers);
    bool current_noop = current_native.is_noop;
    SolinasFp128 outer_current = solinas_zero();
    SolinasFp128 outer_successor = solinas_zero();
    SolinasFp128 product_current = solinas_zero();
    SolinasFp128 product_successor = solinas_zero();

    for (uint high = high_start; high < high_end; high++) {
        bool has_next = high + 1u < params.suffix_elements;
        SolinasFp128 next = solinas_zero();
        bool next_noop = true;
        if (has_next) {
            uint next_row = (high + 1u) * params.prefix_elements + x_lo;
            SpartanShiftNativeValue next_native =
                spartan_shift_native_value(compact, raw, next_row);
            next = spartan_shift_outer_mixed(next_native, gamma_powers);
            next_noop = next_native.is_noop;
        }

        SolinasFp128 outer_weight = high_weights[high];
        SolinasFp128 product_weight =
            high_weights[params.suffix_elements + high];
        outer_current = solinas_add(
            outer_current, solinas_mul_wide(outer_weight, current));
        if (has_next) {
            outer_successor = solinas_add(
                outer_successor, solinas_mul_wide(outer_weight, next));
        }
        if (!current_noop) {
            product_current = solinas_add(product_current, product_weight);
        }
        if (has_next && !next_noop) {
            product_successor = solinas_add(product_successor, product_weight);
        }
        current = next;
        current_noop = next_noop;
    }

    uint partial_count = params.prefix_elements * params.high_tiles;
    uint partial = high_tile * params.prefix_elements + x_lo;
    partials[partial] = outer_current;
    partials[partial_count + partial] = outer_successor;
    partials[2u * partial_count + partial] = product_current;
    partials[3u * partial_count + partial] = product_successor;
}

kernel void solinas_spartan_shift_reduce_prefix(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* q [[buffer(1)]],
    constant SpartanShiftParams& params [[buffer(2)]],
    uint x_lo [[thread_position_in_grid]])
{
    if (x_lo >= params.prefix_elements) {
        return;
    }
    uint partial_count = params.prefix_elements * params.high_tiles;
    for (uint column = 0u; column < SPARTAN_SHIFT_PREFIX_PAIRS; column++) {
        SolinasFp128 sum = solinas_zero();
        uint column_start = column * partial_count;
        for (uint tile = 0u; tile < params.high_tiles; tile++) {
            sum = solinas_add(
                sum,
                partials[column_start + tile * params.prefix_elements + x_lo]);
        }
        q[column * params.prefix_elements + x_lo] = sum;
    }
}

kernel void solinas_spartan_shift_fold_native(
    device const InstructionInputRow* compact [[buffer(0)]],
    device const SpartanRawRow* raw [[buffer(1)]],
    device const SolinasFp128* low_weights [[buffer(2)]],
    device SolinasFp128* dense_outputs [[buffer(3)]],
    constant SpartanShiftParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_hi >= params.suffix_elements) {
        return;
    }
    SolinasFp128 sums[SPARTAN_SHIFT_OUTPUT_COLUMNS];
    for (uint column = 0u; column < SPARTAN_SHIFT_OUTPUT_COLUMNS; column++) {
        sums[column] = solinas_zero();
    }

    uint row_start = x_hi * params.prefix_elements;
    for (uint x_lo = tid; x_lo < params.prefix_elements; x_lo += threads) {
        uint row = row_start + x_lo;
        SolinasFp128 weight = low_weights[x_lo];
        SpartanShiftNativeValue native = spartan_shift_native_value(compact, raw, row);
        sums[0] = solinas_add(
            sums[0], spartan_shift_mul_u64(weight, native.unexpanded_pc));
        sums[1] = solinas_add(sums[1], spartan_shift_mul_u64(weight, native.pc));
        if (native.is_virtual) {
            sums[2] = solinas_add(sums[2], weight);
        }
        if (native.is_first_in_sequence) {
            sums[3] = solinas_add(sums[3], weight);
        }
        if (native.is_noop) {
            sums[4] = solinas_add(sums[4], weight);
        }
    }

    uint simdgroups = threads / 32u;
    for (uint column = 0u; column < SPARTAN_SHIFT_OUTPUT_COLUMNS; column++) {
        SolinasFp128 sum = solinas_simd_sum_32(sums[column]);
        if (lane == 0u) {
            shared[column * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane < SPARTAN_SHIFT_OUTPUT_COLUMNS) {
        SolinasFp128 total = solinas_zero();
        for (uint group = 0u; group < simdgroups; group++) {
            total = solinas_add(total, shared[lane * simdgroups + group]);
        }
        dense_outputs[lane * params.suffix_elements + x_hi] = total;
    }
}
