#define BOOLEANITY_LANES 2u

struct BooleanityParams {
    uint rows;
    uint polys;
    uint k;
    uint branch_width;
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint materialize;
    ulong inc_bias;
    uint chunk_bits;
    uint reserved;
};

struct BooleanityBranchParams {
    uint polys;
    uint k;
    uint branch_width;
    uint reserved;
};

struct BooleanityReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

struct BooleanityLazySum {
    SolinasFp128 low;
    uint overflow;
};

struct BooleanityLazyWideSum {
    SolinasWide256 low;
    uint overflow;
};

inline BooleanityLazySum booleanity_lazy_zero()
{
    BooleanityLazySum sum = {};
    return sum;
}

inline BooleanityLazyWideSum booleanity_lazy_wide_zero()
{
    BooleanityLazyWideSum sum = {};
    return sum;
}

inline void booleanity_lazy_add(thread BooleanityLazySum& sum, SolinasFp128 value)
{
    ulong carry = 0ul;
    for (uint i = 0; i < 4; i++) {
        ulong word = (ulong)sum.low.limb[i] + (ulong)value.limb[i] + carry;
        sum.low.limb[i] = (uint)word;
        carry = word >> 32;
    }
    sum.overflow += (uint)carry;
}

inline void booleanity_lazy_wide_add(thread BooleanityLazyWideSum& sum, SolinasWide256 value)
{
    ulong carry = 0ul;
    for (uint i = 0; i < 8; i++) {
        ulong word = (ulong)sum.low.limb[i] + (ulong)value.limb[i] + carry;
        sum.low.limb[i] = (uint)word;
        carry = word >> 32;
    }
    sum.overflow += (uint)carry;
}

// A lazy sum is exactly low + overflow * 2^bits. overflow counts carries, at
// most one per add: below the materialization width (<= 32 here and <= 512 for
// instruction RA, checked by the sequence preparers) for h sums and below the
// u32-checked polynomial count for per-pair sums. With SOLINAS_OFFSET < 2^32 the
// correction overflow * SOLINAS_OFFSET^(bits / 128) is below 2^96, which
// solinas_add folds canonically next to any 128-bit left operand.
inline SolinasFp128 booleanity_lazy_reduce(BooleanityLazySum sum)
{
    ulong residue = (ulong)sum.overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)residue;
    correction.limb[1] = (uint)(residue >> 32);
    return solinas_add(sum.low, correction);
}

inline SolinasFp128 booleanity_lazy_wide_reduce(BooleanityLazyWideSum sum)
{
    ulong residue = (ulong)sum.overflow * (ulong)SOLINAS_OFFSET;
    ulong low = (residue & 0xfffffffful) * (ulong)SOLINAS_OFFSET;
    ulong high = (residue >> 32) * (ulong)SOLINAS_OFFSET + (low >> 32);
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)low;
    correction.limb[1] = (uint)high;
    correction.limb[2] = (uint)(high >> 32);
    return solinas_add(solinas_reduce(sum.low), correction);
}

// booleanity_hot_index over the resident source planes, loading only the
// source words its selector reads (layout: booleanity_row_word).
inline bool booleanity_row_hot_index(
    device const ulong* rows,
    uint row_count,
    uint row,
    BooleanitySelector selector,
    uint chunk_bits,
    ulong inc_bias,
    thread uint& hot)
{
    uint mask = (1u << chunk_bits) - 1u;
    if (selector.kind == 0u) {
        ulong word = booleanity_source_word(
            rows, row_count, selector.shift < 64u ? 0u : 1u, row);
        hot = (uint)(word >> (selector.shift & 63u)) & mask;
        return true;
    }
    ulong metadata = booleanity_source_word(rows, row_count, 3u, row);
    if (selector.kind == 1u) {
        return booleanity_offset_chunk(
            (metadata >> BOOLEANITY_SOURCE_PC_SHIFT) & BOOLEANITY_SOURCE_PC_MASK,
            selector.shift,
            mask,
            hot);
    }
    if (selector.kind == 2u) {
        return booleanity_offset_chunk(
            metadata & BOOLEANITY_SOURCE_RAM_MASK, selector.shift, mask, hot);
    }
    hot = booleanity_inc_chunk(
        booleanity_source_word(rows, row_count, 2u, row),
        ((metadata >> BOOLEANITY_SOURCE_FUSED_SIGN_SHIFT) & 1ul) != 0ul,
        selector,
        chunk_bits,
        inc_bias);
    return true;
}

// An offset-by-one source (one = 1) adds nothing for a zero (cold) row.
inline void booleanity_lazy_half_sums(
    device const uint* halves,
    device const SolinasFp128* table,
    uint original,
    uint branch_width,
    uint k,
    uint value_mask,
    uint one,
    uint shift,
    uint mask,
    thread BooleanityLazySum& lazy_0,
    thread BooleanityLazySum& lazy_1)
{
    for (uint offset = 0; offset < branch_width; offset++) {
        uint lo = halves[2u * (original + offset)] & value_mask;
        uint hi = halves[2u * (original + branch_width + offset)] & value_mask;
        if (lo >= one) {
            booleanity_lazy_add(lazy_0, table[((lo - one) >> shift) & mask]);
        }
        if (hi >= one) {
            booleanity_lazy_add(lazy_1, table[((hi - one) >> shift) & mask]);
        }
        table += k;
    }
}

inline void booleanity_lazy_pair(
    device const ulong* rows,
    device const BooleanitySelector* selectors,
    device const SolinasFp128* branches,
    device const SolinasFp128* rho,
    device const SolinasFp128* initial_constant,
    constant BooleanityParams& params,
    uint pair,
    device SolinasFp128* dense,
    thread SolinasFp128& constant_pair,
    thread SolinasFp128& leading_pair)
{
    BooleanityLazyWideSum leading = booleanity_lazy_wide_zero();
    if (params.branch_width == 1u && params.materialize == 0u) {
        BooleanityLazySum constant_sum = booleanity_lazy_zero();
        for (uint poly = 0; poly < params.polys; poly++) {
            BooleanitySelector selector = selectors[poly];
            uint first = params.k;
            uint second = params.k;
            booleanity_row_hot_index(
                rows, params.rows, 2u * pair, selector,
                params.chunk_bits, params.inc_bias, first);
            booleanity_row_hot_index(
                rows, params.rows, 2u * pair + 1u, selector,
                params.chunk_bits, params.inc_bias, second);
            booleanity_lazy_add(
                constant_sum,
                initial_constant[poly * (params.k + 1u) + first]);
            // Round 0 branches are the base tables, so derive the leading
            // coefficient here instead of gathering from a k^2 table.
            SolinasFp128 base_0 = first < params.k
                ? branches[poly * params.k + first]
                : solinas_zero();
            SolinasFp128 base_1 = second < params.k
                ? branches[poly * params.k + second]
                : solinas_zero();
            SolinasFp128 pair_delta = solinas_sub(base_1, base_0);
            booleanity_lazy_wide_add(leading, solinas_product_wide(pair_delta, pair_delta));
        }
        constant_pair = booleanity_lazy_reduce(constant_sum);
        leading_pair = booleanity_lazy_wide_reduce(leading);
        return;
    }
    BooleanityLazyWideSum constant_sum = booleanity_lazy_wide_zero();
    for (uint poly = 0; poly < params.polys; poly++) {
        BooleanitySelector selector = selectors[poly];
        BooleanityLazySum lazy_0 = booleanity_lazy_zero();
        BooleanityLazySum lazy_1 = booleanity_lazy_zero();
        uint original = 2u * pair * params.branch_width;
        uint mask = (1u << params.chunk_bits) - 1u;
        device const SolinasFp128* table = branches + poly * params.branch_width * params.k;
        if (selector.kind == 0u && (selector.shift & 31u) + params.chunk_bits <= 32u) {
            booleanity_lazy_half_sums(
                (device const uint*)(rows + (selector.shift < 64u ? 0u : params.rows))
                    + ((selector.shift >> 5) & 1u),
                table, original, params.branch_width, params.k,
                0xffffffffu, 0u, selector.shift & 31u, mask, lazy_0, lazy_1);
        } else if ((selector.kind == 1u || selector.kind == 2u) && selector.shift < 32u) {
            booleanity_lazy_half_sums(
                (device const uint*)(rows + 3u * params.rows) + (selector.kind == 1u ? 1u : 0u),
                table, original, params.branch_width, params.k,
                selector.kind == 1u ? (uint)BOOLEANITY_SOURCE_PC_MASK : 0xffffffffu,
                1u, selector.shift, mask, lazy_0, lazy_1);
        } else if (selector.kind >= 3u) {
            device const ulong* magnitudes = rows + 2u * params.rows;
            device const uint* metadata_high = (device const uint*)(rows + 3u * params.rows) + 1u;
            for (uint offset = 0; offset < params.branch_width; offset++) {
                uint lo_row = original + offset;
                uint hi_row = lo_row + params.branch_width;
                uint lo = booleanity_inc_chunk(
                    magnitudes[lo_row],
                    ((metadata_high[2u * lo_row] >> (BOOLEANITY_SOURCE_FUSED_SIGN_SHIFT - 32u)) & 1u) != 0u,
                    selector, params.chunk_bits, params.inc_bias);
                uint hi = booleanity_inc_chunk(
                    magnitudes[hi_row],
                    ((metadata_high[2u * hi_row] >> (BOOLEANITY_SOURCE_FUSED_SIGN_SHIFT - 32u)) & 1u) != 0u,
                    selector, params.chunk_bits, params.inc_bias);
                booleanity_lazy_add(lazy_0, table[lo]);
                booleanity_lazy_add(lazy_1, table[hi]);
                table += params.k;
            }
        } else {
            for (uint offset = 0; offset < params.branch_width; offset++) {
                uint hot;
                if (booleanity_row_hot_index(
                        rows, params.rows, original + offset, selector,
                        params.chunk_bits, params.inc_bias, hot)) {
                    booleanity_lazy_add(lazy_0, table[hot]);
                }
                if (booleanity_row_hot_index(
                        rows, params.rows, original + params.branch_width + offset, selector,
                        params.chunk_bits, params.inc_bias, hot)) {
                    booleanity_lazy_add(lazy_1, table[hot]);
                }
                table += params.k;
            }
        }
        SolinasFp128 h_0 = booleanity_lazy_reduce(lazy_0);
        SolinasFp128 h_1 = booleanity_lazy_reduce(lazy_1);
        if (params.materialize != 0u) {
            uint destination = poly * params.source_elements + 2u * pair;
            dense[destination] = h_0;
            dense[destination + 1u] = h_1;
        }
        SolinasFp128 delta = solinas_sub(h_1, h_0);
        booleanity_lazy_wide_add(
            constant_sum,
            solinas_product_wide(h_0, solinas_sub(h_0, rho[poly])));
        booleanity_lazy_wide_add(leading, solinas_product_wide(delta, delta));
    }
    constant_pair = booleanity_lazy_wide_reduce(constant_sum);
    leading_pair = booleanity_lazy_wide_reduce(leading);
}

inline void booleanity_finish_block(
    SolinasFp128 constant_lane,
    SolinasFp128 leading_lane,
    SolinasFp128 outer_weight,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane,
    uint simdgroup,
    uint simdgroups)
{
    if (lane == 0u) {
        shared[simdgroup] = constant_lane;
        shared[simdgroups + simdgroup] = leading_lane;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        SolinasFp128 constant_sum = lane < simdgroups
            ? shared[lane]
            : solinas_zero();
        SolinasFp128 leading_sum = lane < simdgroups
            ? shared[simdgroups + lane]
            : solinas_zero();
        constant_sum = solinas_simd_sum_32(constant_sum);
        leading_sum = solinas_simd_sum_32(leading_sum);
        if (lane == 0u) {
            partials[x_out] = solinas_mul_wide(outer_weight, constant_sum);
            partials[e_out_length + x_out] = solinas_mul_wide(outer_weight, leading_sum);
        }
    }
}

kernel void solinas_booleanity_lazy_message(
    device const ulong* rows [[buffer(0)]],
    device const BooleanitySelector* selectors [[buffer(1)]],
    device const SolinasFp128* branches [[buffer(2)]],
    device const SolinasFp128* rho [[buffer(3)]],
    device SolinasFp128* dense [[buffer(4)]],
    device const SolinasFp128* e_in [[buffer(5)]],
    device const SolinasFp128* e_out [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    device const SolinasFp128* initial_constant [[buffer(8)]],
    constant BooleanityParams& params [[buffer(9)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_out [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 constant_sum = solinas_zero();
    SolinasFp128 leading_sum = solinas_zero();
    for (uint x_in = thread_index; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 constant_pair = solinas_zero();
        SolinasFp128 leading_pair = solinas_zero();
        booleanity_lazy_pair(
            rows,
            selectors,
            branches,
            rho,
            initial_constant,
            params,
            pair,
            dense,
            constant_pair,
            leading_pair);
        SolinasFp128 weight = e_in[x_in];
        constant_sum = solinas_add(constant_sum, solinas_mul_wide(weight, constant_pair));
        leading_sum = solinas_add(leading_sum, solinas_mul_wide(weight, leading_pair));
    }
    booleanity_finish_block(
        solinas_simd_sum_32(constant_sum),
        solinas_simd_sum_32(leading_sum),
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_booleanity_double_branches(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant BooleanityBranchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint per_poly = params.branch_width * params.k;
    uint elements = params.polys * per_poly;
    if (gid >= elements) {
        return;
    }
    uint poly = gid / per_poly;
    uint within = gid - poly * per_poly;
    uint destination_base = poly * 2u * per_poly;
    SolinasFp128 value = source[gid];
    SolinasFp128 one = solinas_zero();
    one.limb[0] = 1u;
    SolinasFp128 one_minus = solinas_sub(one, challenge);
    destination[destination_base + within] = solinas_mul_wide(one_minus, value);
    destination[destination_base + per_poly + within] = solinas_mul_wide(challenge, value);
}

kernel void solinas_booleanity_dense_transition(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* rho [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
    constant BooleanityParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint simdgroups = threads / 32u;
    uint bound_elements = params.source_elements / 2u;
    SolinasFp128 constant_sum = solinas_zero();
    SolinasFp128 leading_sum = solinas_zero();
    for (uint x_in = simdgroup; x_in < params.e_in_length; x_in += simdgroups) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 constant_lane = solinas_zero();
        SolinasFp128 leading_lane = solinas_zero();
        for (uint poly = lane; poly < params.polys; poly += 32u) {
            uint source_base = poly * params.source_elements + 4u * pair;
            SolinasFp128 lo_0 = source[source_base];
            SolinasFp128 hi_0 = source[source_base + 1u];
            SolinasFp128 lo_1 = source[source_base + 2u];
            SolinasFp128 hi_1 = source[source_base + 3u];
            SolinasFp128 h_0 = solinas_add(
                lo_0,
                solinas_mul_wide(challenge, solinas_sub(hi_0, lo_0)));
            SolinasFp128 h_1 = solinas_add(
                lo_1,
                solinas_mul_wide(challenge, solinas_sub(hi_1, lo_1)));
            uint output = poly * bound_elements + 2u * pair;
            destination[output] = h_0;
            destination[output + 1u] = h_1;
            SolinasFp128 delta = solinas_sub(h_1, h_0);
            constant_lane = solinas_add(
                constant_lane,
                solinas_mul_wide(h_0, solinas_sub(h_0, rho[poly])));
            leading_lane = solinas_add(
                leading_lane,
                solinas_mul_wide(delta, delta));
        }
        constant_lane = solinas_simd_sum_32(constant_lane);
        leading_lane = solinas_simd_sum_32(leading_lane);
        if (lane == 0u) {
            constant_sum = solinas_add(
                constant_sum,
                solinas_mul_wide(e_in[x_in], constant_lane));
            leading_sum = solinas_add(
                leading_sum,
                solinas_mul_wide(e_in[x_in], leading_lane));
        }
    }
    booleanity_finish_block(
        constant_sum,
        leading_sum,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        simdgroups);
}

kernel void solinas_booleanity_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant BooleanityReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint value_index = 0; value_index < BOOLEANITY_LANES; value_index++) {
        SolinasFp128 value = gid < params.input_count
            ? input[value_index * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[value_index * params.output_count + gid / 32u] = value;
        }
    }
}
