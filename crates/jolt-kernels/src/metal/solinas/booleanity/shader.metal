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

inline ulong booleanity_broadcast_ulong(ulong value, ushort source_lane)
{
    uint lo = simd_broadcast((uint)value, source_lane);
    uint hi = simd_broadcast((uint)(value >> 32), source_lane);
    return ((ulong)hi << 32) | (ulong)lo;
}

inline BooleanityRow booleanity_load_row_simd(
    device const ulong* rows,
    uint row_count,
    uint row,
    uint lane)
{
    BooleanityRow value;
    value.lookup_lo = booleanity_broadcast_ulong(
        lane == 0 ? booleanity_row_word(rows, row_count, 0u, row) : 0ul,
        0);
    value.lookup_hi = booleanity_broadcast_ulong(
        lane == 1 ? booleanity_row_word(rows, row_count, 1u, row) : 0ul,
        1);
    value.ram_address_plus_one = booleanity_broadcast_ulong(
        lane == 2 ? booleanity_row_word(rows, row_count, 2u, row) : 0ul,
        2);
    value.fused_inc_magnitude = booleanity_broadcast_ulong(
        lane == 3 ? booleanity_row_word(rows, row_count, 3u, row) : 0ul,
        3);
    value.packed_pc_and_flags = booleanity_broadcast_ulong(
        lane == 4 ? booleanity_row_word(rows, row_count, 4u, row) : 0ul,
        4);
    return value;
}

inline void booleanity_lazy_pair(
    device const ulong* rows,
    device const BooleanitySelector* selectors,
    device const SolinasFp128* branches,
    device const SolinasFp128* rho,
    device const SolinasFp128* initial_constant,
    constant BooleanityParams& params,
    uint pair,
    uint lane,
    device SolinasFp128* dense,
    thread SolinasFp128& constant_lane,
    thread SolinasFp128& leading_lane)
{
    for (uint poly = lane; poly < params.polys; poly += 32u) {
        if (params.branch_width == 1u && params.materialize == 0u) {
            BooleanitySelector selector = selectors[poly];
            BooleanityRow row_0 = booleanity_load_row_simd(
                rows, params.rows, 2u * pair, lane);
            BooleanityRow row_1 = booleanity_load_row_simd(
                rows, params.rows, 2u * pair + 1u, lane);
            uint first = params.k;
            uint second = params.k;
            booleanity_hot_index(
                row_0, selector, params.chunk_bits, params.inc_bias, first);
            booleanity_hot_index(
                row_1, selector, params.chunk_bits, params.inc_bias, second);
            uint stride = params.k + 1u;
            constant_lane = solinas_add(
                constant_lane,
                initial_constant[poly * stride + first]);
            // Round 0 branches are the base tables, so derive the leading
            // coefficient here instead of gathering from a k^2 table.
            SolinasFp128 base_0 = first < params.k
                ? branches[poly * params.k + first]
                : solinas_zero();
            SolinasFp128 base_1 = second < params.k
                ? branches[poly * params.k + second]
                : solinas_zero();
            SolinasFp128 pair_delta = solinas_sub(base_1, base_0);
            leading_lane = solinas_add(
                leading_lane, solinas_mul_wide(pair_delta, pair_delta));
            continue;
        }
        SolinasFp128 h_0 = solinas_zero();
        SolinasFp128 h_1 = solinas_zero();
        BooleanitySelector selector = selectors[poly];
        uint original = 2u * pair * params.branch_width;
        for (uint offset = 0; offset < params.branch_width; offset++) {
            BooleanityRow row_0 = booleanity_load_row_simd(
                rows, params.rows, original + offset, lane);
            BooleanityRow row_1 = booleanity_load_row_simd(
                rows,
                params.rows,
                original + params.branch_width + offset,
                lane);
            uint hot;
            uint table = (poly * params.branch_width + offset) * params.k;
            if (booleanity_hot_index(
                    row_0, selector, params.chunk_bits, params.inc_bias, hot)) {
                h_0 = solinas_add(h_0, branches[table + hot]);
            }
            if (booleanity_hot_index(
                    row_1, selector, params.chunk_bits, params.inc_bias, hot)) {
                h_1 = solinas_add(h_1, branches[table + hot]);
            }
        }
        if (params.materialize != 0u) {
            uint destination = poly * params.source_elements + 2u * pair;
            dense[destination] = h_0;
            dense[destination + 1u] = h_1;
        }
        SolinasFp128 delta = solinas_sub(h_1, h_0);
        constant_lane = solinas_add(
            constant_lane,
            solinas_mul_wide(h_0, solinas_sub(h_0, rho[poly])));
        leading_lane = solinas_add(leading_lane, solinas_mul_wide(delta, delta));
    }
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
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint simdgroups = threads / 32u;
    SolinasFp128 constant_sum = solinas_zero();
    SolinasFp128 leading_sum = solinas_zero();
    for (uint x_in = simdgroup; x_in < params.e_in_length; x_in += simdgroups) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 constant_lane = solinas_zero();
        SolinasFp128 leading_lane = solinas_zero();
        booleanity_lazy_pair(
            rows,
            selectors,
            branches,
            rho,
            initial_constant,
            params,
            pair,
            lane,
            dense,
            constant_lane,
            leading_lane);
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
