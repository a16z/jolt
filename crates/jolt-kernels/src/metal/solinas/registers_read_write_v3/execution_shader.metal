#define REGISTERS_RW_V3_COLUMNS 128u
#define REGISTERS_RW_V3_SIMD_WIDTH 32u
#define REGISTERS_RW_V3_SIMDGROUPS 4u

struct RegistersRwV3DenseState {
    SolinasFp128 val;
    SolinasFp128 ra;
    SolinasFp128 wa;
};

struct RegistersRwV3RawRoundZeroParams {
    uint cycles;
    uint blocks;
    uint blocks_per_outer;
    uint e_in_length;
    uint e_out_length;
    uint columns;
    uint offset_stride;
    uint position_stride;
};

struct RegistersRwV3RawCoefficientParams {
    uint round;
    uint width;
    uint basis_weight_fields;
    uint strict_suffix_fields;
    uint local_weight_fields;
    uint coefficient_fields;
    uint logical_products;
    uint reserved;
};

struct RegistersRwV3RawReplayParams {
    uint round;
    uint cycles;
    uint blocks;
    uint width;
    uint remaining_cycles;
    uint nonempty_pairs;
    uint replay_e_in_length;
    uint e_out_length;
    uint columns;
    uint offset_stride;
    uint position_stride;
    uint flags;
};

struct RegistersRwV3DenseRoundParams {
    uint source_rows;
    uint destination_rows;
    uint pair_count;
    uint e_in_length;
    uint e_out_length;
    uint columns;
    uint round;
    uint reserved;
};

struct RegistersRwV3ReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

struct RegistersRwV3HistogramParams {
    uint cycles;
    uint blocks;
    uint blocks_per_hi;
    uint e_hi_length;
    uint e_lo_length;
    uint columns;
    uint offset_stride;
    uint position_stride;
};

inline SolinasFp128 registers_rw_v3_bind(
    SolinasFp128 lo,
    SolinasFp128 hi,
    SolinasFp128 challenge)
{
    return solinas_add(lo, solinas_mul_wide(challenge, solinas_sub(hi, lo)));
}

inline RegistersRwV3DenseState registers_rw_v3_bind_state(
    RegistersRwV3DenseState lo,
    RegistersRwV3DenseState hi,
    SolinasFp128 challenge)
{
    RegistersRwV3DenseState result;
    result.val = registers_rw_v3_bind(lo.val, hi.val, challenge);
    result.ra = registers_rw_v3_bind(lo.ra, hi.ra, challenge);
    result.wa = registers_rw_v3_bind(lo.wa, hi.wa, challenge);
    return result;
}

inline SolinasFp128 registers_rw_v3_simd_broadcast(
    SolinasFp128 value,
    ushort source_lane)
{
    SolinasFp128 result;
    result.limb.x = simd_broadcast(value.limb.x, source_lane);
    result.limb.y = simd_broadcast(value.limb.y, source_lane);
    result.limb.z = simd_broadcast(value.limb.z, source_lane);
    result.limb.w = simd_broadcast(value.limb.w, source_lane);
    return result;
}

kernel void solinas_registers_rw_v3_dense_bind_message(
    device const RegistersRwV3DenseState* state_source [[buffer(0)]],
    device const SolinasFp128* inc_source [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    constant SolinasFp128& challenge [[buffer(4)]],
    device RegistersRwV3DenseState* state_destination [[buffer(5)]],
    device SolinasFp128* inc_destination [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant RegistersRwV3DenseRoundParams& params [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    SolinasFp128 outer_zero = solinas_zero();
    SolinasFp128 outer_infinity = solinas_zero();

    for (uint x_in = simdgroup;
         x_in < params.e_in_length;
         x_in += REGISTERS_RW_V3_SIMDGROUPS) {
        uint pair = x_out * params.e_in_length + x_in;
        uint source_row = 4u * pair;
        SolinasFp128 inc_zero = solinas_zero();
        SolinasFp128 inc_one = solinas_zero();
        if (lane == 0u) {
            inc_zero = registers_rw_v3_bind(
                inc_source[source_row],
                inc_source[source_row + 1u],
                challenge);
            inc_one = registers_rw_v3_bind(
                inc_source[source_row + 2u],
                inc_source[source_row + 3u],
                challenge);
            inc_destination[2u * pair] = inc_zero;
            inc_destination[2u * pair + 1u] = inc_one;
        }
        inc_zero = registers_rw_v3_simd_broadcast(inc_zero, 0);
        inc_one = registers_rw_v3_simd_broadcast(inc_one, 0);

        SolinasFp128 q_zero = solinas_zero();
        SolinasFp128 q_infinity = solinas_zero();
        SolinasFp128 wa_zero = solinas_zero();
        SolinasFp128 wa_infinity = solinas_zero();
        for (uint bank = 0u; bank < 4u; bank++) {
            uint column = lane + bank * REGISTERS_RW_V3_SIMD_WIDTH;
            uint source_index = source_row * params.columns + column;
            RegistersRwV3DenseState even = registers_rw_v3_bind_state(
                state_source[source_index],
                state_source[source_index + params.columns],
                challenge);
            RegistersRwV3DenseState odd = registers_rw_v3_bind_state(
                state_source[source_index + 2u * params.columns],
                state_source[source_index + 3u * params.columns],
                challenge);
            uint destination_index = 2u * pair * params.columns + column;
            state_destination[destination_index] = even;
            state_destination[destination_index + params.columns] = odd;

            SolinasFp128 val_slope = solinas_sub(odd.val, even.val);
            SolinasFp128 ra_slope = solinas_sub(odd.ra, even.ra);
            SolinasFp128 wa_slope = solinas_sub(odd.wa, even.wa);
            q_zero = solinas_add(
                q_zero,
                solinas_mul_wide(solinas_add(even.ra, even.wa), even.val));
            q_infinity = solinas_add(
                q_infinity,
                solinas_mul_wide(solinas_add(ra_slope, wa_slope), val_slope));
            wa_zero = solinas_add(wa_zero, even.wa);
            wa_infinity = solinas_add(wa_infinity, wa_slope);
        }

        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        wa_zero = solinas_simd_sum_32(wa_zero);
        wa_infinity = solinas_simd_sum_32(wa_infinity);
        if (lane == 0u) {
            q_zero = solinas_add(
                q_zero,
                solinas_mul_wide(inc_zero, wa_zero));
            q_infinity = solinas_add(
                q_infinity,
                solinas_mul_wide(solinas_sub(inc_one, inc_zero), wa_infinity));
            SolinasFp128 inner_weight = e_in[x_in];
            outer_zero = solinas_add(
                outer_zero,
                solinas_mul_wide(inner_weight, q_zero));
            outer_infinity = solinas_add(
                outer_infinity,
                solinas_mul_wide(inner_weight, q_infinity));
        }
    }

    if (lane == 0u) {
        shared[2u * simdgroup] = outer_zero;
        shared[2u * simdgroup + 1u] = outer_infinity;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        SolinasFp128 total_zero = solinas_zero();
        SolinasFp128 total_infinity = solinas_zero();
        for (uint group = 0u; group < REGISTERS_RW_V3_SIMDGROUPS; group++) {
            total_zero = solinas_add(total_zero, shared[2u * group]);
            total_infinity = solinas_add(total_infinity, shared[2u * group + 1u]);
        }
        SolinasFp128 outer_weight = e_out[x_out];
        partials[x_out] = solinas_mul_wide(outer_weight, total_zero);
        partials[params.e_out_length + x_out] =
            solinas_mul_wide(outer_weight, total_infinity);
    }
}

kernel void solinas_registers_rw_v3_reduce_columns(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RegistersRwV3ReductionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (gid.y >= params.columns) {
        return;
    }
    SolinasFp128 value = gid.x < params.input_count
        ? input[gid.y * params.input_count + gid.x]
        : solinas_zero();
    value = solinas_simd_sum_32(value);
    if (lane == 0u && gid.x / REGISTERS_RW_V3_SIMD_WIDTH < params.output_count) {
        output[gid.y * params.output_count
            + gid.x / REGISTERS_RW_V3_SIMD_WIDTH] = value;
    }
}

kernel void solinas_registers_rw_v3_histogram(
    device const ushort* rs1_offsets [[buffer(0)]],
    device const uchar* rs1_positions [[buffer(1)]],
    device const ushort* rs2_offsets [[buffer(2)]],
    device const uchar* rs2_positions [[buffer(3)]],
    device const SolinasFp128* e_hi [[buffer(4)]],
    device const SolinasFp128* e_lo [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant RegistersRwV3HistogramParams& params [[buffer(7)]],
    uint column [[thread_index_in_threadgroup]],
    uint hi_index [[threadgroup_position_in_grid]])
{
    if (column >= params.columns || hi_index >= params.e_hi_length) {
        return;
    }
    SolinasFp128 rs1_claim = solinas_zero();
    SolinasFp128 rs2_claim = solinas_zero();
    for (uint block_in_hi = 0u;
         block_in_hi < params.blocks_per_hi;
         block_in_hi++) {
        uint block = hi_index * params.blocks_per_hi + block_in_hi;
        uint offset_base = block * params.offset_stride;
        uint position_base = block * params.position_stride;
        uint cycle_base = block_in_hi * params.position_stride;

        uint rs1_start = (uint)rs1_offsets[offset_base + column];
        uint rs1_end = (uint)rs1_offsets[offset_base + column + 1u];
        for (uint event = rs1_start; event < rs1_end; event++) {
            uint cycle_low = cycle_base + (uint)rs1_positions[position_base + event];
            rs1_claim = solinas_add(rs1_claim, e_lo[cycle_low]);
        }

        uint rs2_start = (uint)rs2_offsets[offset_base + column];
        uint rs2_end = (uint)rs2_offsets[offset_base + column + 1u];
        for (uint event = rs2_start; event < rs2_end; event++) {
            uint cycle_low = cycle_base + (uint)rs2_positions[position_base + event];
            rs2_claim = solinas_add(rs2_claim, e_lo[cycle_low]);
        }
    }

    SolinasFp128 high_weight = e_hi[hi_index];
    partials[column * params.e_hi_length + hi_index] =
        solinas_mul_wide(high_weight, rs1_claim);
    partials[(params.columns + column) * params.e_hi_length + hi_index] =
        solinas_mul_wide(high_weight, rs2_claim);
}

#undef REGISTERS_RW_V3_SIMDGROUPS
#undef REGISTERS_RW_V3_SIMD_WIDTH
#undef REGISTERS_RW_V3_COLUMNS
