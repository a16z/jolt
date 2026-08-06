#define REGISTERS_RW_DENSE_COLUMNS 128u
#define REGISTERS_RW_DENSE_BLOCK_CYCLES 256u
#define REGISTERS_RW_DENSE_OFFSET_STRIDE 129u

struct RegistersRwDenseState {
    SolinasFp128 val;
    SolinasFp128 ra;
    SolinasFp128 wa;
};

struct RegistersRwDensePhaseParams {
    uint source_rows;
    uint destination_rows;
    uint pair_count;
    uint e_in_length;
    uint e_out_length;
    uint columns;
    uint2 reserved;
};

struct RegistersRwDenseReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

struct RegistersRwDenseOutputParams {
    uint cycles;
    uint blocks;
    uint blocks_per_hi;
    uint block_columns;
    uint e_hi_length;
    uint e_lo_length;
    uint address_bits;
    uint reserved;
};

inline SolinasFp128 registers_rw_dense_bind(
    SolinasFp128 lo,
    SolinasFp128 hi,
    SolinasFp128 challenge)
{
    return solinas_add(lo, solinas_mul_wide(challenge, solinas_sub(hi, lo)));
}

inline RegistersRwDenseState registers_rw_dense_bind_state(
    RegistersRwDenseState lo,
    RegistersRwDenseState hi,
    SolinasFp128 challenge)
{
    RegistersRwDenseState result;
    result.val = registers_rw_dense_bind(lo.val, hi.val, challenge);
    result.ra = registers_rw_dense_bind(lo.ra, hi.ra, challenge);
    result.wa = registers_rw_dense_bind(lo.wa, hi.wa, challenge);
    return result;
}

kernel void solinas_registers_rw_dense_bind_message_p1(
    device const RegistersRwDenseState* source [[buffer(0)]],
    device const SolinasFp128* source_inc [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    constant SolinasFp128& challenge [[buffer(4)]],
    device RegistersRwDenseState* destination [[buffer(5)]],
    device SolinasFp128* destination_inc [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant RegistersRwDensePhaseParams& params [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint simdgroups = threads / 32u;
    SolinasFp128 outer_zero = solinas_zero();
    SolinasFp128 outer_infinity = solinas_zero();

    for (uint x_in = 0u; x_in < params.e_in_length; x_in++) {
        uint pair = x_out * params.e_in_length + x_in;
        if (tid == 0u) {
            uint source_row = 4u * pair;
            shared[0] = registers_rw_dense_bind(
                source_inc[source_row], source_inc[source_row + 1u], challenge);
            shared[1] = registers_rw_dense_bind(
                source_inc[source_row + 2u], source_inc[source_row + 3u], challenge);
            destination_inc[2u * pair] = shared[0];
            destination_inc[2u * pair + 1u] = shared[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        SolinasFp128 q_zero = solinas_zero();
        SolinasFp128 q_infinity = solinas_zero();
        SolinasFp128 wa_zero = solinas_zero();
        SolinasFp128 wa_infinity = solinas_zero();
        if (tid < params.columns) {
            uint source_row = 4u * pair;
            uint source_index = source_row * params.columns + tid;
            RegistersRwDenseState even = registers_rw_dense_bind_state(
                source[source_index], source[source_index + params.columns], challenge);
            RegistersRwDenseState odd = registers_rw_dense_bind_state(
                source[source_index + 2u * params.columns],
                source[source_index + 3u * params.columns],
                challenge);
            uint destination_index = (2u * pair) * params.columns + tid;
            destination[destination_index] = even;
            destination[destination_index + params.columns] = odd;

            SolinasFp128 val_m = solinas_sub(odd.val, even.val);
            SolinasFp128 ra_m = solinas_sub(odd.ra, even.ra);
            SolinasFp128 wa_m = solinas_sub(odd.wa, even.wa);
            q_zero = solinas_mul_wide(solinas_add(even.ra, even.wa), even.val);
            q_infinity = solinas_mul_wide(solinas_add(ra_m, wa_m), val_m);
            wa_zero = even.wa;
            wa_infinity = wa_m;
        }

        q_zero = solinas_simd_sum_32(q_zero);
        q_infinity = solinas_simd_sum_32(q_infinity);
        wa_zero = solinas_simd_sum_32(wa_zero);
        wa_infinity = solinas_simd_sum_32(wa_infinity);
        if (lane == 0u) {
            shared[2u + simdgroup] = q_zero;
            shared[2u + simdgroups + simdgroup] = q_infinity;
            shared[2u + 2u * simdgroups + simdgroup] = wa_zero;
            shared[2u + 3u * simdgroups + simdgroup] = wa_infinity;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            SolinasFp128 group_zero = solinas_zero();
            SolinasFp128 group_infinity = solinas_zero();
            SolinasFp128 group_wa_zero = solinas_zero();
            SolinasFp128 group_wa_infinity = solinas_zero();
            for (uint group = 0u; group < simdgroups; group++) {
                group_zero = solinas_add(group_zero, shared[2u + group]);
                group_infinity = solinas_add(
                    group_infinity, shared[2u + simdgroups + group]);
                group_wa_zero = solinas_add(
                    group_wa_zero, shared[2u + 2u * simdgroups + group]);
                group_wa_infinity = solinas_add(
                    group_wa_infinity, shared[2u + 3u * simdgroups + group]);
            }
            SolinasFp128 inc_zero = shared[0];
            SolinasFp128 inc_infinity = solinas_sub(shared[1], inc_zero);
            group_zero = solinas_add(
                group_zero, solinas_mul_wide(inc_zero, group_wa_zero));
            group_infinity = solinas_add(
                group_infinity,
                solinas_mul_wide(inc_infinity, group_wa_infinity));
            SolinasFp128 inner_weight = e_in[x_in];
            outer_zero = solinas_add(
                outer_zero, solinas_mul_wide(inner_weight, group_zero));
            outer_infinity = solinas_add(
                outer_infinity, solinas_mul_wide(inner_weight, group_infinity));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        SolinasFp128 outer_weight = e_out[x_out];
        partials[x_out] = solinas_mul_wide(outer_weight, outer_zero);
        partials[params.e_out_length + x_out] =
            solinas_mul_wide(outer_weight, outer_infinity);
    }
}

kernel void solinas_registers_rw_dense_reduce2(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RegistersRwDenseReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint column = 0u; column < 2u; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[column * params.output_count + gid / 32u] = value;
        }
    }
}

kernel void solinas_registers_rw_dense_output_claims_p1(
    device const uint* rs1_offsets [[buffer(0)]],
    device const uint* rs2_offsets [[buffer(1)]],
    device const uchar* rs1_positions [[buffer(2)]],
    device const uchar* rs2_positions [[buffer(3)]],
    device const SolinasFp128* e_hi [[buffer(4)]],
    device const SolinasFp128* e_lo [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant RegistersRwDenseOutputParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    threadgroup uint* offset_shared [[threadgroup(1)]],
    uint column [[thread_index_in_threadgroup]],
    uint hi_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 claims[2] = {solinas_zero(), solinas_zero()};
    for (uint block_in_hi = 0u;
         block_in_hi < params.blocks_per_hi;
         block_in_hi++) {
        uint block = hi_index * params.blocks_per_hi + block_in_hi;
        uint header_base = block * REGISTERS_RW_DENSE_COLUMNS;
        if (column < REGISTERS_RW_DENSE_COLUMNS) {
            uint header = header_base + column;
            offset_shared[column] = rs1_offsets[header];
            offset_shared[REGISTERS_RW_DENSE_OFFSET_STRIDE + column] =
                rs2_offsets[header];
        }
        if (column == 0u) {
            offset_shared[REGISTERS_RW_DENSE_COLUMNS] =
                rs1_offsets[header_base + REGISTERS_RW_DENSE_COLUMNS];
            offset_shared[REGISTERS_RW_DENSE_OFFSET_STRIDE +
                REGISTERS_RW_DENSE_COLUMNS] =
                rs2_offsets[header_base + REGISTERS_RW_DENSE_COLUMNS];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (column < REGISTERS_RW_DENSE_COLUMNS) {
            uint cycle_base = block_in_hi * REGISTERS_RW_DENSE_BLOCK_CYCLES;
            uint starts[2] = {
                offset_shared[column],
                offset_shared[REGISTERS_RW_DENSE_OFFSET_STRIDE + column]
            };
            uint ends[2] = {
                offset_shared[column + 1u],
                offset_shared[REGISTERS_RW_DENSE_OFFSET_STRIDE + column + 1u]
            };
            for (uint operand = 0u; operand < 2u; operand++) {
                device const uchar* positions = operand == 0u
                    ? rs1_positions
                    : rs2_positions;
                for (uint event = starts[operand]; event < ends[operand]; event++) {
                    uint cycle_low = cycle_base + (uint)positions[event];
                    uint low_index = (cycle_low << params.address_bits) | column;
                    claims[operand] = solinas_add(claims[operand], e_lo[low_index]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint simdgroups = threads / 32u;
    for (uint operand = 0u; operand < 2u; operand++) {
        claims[operand] = solinas_simd_sum_32(claims[operand]);
        if (lane == 0u) {
            shared[operand * simdgroups + simdgroup] = claims[operand];
        }
    }
    if (column == 0u) {
        shared[2u * simdgroups] = e_hi[hi_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u && lane < 2u) {
        SolinasFp128 sum = solinas_zero();
        for (uint group = 0u; group < simdgroups; group++) {
            sum = solinas_add(sum, shared[lane * simdgroups + group]);
        }
        partials[lane * params.e_hi_length + hi_index] =
            solinas_mul_wide(shared[2u * simdgroups], sum);
    }
}

#undef REGISTERS_RW_DENSE_BLOCK_CYCLES
#undef REGISTERS_RW_DENSE_COLUMNS
#undef REGISTERS_RW_DENSE_OFFSET_STRIDE
