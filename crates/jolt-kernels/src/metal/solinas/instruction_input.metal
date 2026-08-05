#define INSTRUCTION_INPUT_TABLES 8u
#define INSTRUCTION_INPUT_COEFFICIENTS 3u

#define INSTRUCTION_INPUT_FLAG_LOAD 0u
#define INSTRUCTION_INPUT_FLAG_IMM_POSITIVE 18u
#define INSTRUCTION_INPUT_FLAG_LEFT_IS_RS1 20u
#define INSTRUCTION_INPUT_FLAG_LEFT_IS_PC 21u
#define INSTRUCTION_INPUT_FLAG_RIGHT_IS_RS2 22u
#define INSTRUCTION_INPUT_FLAG_RIGHT_IS_IMM 23u

struct InstructionInputParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

struct InstructionInputReductionParams {
    uint input_count;
    uint output_count;
    uint2 reserved;
};

inline SolinasFp128 instruction_input_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 instruction_input_from_i128(
    ulong low,
    ulong high,
    bool positive)
{
    SpartanSigned192 value = spartan_scaled_u128(low, high, 1u);
    if (!positive) {
        value = spartan_s192_negate(value);
    }
    return spartan_small_times_s192(1, value);
}

inline int instruction_input_flag(
    device const SpartanOuterUniskipRow& row,
    uint bit)
{
    return (int)((row.words[19] >> bit) & 1ul);
}

inline ulong instruction_input_rs2(device const SpartanOuterUniskipRow& row) {
    return instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_LOAD) != 0
        ? 0ul
        : row.words[10];
}

inline void instruction_input_accumulate_u64(
    thread SpartanSigned192* descriptors,
    int flag_0,
    int flag_1,
    ulong value_0,
    ulong value_1)
{
    spartan_accumulate_scaled_u64(descriptors[0], value_0, flag_0);
    spartan_accumulate_scaled_u64(descriptors[1], value_1, flag_1);
    int flag_step = flag_1 - flag_0;
    spartan_accumulate_scaled_u64(descriptors[2], value_1, flag_step);
    spartan_accumulate_scaled_u64(descriptors[2], value_0, -flag_step);
}

inline void instruction_input_accumulate_i128(
    thread SpartanSigned192* descriptors,
    int flag_0,
    int flag_1,
    ulong low_0,
    ulong high_0,
    bool positive_0,
    ulong low_1,
    ulong high_1,
    bool positive_1)
{
    spartan_accumulate_scaled_u128(
        descriptors[0], low_0, high_0, positive_0, flag_0);
    spartan_accumulate_scaled_u128(
        descriptors[1], low_1, high_1, positive_1, flag_1);
    int flag_step = flag_1 - flag_0;
    spartan_accumulate_scaled_u128(
        descriptors[2], low_1, high_1, positive_1, flag_step);
    spartan_accumulate_scaled_u128(
        descriptors[2], low_0, high_0, positive_0, -flag_step);
}

inline void instruction_input_native_pair(
    device const SpartanOuterUniskipRow& row_0,
    device const SpartanOuterUniskipRow& row_1,
    SolinasFp128 gamma,
    SolinasFp128 weight,
    thread SolinasFp128* lanes)
{
    SpartanSigned192 right[INSTRUCTION_INPUT_COEFFICIENTS];
    SpartanSigned192 left[INSTRUCTION_INPUT_COEFFICIENTS];
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        right[descriptor] = spartan_s192_zero();
        left[descriptor] = spartan_s192_zero();
    }

    instruction_input_accumulate_u64(
        left,
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_LEFT_IS_RS1),
        instruction_input_flag(row_1, INSTRUCTION_INPUT_FLAG_LEFT_IS_RS1),
        row_0.words[9],
        row_1.words[9]);
    instruction_input_accumulate_u64(
        left,
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_LEFT_IS_PC),
        instruction_input_flag(row_1, INSTRUCTION_INPUT_FLAG_LEFT_IS_PC),
        row_0.words[6],
        row_1.words[6]);
    instruction_input_accumulate_u64(
        right,
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_RIGHT_IS_RS2),
        instruction_input_flag(row_1, INSTRUCTION_INPUT_FLAG_RIGHT_IS_RS2),
        instruction_input_rs2(row_0),
        instruction_input_rs2(row_1));
    instruction_input_accumulate_i128(
        right,
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_RIGHT_IS_IMM),
        instruction_input_flag(row_1, INSTRUCTION_INPUT_FLAG_RIGHT_IS_IMM),
        row_0.words[7],
        row_0.words[8],
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_IMM_POSITIVE) != 0,
        row_1.words[7],
        row_1.words[8],
        instruction_input_flag(row_1, INSTRUCTION_INPUT_FLAG_IMM_POSITIVE) != 0);

    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        SolinasFp128 q = solinas_add(
            spartan_small_times_s192(1, right[descriptor]),
            solinas_mul_wide(gamma, spartan_small_times_s192(1, left[descriptor])));
        lanes[descriptor] = solinas_add(
            lanes[descriptor],
            solinas_mul_wide(weight, q));
    }
}

inline SolinasFp128 instruction_input_row_field(
    device const SpartanOuterUniskipRow& row,
    uint table)
{
    switch (table) {
        case 0u:
            return instruction_input_from_u64(
                (ulong)instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_LEFT_IS_RS1));
        case 1u:
            return instruction_input_from_u64(row.words[9]);
        case 2u:
            return instruction_input_from_u64(
                (ulong)instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_LEFT_IS_PC));
        case 3u:
            return instruction_input_from_u64(row.words[6]);
        case 4u:
            return instruction_input_from_u64(
                (ulong)instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_RIGHT_IS_RS2));
        case 5u:
            return instruction_input_from_u64(instruction_input_rs2(row));
        case 6u:
            return instruction_input_from_u64(
                (ulong)instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_RIGHT_IS_IMM));
        default:
            return instruction_input_from_i128(
                row.words[7],
                row.words[8],
                instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_IMM_POSITIVE) != 0);
    }
}

inline SolinasFp128 instruction_input_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(low, solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline SolinasFp128 instruction_input_relation(
    thread const SolinasFp128* values,
    SolinasFp128 gamma)
{
    SolinasFp128 right = solinas_add(
        solinas_mul_wide(values[4], values[5]),
        solinas_mul_wide(values[6], values[7]));
    SolinasFp128 left = solinas_add(
        solinas_mul_wide(values[0], values[1]),
        solinas_mul_wide(values[2], values[3]));
    return solinas_add(right, solinas_mul_wide(gamma, left));
}

inline SolinasFp128 instruction_input_relation_lead(
    thread const SolinasFp128* at_0,
    thread const SolinasFp128* at_1,
    SolinasFp128 gamma)
{
    SolinasFp128 right = solinas_add(
        solinas_mul_wide(
            solinas_sub(at_1[4], at_0[4]),
            solinas_sub(at_1[5], at_0[5])),
        solinas_mul_wide(
            solinas_sub(at_1[6], at_0[6]),
            solinas_sub(at_1[7], at_0[7])));
    SolinasFp128 left = solinas_add(
        solinas_mul_wide(
            solinas_sub(at_1[0], at_0[0]),
            solinas_sub(at_1[1], at_0[1])),
        solinas_mul_wide(
            solinas_sub(at_1[2], at_0[2]),
            solinas_sub(at_1[3], at_0[3])));
    return solinas_add(right, solinas_mul_wide(gamma, left));
}

inline void instruction_input_accumulate_bound_pair(
    thread const SolinasFp128* at_0,
    thread const SolinasFp128* at_1,
    SolinasFp128 gamma,
    SolinasFp128 weight,
    thread SolinasFp128* lanes)
{
    SolinasFp128 coefficients[INSTRUCTION_INPUT_COEFFICIENTS] = {
        instruction_input_relation(at_0, gamma),
        instruction_input_relation(at_1, gamma),
        instruction_input_relation_lead(at_0, at_1, gamma),
    };
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        lanes[descriptor] = solinas_add(
            lanes[descriptor],
            solinas_mul_wide(weight, coefficients[descriptor]));
    }
}

inline SolinasFp128 instruction_input_simd_sum(SolinasFp128 value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb = simd_shuffle_down(value.limb, offset);
        value = solinas_add(value, other);
    }
    return value;
}

inline void instruction_input_finish_block(
    thread SolinasFp128* lanes,
    SolinasFp128 outer_weight,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        SolinasFp128 sum = instruction_input_simd_sum(lanes[descriptor]);
        if (lane_in_simd == 0) {
            shared[descriptor * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[descriptor * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = instruction_input_simd_sum(sum);
            if (lane_in_simd == 0) {
                partials[descriptor * e_out_length + x_out] =
                    solinas_mul_wide(outer_weight, sum);
            }
        }
    }
}

kernel void solinas_instruction_input_native_message(
    device const SpartanOuterUniskipRow* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant SolinasFp128& gamma [[buffer(4)]],
    constant InstructionInputParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_INPUT_COEFFICIENTS];
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        lanes[descriptor] = solinas_zero();
    }
    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        instruction_input_native_pair(
            rows[2u * pair],
            rows[2u * pair + 1u],
            gamma,
            e_in[x_in],
            lanes);
    }
    instruction_input_finish_block(
        lanes, e_out[x_out], partials, shared, x_out, params.e_out_length,
        lane_in_simd, simdgroup, threads / 32u);
}

kernel void solinas_instruction_input_native_transition(
    device const SpartanOuterUniskipRow* rows [[buffer(0)]],
    device SolinasFp128* dense [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant SolinasFp128& gamma [[buffer(6)]],
    constant InstructionInputParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_INPUT_COEFFICIENTS];
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        lanes[descriptor] = solinas_zero();
    }
    uint bound_elements = params.source_elements / 2u;
    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint source = 4u * pair;
        SolinasFp128 at_0[INSTRUCTION_INPUT_TABLES];
        SolinasFp128 at_1[INSTRUCTION_INPUT_TABLES];
        for (uint table = 0; table < INSTRUCTION_INPUT_TABLES; table++) {
            at_0[table] = instruction_input_bind(
                instruction_input_row_field(rows[source], table),
                instruction_input_row_field(rows[source + 1u], table),
                challenge);
            at_1[table] = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], table),
                instruction_input_row_field(rows[source + 3u], table),
                challenge);
            uint destination = table * bound_elements + 2u * pair;
            dense[destination] = at_0[table];
            dense[destination + 1u] = at_1[table];
        }
        instruction_input_accumulate_bound_pair(
            at_0, at_1, gamma, e_in[x_in], lanes);
    }
    instruction_input_finish_block(
        lanes, e_out[x_out], partials, shared, x_out, params.e_out_length,
        lane_in_simd, simdgroup, threads / 32u);
}

kernel void solinas_instruction_input_dense_transition(
    device const SolinasFp128* tables [[buffer(0)]],
    device SolinasFp128* bound [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant SolinasFp128& gamma [[buffer(6)]],
    constant InstructionInputParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_INPUT_COEFFICIENTS];
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        lanes[descriptor] = solinas_zero();
    }
    uint bound_elements = params.source_elements / 2u;
    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        SolinasFp128 at_0[INSTRUCTION_INPUT_TABLES];
        SolinasFp128 at_1[INSTRUCTION_INPUT_TABLES];
        for (uint table = 0; table < INSTRUCTION_INPUT_TABLES; table++) {
            uint source = table * params.source_elements + 4u * pair;
            at_0[table] = instruction_input_bind(
                tables[source], tables[source + 1u], challenge);
            at_1[table] = instruction_input_bind(
                tables[source + 2u], tables[source + 3u], challenge);
            uint destination = table * bound_elements + 2u * pair;
            bound[destination] = at_0[table];
            bound[destination + 1u] = at_1[table];
        }
        instruction_input_accumulate_bound_pair(
            at_0, at_1, gamma, e_in[x_in], lanes);
    }
    instruction_input_finish_block(
        lanes, e_out[x_out], partials, shared, x_out, params.e_out_length,
        lane_in_simd, simdgroup, threads / 32u);
}

kernel void solinas_instruction_input_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant InstructionInputReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]])
{
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        SolinasFp128 value = gid < params.input_count
            ? input[descriptor * params.input_count + gid]
            : solinas_zero();
        value = instruction_input_simd_sum(value);
        if (lane_in_simd == 0) {
            output[descriptor * params.output_count + gid / 32u] = value;
        }
    }
}
