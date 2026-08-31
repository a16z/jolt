#define INSTRUCTION_INPUT_COEFFICIENTS 3u

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
    device const InstructionInputRow& row,
    uint bit)
{
    return (int)((instruction_input_row_word(row, 5u) >> bit) & 1ul);
}

inline ulong instruction_input_rs2(device const InstructionInputRow& row) {
    return instruction_input_row_word(row, 2u);
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
    device const InstructionInputRow& row_0,
    device const InstructionInputRow& row_1,
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
        instruction_input_row_word(row_0, 0u),
        instruction_input_row_word(row_1, 0u));
    instruction_input_accumulate_u64(
        left,
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_LEFT_IS_PC),
        instruction_input_flag(row_1, INSTRUCTION_INPUT_FLAG_LEFT_IS_PC),
        instruction_input_row_word(row_0, 1u),
        instruction_input_row_word(row_1, 1u));
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
        instruction_input_row_word(row_0, 3u),
        instruction_input_row_word(row_0, 4u),
        instruction_input_flag(row_0, INSTRUCTION_INPUT_FLAG_IMM_POSITIVE) != 0,
        instruction_input_row_word(row_1, 3u),
        instruction_input_row_word(row_1, 4u),
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
    device const InstructionInputRow& row,
    uint table)
{
    switch (table) {
        case 0u:
            return instruction_input_from_u64(
                (ulong)instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_LEFT_IS_RS1));
        case 1u:
            return instruction_input_from_u64(instruction_input_row_word(row, 0u));
        case 2u:
            return instruction_input_from_u64(
                (ulong)instruction_input_flag(row, INSTRUCTION_INPUT_FLAG_LEFT_IS_PC));
        case 3u:
            return instruction_input_from_u64(instruction_input_row_word(row, 1u));
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
                instruction_input_row_word(row, 3u),
                instruction_input_row_word(row, 4u),
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

inline void instruction_input_set_factor_pair(
    SolinasFp128 a_at_0,
    SolinasFp128 a_at_1,
    SolinasFp128 b_at_0,
    SolinasFp128 b_at_1,
    thread SolinasFp128* relation)
{
    relation[0] = solinas_mul_wide(a_at_0, b_at_0);
    relation[1] = solinas_mul_wide(a_at_1, b_at_1);
    relation[2] = solinas_mul_wide(
        solinas_sub(a_at_1, a_at_0),
        solinas_sub(b_at_1, b_at_0));
}

inline void instruction_input_add_factor_pair(
    SolinasFp128 a_at_0,
    SolinasFp128 a_at_1,
    SolinasFp128 b_at_0,
    SolinasFp128 b_at_1,
    thread SolinasFp128* relation)
{
    relation[0] = solinas_add(
        relation[0], solinas_mul_wide(a_at_0, b_at_0));
    relation[1] = solinas_add(
        relation[1], solinas_mul_wide(a_at_1, b_at_1));
    relation[2] = solinas_add(
        relation[2],
        solinas_mul_wide(
            solinas_sub(a_at_1, a_at_0),
            solinas_sub(b_at_1, b_at_0)));
}

inline void instruction_input_accumulate_relation(
    thread const SolinasFp128* left,
    thread const SolinasFp128* right,
    SolinasFp128 gamma,
    SolinasFp128 weight,
    thread SolinasFp128* lanes)
{
    for (uint descriptor = 0; descriptor < INSTRUCTION_INPUT_COEFFICIENTS; descriptor++) {
        SolinasFp128 q = solinas_add(
            right[descriptor],
            solinas_mul_wide(gamma, left[descriptor]));
        lanes[descriptor] = solinas_add(
            lanes[descriptor],
            solinas_mul_wide(weight, q));
    }
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
        SolinasFp128 sum = solinas_simd_sum_32(lanes[descriptor]);
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
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[descriptor * e_out_length + x_out] =
                    solinas_mul_wide(outer_weight, sum);
            }
        }
    }
}

kernel void solinas_instruction_input_native_message(
    device const InstructionInputRow* rows [[buffer(0)]],
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
    device const InstructionInputRow* rows [[buffer(0)]],
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
        uint destination = 2u * pair;
        SolinasFp128 left[INSTRUCTION_INPUT_COEFFICIENTS];
        SolinasFp128 right[INSTRUCTION_INPUT_COEFFICIENTS];
        {
            SolinasFp128 a_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 0u),
                instruction_input_row_field(rows[source + 1u], 0u),
                challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 0u),
                instruction_input_row_field(rows[source + 3u], 0u),
                challenge);
            dense[destination] = a_at_0;
            dense[destination + 1u] = a_at_1;
            SolinasFp128 b_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 1u),
                instruction_input_row_field(rows[source + 1u], 1u),
                challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 1u),
                instruction_input_row_field(rows[source + 3u], 1u),
                challenge);
            dense[bound_elements + destination] = b_at_0;
            dense[bound_elements + destination + 1u] = b_at_1;
            instruction_input_set_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, left);
        }
        {
            SolinasFp128 a_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 2u),
                instruction_input_row_field(rows[source + 1u], 2u),
                challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 2u),
                instruction_input_row_field(rows[source + 3u], 2u),
                challenge);
            dense[2u * bound_elements + destination] = a_at_0;
            dense[2u * bound_elements + destination + 1u] = a_at_1;
            SolinasFp128 b_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 3u),
                instruction_input_row_field(rows[source + 1u], 3u),
                challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 3u),
                instruction_input_row_field(rows[source + 3u], 3u),
                challenge);
            dense[3u * bound_elements + destination] = b_at_0;
            dense[3u * bound_elements + destination + 1u] = b_at_1;
            instruction_input_add_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, left);
        }
        {
            SolinasFp128 a_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 4u),
                instruction_input_row_field(rows[source + 1u], 4u),
                challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 4u),
                instruction_input_row_field(rows[source + 3u], 4u),
                challenge);
            dense[4u * bound_elements + destination] = a_at_0;
            dense[4u * bound_elements + destination + 1u] = a_at_1;
            SolinasFp128 b_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 5u),
                instruction_input_row_field(rows[source + 1u], 5u),
                challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 5u),
                instruction_input_row_field(rows[source + 3u], 5u),
                challenge);
            dense[5u * bound_elements + destination] = b_at_0;
            dense[5u * bound_elements + destination + 1u] = b_at_1;
            instruction_input_set_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, right);
        }
        {
            SolinasFp128 a_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 6u),
                instruction_input_row_field(rows[source + 1u], 6u),
                challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 6u),
                instruction_input_row_field(rows[source + 3u], 6u),
                challenge);
            dense[6u * bound_elements + destination] = a_at_0;
            dense[6u * bound_elements + destination + 1u] = a_at_1;
            SolinasFp128 b_at_0 = instruction_input_bind(
                instruction_input_row_field(rows[source], 7u),
                instruction_input_row_field(rows[source + 1u], 7u),
                challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                instruction_input_row_field(rows[source + 2u], 7u),
                instruction_input_row_field(rows[source + 3u], 7u),
                challenge);
            dense[7u * bound_elements + destination] = b_at_0;
            dense[7u * bound_elements + destination + 1u] = b_at_1;
            instruction_input_add_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, right);
        }
        instruction_input_accumulate_relation(
            left, right, gamma, e_in[x_in], lanes);
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
        uint source = 4u * pair;
        uint destination = 2u * pair;
        SolinasFp128 left[INSTRUCTION_INPUT_COEFFICIENTS];
        SolinasFp128 right[INSTRUCTION_INPUT_COEFFICIENTS];
        {
            SolinasFp128 a_at_0 = instruction_input_bind(
                tables[source], tables[source + 1u], challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                tables[source + 2u], tables[source + 3u], challenge);
            bound[destination] = a_at_0;
            bound[destination + 1u] = a_at_1;
            uint b_source = params.source_elements + source;
            SolinasFp128 b_at_0 = instruction_input_bind(
                tables[b_source], tables[b_source + 1u], challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                tables[b_source + 2u], tables[b_source + 3u], challenge);
            bound[bound_elements + destination] = b_at_0;
            bound[bound_elements + destination + 1u] = b_at_1;
            instruction_input_set_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, left);
        }
        {
            uint a_source = 2u * params.source_elements + source;
            SolinasFp128 a_at_0 = instruction_input_bind(
                tables[a_source], tables[a_source + 1u], challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                tables[a_source + 2u], tables[a_source + 3u], challenge);
            bound[2u * bound_elements + destination] = a_at_0;
            bound[2u * bound_elements + destination + 1u] = a_at_1;
            uint b_source = 3u * params.source_elements + source;
            SolinasFp128 b_at_0 = instruction_input_bind(
                tables[b_source], tables[b_source + 1u], challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                tables[b_source + 2u], tables[b_source + 3u], challenge);
            bound[3u * bound_elements + destination] = b_at_0;
            bound[3u * bound_elements + destination + 1u] = b_at_1;
            instruction_input_add_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, left);
        }
        {
            uint a_source = 4u * params.source_elements + source;
            SolinasFp128 a_at_0 = instruction_input_bind(
                tables[a_source], tables[a_source + 1u], challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                tables[a_source + 2u], tables[a_source + 3u], challenge);
            bound[4u * bound_elements + destination] = a_at_0;
            bound[4u * bound_elements + destination + 1u] = a_at_1;
            uint b_source = 5u * params.source_elements + source;
            SolinasFp128 b_at_0 = instruction_input_bind(
                tables[b_source], tables[b_source + 1u], challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                tables[b_source + 2u], tables[b_source + 3u], challenge);
            bound[5u * bound_elements + destination] = b_at_0;
            bound[5u * bound_elements + destination + 1u] = b_at_1;
            instruction_input_set_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, right);
        }
        {
            uint a_source = 6u * params.source_elements + source;
            SolinasFp128 a_at_0 = instruction_input_bind(
                tables[a_source], tables[a_source + 1u], challenge);
            SolinasFp128 a_at_1 = instruction_input_bind(
                tables[a_source + 2u], tables[a_source + 3u], challenge);
            bound[6u * bound_elements + destination] = a_at_0;
            bound[6u * bound_elements + destination + 1u] = a_at_1;
            uint b_source = 7u * params.source_elements + source;
            SolinasFp128 b_at_0 = instruction_input_bind(
                tables[b_source], tables[b_source + 1u], challenge);
            SolinasFp128 b_at_1 = instruction_input_bind(
                tables[b_source + 2u], tables[b_source + 3u], challenge);
            bound[7u * bound_elements + destination] = b_at_0;
            bound[7u * bound_elements + destination + 1u] = b_at_1;
            instruction_input_add_factor_pair(
                a_at_0, a_at_1, b_at_0, b_at_1, right);
        }
        instruction_input_accumulate_relation(
            left, right, gamma, e_in[x_in], lanes);
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
        value = solinas_simd_sum_32(value);
        if (lane_in_simd == 0) {
            output[descriptor * params.output_count + gid / 32u] = value;
        }
    }
}
