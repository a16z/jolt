// This fragment must follow the production InstructionInput fragment. It uses
// its resident row ABI, field conversion, bind, and three-lane reduction.

#define INSTRUCTION_INPUT_SUCCESSOR_FLAG_IMM_POSITIVE 18u
#define INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_RS1 20u
#define INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_PC 21u
#define INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_RS2 22u
#define INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_IMM 23u

struct InstructionInputSuccessorMaterializeParams {
    uint source_elements;
    uint bound_elements;
    uint2 reserved;
};

struct InstructionInputSuccessorDenseMessageParams {
    uint table_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

struct InstructionInputSuccessorQuadratic {
    SolinasFp128 at_0;
    SolinasFp128 at_1;
    SolinasFp128 quadratic;
};

inline uint instruction_input_successor_flag(ulong flags, uint bit) {
    return (uint)((flags >> bit) & 1ul);
}

inline SolinasFp128 instruction_input_successor_boolean_bind(
    uint low,
    uint high,
    SolinasFp128 challenge)
{
    SolinasFp128 zero = solinas_zero();
    SolinasFp128 one = instruction_input_from_u64(1ul);
    SolinasFp128 constant_value = solinas_select(low != 0u, one, zero);
    SolinasFp128 different = solinas_select(
        low != 0u,
        solinas_sub(one, challenge),
        challenge);
    return solinas_select(low == high, constant_value, different);
}

inline InstructionInputSuccessorQuadratic
instruction_input_successor_factor_pair(
    SolinasFp128 a_at_0,
    SolinasFp128 a_at_1,
    SolinasFp128 b_at_0,
    SolinasFp128 b_at_1)
{
    InstructionInputSuccessorQuadratic result;
    result.at_0 = solinas_mul_wide(a_at_0, b_at_0);
    result.at_1 = solinas_mul_wide(a_at_1, b_at_1);
    result.quadratic = solinas_mul_wide(
        solinas_sub(a_at_1, a_at_0),
        solinas_sub(b_at_1, b_at_0));
    return result;
}

inline InstructionInputSuccessorQuadratic
instruction_input_successor_add_quadratic(
    InstructionInputSuccessorQuadratic lhs,
    InstructionInputSuccessorQuadratic rhs)
{
    InstructionInputSuccessorQuadratic result;
    result.at_0 = solinas_add(lhs.at_0, rhs.at_0);
    result.at_1 = solinas_add(lhs.at_1, rhs.at_1);
    result.quadratic = solinas_add(lhs.quadratic, rhs.quadratic);
    return result;
}

inline InstructionInputSuccessorQuadratic
instruction_input_successor_table_factor(
    device const SolinasFp128* tables,
    uint table_elements,
    uint source,
    uint flag_table,
    uint value_table)
{
    uint flag_base = flag_table * table_elements + source;
    uint value_base = value_table * table_elements + source;
    return instruction_input_successor_factor_pair(
        tables[flag_base],
        tables[flag_base + 1u],
        tables[value_base],
        tables[value_base + 1u]);
}

kernel void solinas_instruction_input_successor_materialize(
    device const InstructionInputRow* rows [[buffer(0)]],
    device SolinasFp128* dense [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant InstructionInputSuccessorMaterializeParams& params [[buffer(3)]],
    uint y [[thread_position_in_grid]])
{
    if (y >= params.bound_elements) {
        return;
    }

    device const InstructionInputRow& low = rows[2u * y];
    device const InstructionInputRow& high = rows[2u * y + 1u];
    ulong low_flags = instruction_input_row_word(low, 5u);
    ulong high_flags = instruction_input_row_word(high, 5u);

    dense[y] = instruction_input_successor_boolean_bind(
        instruction_input_successor_flag(
            low_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_RS1),
        instruction_input_successor_flag(
            high_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_RS1),
        challenge);
    dense[params.bound_elements + y] = instruction_input_bind(
        instruction_input_from_u64(instruction_input_row_word(low, 0u)),
        instruction_input_from_u64(instruction_input_row_word(high, 0u)),
        challenge);
    dense[2u * params.bound_elements + y] =
        instruction_input_successor_boolean_bind(
            instruction_input_successor_flag(
                low_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_PC),
            instruction_input_successor_flag(
                high_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_PC),
            challenge);
    dense[3u * params.bound_elements + y] = instruction_input_bind(
        instruction_input_from_u64(instruction_input_row_word(low, 1u)),
        instruction_input_from_u64(instruction_input_row_word(high, 1u)),
        challenge);
    dense[4u * params.bound_elements + y] =
        instruction_input_successor_boolean_bind(
            instruction_input_successor_flag(
                low_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_RS2),
            instruction_input_successor_flag(
                high_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_RS2),
            challenge);
    dense[5u * params.bound_elements + y] = instruction_input_bind(
        instruction_input_from_u64(instruction_input_row_word(low, 2u)),
        instruction_input_from_u64(instruction_input_row_word(high, 2u)),
        challenge);
    dense[6u * params.bound_elements + y] =
        instruction_input_successor_boolean_bind(
            instruction_input_successor_flag(
                low_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_IMM),
            instruction_input_successor_flag(
                high_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_IMM),
            challenge);
    dense[7u * params.bound_elements + y] = instruction_input_bind(
        instruction_input_from_i128(
            instruction_input_row_word(low, 3u),
            instruction_input_row_word(low, 4u),
            instruction_input_successor_flag(
                low_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_IMM_POSITIVE) != 0u),
        instruction_input_from_i128(
            instruction_input_row_word(high, 3u),
            instruction_input_row_word(high, 4u),
            instruction_input_successor_flag(
                high_flags, INSTRUCTION_INPUT_SUCCESSOR_FLAG_IMM_POSITIVE) != 0u),
        challenge);
}

kernel void solinas_instruction_input_successor_dense_message(
    device const SolinasFp128* tables [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant SolinasFp128& gamma [[buffer(4)]],
    constant InstructionInputSuccessorDenseMessageParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[INSTRUCTION_INPUT_COEFFICIENTS];
    lanes[0] = solinas_zero();
    lanes[1] = solinas_zero();
    lanes[2] = solinas_zero();

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint source = 2u * pair;
        InstructionInputSuccessorQuadratic left =
            instruction_input_successor_add_quadratic(
                instruction_input_successor_table_factor(
                    tables, params.table_elements, source, 0u, 1u),
                instruction_input_successor_table_factor(
                    tables, params.table_elements, source, 2u, 3u));
        InstructionInputSuccessorQuadratic right =
            instruction_input_successor_add_quadratic(
                instruction_input_successor_table_factor(
                    tables, params.table_elements, source, 4u, 5u),
                instruction_input_successor_table_factor(
                    tables, params.table_elements, source, 6u, 7u));
        SolinasFp128 weight = e_in[x_in];
        lanes[0] = solinas_add(
            lanes[0],
            solinas_mul_wide(
                weight,
                solinas_add(right.at_0, solinas_mul_wide(gamma, left.at_0))));
        lanes[1] = solinas_add(
            lanes[1],
            solinas_mul_wide(
                weight,
                solinas_add(right.at_1, solinas_mul_wide(gamma, left.at_1))));
        lanes[2] = solinas_add(
            lanes[2],
            solinas_mul_wide(
                weight,
                solinas_add(
                    right.quadratic,
                    solinas_mul_wide(gamma, left.quadratic))));
    }

    instruction_input_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

#undef INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_IMM
#undef INSTRUCTION_INPUT_SUCCESSOR_FLAG_RIGHT_IS_RS2
#undef INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_PC
#undef INSTRUCTION_INPUT_SUCCESSOR_FLAG_LEFT_IS_RS1
#undef INSTRUCTION_INPUT_SUCCESSOR_FLAG_IMM_POSITIVE
