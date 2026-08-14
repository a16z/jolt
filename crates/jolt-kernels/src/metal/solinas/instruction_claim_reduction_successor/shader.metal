#define SUCCESSOR_MESSAGE_COLUMNS 4u

struct ProductInstructionPhaseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

inline SolinasFp128 stage1_instruction_combined(
    device const InstructionInputRow& compact,
    device const SpartanOuterUniskipResidualRow& residual,
    constant const SolinasFp128* gamma_powers)
{
    ulong flags = instruction_input_row_word(compact, 5u);
    SolinasFp128 value = product_remainder_from_u64(
        spartan_outer_residual_word(residual, 13u));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[0],
            product_remainder_from_u64(
                spartan_outer_residual_word(residual, 8u))));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[1],
            instruction_claim_from_u128(
                spartan_outer_residual_word(residual, 9u),
                spartan_outer_residual_word(residual, 10u))));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[2],
            product_remainder_from_u64(
                spartan_outer_residual_word(residual, 0u))));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[3],
            product_remainder_from_signed_u128(
                spartan_outer_residual_word(residual, 1u),
                spartan_outer_residual_word(residual, 2u),
                product_remainder_flag(
                    flags,
                    SPARTAN_PRODUCT_FLAG_RIGHT_NONNEGATIVE))));
    return value;
}

kernel void solinas_instruction_claim_materialize_stage1_rows(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterUniskipResidualRow* residual_rows [[buffer(1)]],
    constant const SolinasFp128* gamma_powers [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* state [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant InstructionClaimPhaseParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length) {
        return;
    }

    SolinasFp128 sums[INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint low_index = 2u * pair;
        uint high_index = low_index + 1u;
        SolinasFp128 low = stage1_instruction_combined(
            compact_rows[low_index], residual_rows[low_index], gamma_powers);
        SolinasFp128 high = stage1_instruction_combined(
            compact_rows[high_index], residual_rows[high_index], gamma_powers);
        state[low_index] = low;
        state[high_index] = high;

        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(sums[0], solinas_mul_wide(weight, low));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                solinas_sub(solinas_add(high, high), low)));
    }

    instruction_claim_finish_block(
        sums,
        INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_instruction_claim_open_stage1_lookup_operands(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterUniskipResidualRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant InstructionClaimOpeningParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length
        || params.columns != INSTRUCTION_CLAIM_ALIASED_OPENINGS)
    {
        return;
    }

    SolinasFp128 sums[INSTRUCTION_CLAIM_ALIASED_OPENINGS];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        device const SpartanOuterUniskipResidualRow& residual = residual_rows[row_index];
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight,
                instruction_claim_from_u64(
                    spartan_outer_residual_word(residual, 8u))));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                instruction_claim_from_u128(
                    spartan_outer_residual_word(residual, 9u),
                    spartan_outer_residual_word(residual, 10u))));
    }

    instruction_claim_finish_block(
        sums,
        INSTRUCTION_CLAIM_ALIASED_OPENINGS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_instruction_materialize_stage1_message(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterUniskipResidualRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* lagrange [[buffer(2)]],
    constant const SolinasFp128* gamma_powers [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* product_state [[buffer(6)]],
    device SolinasFp128* instruction_state [[buffer(7)]],
    device SolinasFp128* product_partials [[buffer(8)]],
    device SolinasFp128* instruction_partials [[buffer(9)]],
    constant ProductInstructionPhaseParams& params [[buffer(10)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length) {
        return;
    }

    SolinasFp128 sums[SUCCESSOR_MESSAGE_COLUMNS];
    for (uint column = 0u; column < SUCCESSOR_MESSAGE_COLUMNS; column++) {
        sums[column] = solinas_zero();
    }

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint low_index = 2u * pair;
        uint high_index = low_index + 1u;
        SolinasFp128 left_low;
        SolinasFp128 right_low;
        SolinasFp128 left_high;
        SolinasFp128 right_high;
        product_remainder_stage1_relation_values(
            compact_rows[low_index],
            residual_rows[low_index],
            lagrange,
            left_low,
            right_low);
        product_remainder_stage1_relation_values(
            compact_rows[high_index],
            residual_rows[high_index],
            lagrange,
            left_high,
            right_high);
        SolinasFp128 instruction_low = stage1_instruction_combined(
            compact_rows[low_index], residual_rows[low_index], gamma_powers);
        SolinasFp128 instruction_high = stage1_instruction_combined(
            compact_rows[high_index], residual_rows[high_index], gamma_powers);

        product_state[low_index] = left_low;
        product_state[high_index] = left_high;
        product_state[params.source_elements + low_index] = right_low;
        product_state[params.source_elements + high_index] = right_high;
        instruction_state[low_index] = instruction_low;
        instruction_state[high_index] = instruction_high;

        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(weight, solinas_mul_wide(left_low, right_low)));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                solinas_mul_wide(
                    solinas_sub(left_high, left_low),
                    solinas_sub(right_high, right_low))));
        sums[2] = solinas_add(
            sums[2], solinas_mul_wide(weight, instruction_low));
        sums[3] = solinas_add(
            sums[3],
            solinas_mul_wide(
                weight,
                solinas_sub(
                    solinas_add(instruction_high, instruction_high),
                    instruction_low)));
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        e_out[x_out],
        product_partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    instruction_claim_finish_block(
        sums + PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
        e_out[x_out],
        instruction_partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}
