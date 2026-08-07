// Unregistered sketch. Concatenate after the instruction-claim and product helpers.

#define SUCCESSOR_MESSAGE_COLUMNS 4u

struct InstructionLookupCompanion {
    ulong left_lookup_operand;
    ulong right_lookup_low;
    ulong right_lookup_high;
};

struct ProductInstructionPhaseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

inline SolinasFp128 successor_instruction_combined(
    device const ProductRemainderRow& product,
    device const InstructionLookupCompanion& lookup,
    constant const SolinasFp128* gamma_powers)
{
    ulong flags = product.words[4];
    SolinasFp128 value = product_remainder_from_u64(product.words[3]);
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[0],
            product_remainder_from_u64(lookup.left_lookup_operand)));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[1],
            instruction_claim_from_u128(
                lookup.right_lookup_low, lookup.right_lookup_high)));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[2],
            product_remainder_from_u64(product.words[0])));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[3],
            product_remainder_from_signed_u128(
                product.words[1],
                product.words[2],
                product_remainder_flag(
                    flags, PRODUCT_REMAINDER_FLAG_RIGHT_NONNEGATIVE))));
    return value;
}

kernel void solinas_product_instruction_materialize_message(
    device const ProductRemainderRow* product_rows [[buffer(0)]],
    device const InstructionLookupCompanion* lookup_rows [[buffer(1)]],
    device const SolinasFp128* lagrange [[buffer(2)]],
    constant const SolinasFp128* gamma_powers [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* product_state [[buffer(6)]],
    device SolinasFp128* instruction_state [[buffer(7)]],
    device SolinasFp128* partials [[buffer(8)]],
    constant ProductInstructionPhaseParams& params [[buffer(9)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
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
        product_remainder_relation_values(
            product_rows[low_index], lagrange, left_low, right_low);
        product_remainder_relation_values(
            product_rows[high_index], lagrange, left_high, right_high);
        SolinasFp128 instruction_low = successor_instruction_combined(
            product_rows[low_index], lookup_rows[low_index], gamma_powers);
        SolinasFp128 instruction_high = successor_instruction_combined(
            product_rows[high_index], lookup_rows[high_index], gamma_powers);

        product_state[low_index] = left_low;
        product_state[high_index] = left_high;
        product_state[params.source_elements + low_index] = right_low;
        product_state[params.source_elements + high_index] = right_high;
        instruction_state[low_index] = instruction_low;
        instruction_state[high_index] = instruction_high;

        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight, solinas_mul_wide(left_low, right_low)));
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
        SUCCESSOR_MESSAGE_COLUMNS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}
