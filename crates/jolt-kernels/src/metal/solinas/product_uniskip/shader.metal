// Concatenate after fp128.metal, simd_reduce.metal, and product_remainder/shader.metal.

#define PRODUCT_UNISKIP_EXTENDED_NODES 2u

struct ProductUniskipBlockParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

inline SolinasFp128 product_uniskip_triple(SolinasFp128 value) {
    return solinas_add(value, solinas_add(value, value));
}

inline void product_uniskip_accumulate(
    SolinasFp128 left_input,
    SolinasFp128 lookup_output,
    SolinasFp128 right_input,
    bool jump,
    bool branch,
    bool next_is_noop,
    SolinasFp128 weight,
    thread SolinasFp128* sums)
{
    SolinasFp128 one = product_remainder_from_u64(1ul);
    SolinasFp128 three = product_uniskip_triple(one);
    SolinasFp128 three_lookup = product_uniskip_triple(lookup_output);

    SolinasFp128 left_minus_two = solinas_sub(
        product_uniskip_triple(left_input), three_lookup);
    if (jump) {
        left_minus_two = solinas_add(left_minus_two, one);
    }
    SolinasFp128 right_minus_two = product_uniskip_triple(right_input);
    if (branch) {
        right_minus_two = solinas_sub(right_minus_two, three);
    }
    if (!next_is_noop) {
        right_minus_two = solinas_add(right_minus_two, one);
    }

    SolinasFp128 left_plus_two = solinas_sub(left_input, three_lookup);
    if (jump) {
        left_plus_two = solinas_add(left_plus_two, three);
    }
    SolinasFp128 right_plus_two = right_input;
    if (branch) {
        right_plus_two = solinas_sub(right_plus_two, three);
    }
    if (!next_is_noop) {
        right_plus_two = solinas_add(right_plus_two, three);
    }

    sums[0] = solinas_add(
        sums[0],
        solinas_mul_wide(
            weight, solinas_mul_wide(left_minus_two, right_minus_two)));
    sums[1] = solinas_add(
        sums[1],
        solinas_mul_wide(
            weight, solinas_mul_wide(left_plus_two, right_plus_two)));
}

kernel void solinas_product_uniskip_extended_blocks2(
    device const ProductRemainderRow* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant ProductUniskipBlockParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_UNISKIP_EXTENDED_NODES];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        device const ProductRemainderRow& row = rows[row_index];
        ulong flags = row.words[4];
        product_uniskip_accumulate(
            product_remainder_from_u64(row.words[0]),
            product_remainder_from_u64(row.words[3]),
            product_remainder_from_signed_u128(
                row.words[1],
                row.words[2],
                product_remainder_flag(
                    flags, PRODUCT_REMAINDER_FLAG_RIGHT_NONNEGATIVE)),
            product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_JUMP),
            product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_BRANCH),
            product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_NEXT_IS_NOOP),
            e_in[x_in],
            sums);
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_UNISKIP_EXTENDED_NODES,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_uniskip_stage1_extended_blocks2(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant ProductUniskipBlockParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_UNISKIP_EXTENDED_NODES];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        device const InstructionInputRow& compact = compact_rows[row_index];
        device const SpartanOuterSuccessorRow& residual = residual_rows[row_index];
        ulong flags = instruction_input_row_word(compact, 5u);
        product_uniskip_accumulate(
            product_remainder_from_u64(spartan_outer_successor_word(residual, 0u)),
            product_remainder_from_u64(spartan_outer_successor_word(residual, 13u)),
            product_remainder_from_signed_u128(
                spartan_outer_successor_word(residual, 1u),
                spartan_outer_successor_word(residual, 2u),
                product_remainder_flag(
                    flags,
                    SPARTAN_PRODUCT_FLAG_RIGHT_NONNEGATIVE)),
            product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_JUMP),
            product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_BRANCH),
            product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_NEXT_IS_NOOP),
            e_in[x_in],
            sums);
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_UNISKIP_EXTENDED_NODES,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}
