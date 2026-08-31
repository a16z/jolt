// Concatenate after fp128.metal and simd_reduce.metal.

#define PRODUCT_REMAINDER_MESSAGE_COLUMNS 2u
#define PRODUCT_REMAINDER_OPENINGS 8u

#define PRODUCT_REMAINDER_FLAG_RIGHT_NONNEGATIVE 0u
#define PRODUCT_REMAINDER_FLAG_JUMP 1u
#define PRODUCT_REMAINDER_FLAG_WRITE_LOOKUP 2u
#define PRODUCT_REMAINDER_FLAG_BRANCH 3u
#define PRODUCT_REMAINDER_FLAG_NEXT_IS_NOOP 4u
#define PRODUCT_REMAINDER_FLAG_VIRTUAL 5u

#define SPARTAN_PRODUCT_FLAG_JUMP 5u
#define SPARTAN_PRODUCT_FLAG_VIRTUAL 9u
#define SPARTAN_PRODUCT_FLAG_WRITE_LOOKUP 14u
#define SPARTAN_PRODUCT_FLAG_RIGHT_NONNEGATIVE 17u
#define SPARTAN_PRODUCT_FLAG_BRANCH 25u
#define SPARTAN_PRODUCT_FLAG_NEXT_IS_NOOP 26u

struct ProductRemainderRow {
    ulong words[5];
};

struct ProductRemainderPhaseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

struct ProductRemainderOpeningParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

struct ProductRemainderReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

struct ProductRemainderBindRangeParams {
    uint source_offset;
    uint destination_offset;
    uint output_start;
    uint output_count;
};

inline bool product_remainder_flag(ulong flags, uint bit) {
    return ((flags >> bit) & 1ul) != 0ul;
}

inline SolinasFp128 product_remainder_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 product_remainder_from_signed_u128(
    ulong low,
    ulong high,
    bool nonnegative)
{
    SolinasFp128 magnitude;
    magnitude.limb = uint4(
        (uint)low,
        (uint)(low >> 32),
        (uint)high,
        (uint)(high >> 32));
    return nonnegative
        ? magnitude
        : solinas_sub(solinas_zero(), magnitude);
}

inline SolinasFp128 product_remainder_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline void product_remainder_relation_values(
    device const ProductRemainderRow& row,
    device const SolinasFp128* lagrange,
    thread SolinasFp128& left,
    thread SolinasFp128& right)
{
    ulong flags = row.words[4];
    left = solinas_add(
        solinas_mul_wide(
            lagrange[0], product_remainder_from_u64(row.words[0])),
        solinas_mul_wide(
            lagrange[1], product_remainder_from_u64(row.words[3])));
    if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_JUMP)) {
        left = solinas_add(left, lagrange[2]);
    }

    right = solinas_mul_wide(
        lagrange[0],
        product_remainder_from_signed_u128(
            row.words[1],
            row.words[2],
            product_remainder_flag(
                flags, PRODUCT_REMAINDER_FLAG_RIGHT_NONNEGATIVE)));
    if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_BRANCH)) {
        right = solinas_add(right, lagrange[1]);
    }
    if (!product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_NEXT_IS_NOOP)) {
        right = solinas_add(right, lagrange[2]);
    }
}

inline void product_remainder_stage1_relation_values(
    device const InstructionInputRow& compact,
    device const SpartanOuterSuccessorRow& residual,
    device const SolinasFp128* lagrange,
    thread SolinasFp128& left,
    thread SolinasFp128& right)
{
    ulong flags = instruction_input_row_word(compact, 5u);
    left = solinas_add(
        solinas_mul_wide(
            lagrange[0],
            product_remainder_from_u64(spartan_outer_successor_word(residual, 0u))),
        solinas_mul_wide(
            lagrange[1],
            product_remainder_from_u64(spartan_outer_successor_word(residual, 13u))));
    if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_JUMP)) {
        left = solinas_add(left, lagrange[2]);
    }

    right = solinas_mul_wide(
        lagrange[0],
        product_remainder_from_signed_u128(
            spartan_outer_successor_word(residual, 1u),
            spartan_outer_successor_word(residual, 2u),
            product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_RIGHT_NONNEGATIVE)));
    if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_BRANCH)) {
        right = solinas_add(right, lagrange[1]);
    }
    if (!product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_NEXT_IS_NOOP)) {
        right = solinas_add(right, lagrange[2]);
    }
}

inline void product_remainder_finish_block(
    thread SolinasFp128* lanes,
    uint columns,
    SolinasFp128 outer_weight,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane,
    uint simdgroup,
    uint simdgroups)
{
    for (uint column = 0u; column < columns; column++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[column]);
        if (lane == 0u) {
            shared[column * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u && lane < columns) {
        SolinasFp128 sum = solinas_zero();
        for (uint group = 0u; group < simdgroups; group++) {
            sum = solinas_add(sum, shared[lane * simdgroups + group]);
        }
        partials[lane * e_out_length + x_out] =
            solinas_mul_wide(outer_weight, sum);
    }
}

kernel void solinas_product_remainder_materialize_message(
    device const ProductRemainderRow* rows [[buffer(0)]],
    device const SolinasFp128* lagrange [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* state [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant ProductRemainderPhaseParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_REMAINDER_MESSAGE_COLUMNS];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint low_index = 2u * pair;
        uint high_index = low_index + 1u;
        SolinasFp128 left_low;
        SolinasFp128 right_low;
        SolinasFp128 left_high;
        SolinasFp128 right_high;
        product_remainder_relation_values(
            rows[low_index], lagrange, left_low, right_low);
        product_remainder_relation_values(
            rows[high_index], lagrange, left_high, right_high);

        state[low_index] = left_low;
        state[high_index] = left_high;
        state[params.source_elements + low_index] = right_low;
        state[params.source_elements + high_index] = right_high;

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
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_remainder_materialize_stage1_message(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* lagrange [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* state [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant ProductRemainderPhaseParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_REMAINDER_MESSAGE_COLUMNS];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();

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

        state[low_index] = left_low;
        state[high_index] = left_high;
        state[params.source_elements + low_index] = right_low;
        state[params.source_elements + high_index] = right_high;

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
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_remainder_bind_and_message(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant ProductRemainderPhaseParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_REMAINDER_MESSAGE_COLUMNS];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();
    uint bound_elements = params.source_elements / 2u;

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint source_index = 4u * pair;
        uint destination_index = 2u * pair;
        SolinasFp128 left_0 = product_remainder_bind(
            source[source_index], source[source_index + 1u], challenge);
        SolinasFp128 left_1 = product_remainder_bind(
            source[source_index + 2u], source[source_index + 3u], challenge);
        uint right_source = params.source_elements + source_index;
        SolinasFp128 right_0 = product_remainder_bind(
            source[right_source], source[right_source + 1u], challenge);
        SolinasFp128 right_1 = product_remainder_bind(
            source[right_source + 2u], source[right_source + 3u], challenge);

        destination[destination_index] = left_0;
        destination[destination_index + 1u] = left_1;
        destination[bound_elements + destination_index] = right_0;
        destination[bound_elements + destination_index + 1u] = right_1;

        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight, solinas_mul_wide(left_0, right_0)));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                solinas_mul_wide(
                    solinas_sub(left_1, left_0),
                    solinas_sub(right_1, right_0))));
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_remainder_bind_range(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant ProductRemainderBindRangeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.output_count) {
        return;
    }
    uint output_index = params.output_start + gid;
    uint source_index = params.source_offset + 2u * output_index;
    destination[params.destination_offset + output_index] = product_remainder_bind(
        source[source_index], source[source_index + 1u], challenge);
}

kernel void solinas_product_remainder_copy_prefix(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        destination[gid] = source[gid];
    }
}

kernel void solinas_product_remainder_bound_message(
    device const SolinasFp128* state [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant ProductRemainderPhaseParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_REMAINDER_MESSAGE_COLUMNS];
    sums[0] = solinas_zero();
    sums[1] = solinas_zero();
    uint bound_elements = params.source_elements / 2u;

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        uint index = 2u * pair;
        SolinasFp128 left_0 = state[index];
        SolinasFp128 left_1 = state[index + 1u];
        SolinasFp128 right_0 = state[bound_elements + index];
        SolinasFp128 right_1 = state[bound_elements + index + 1u];
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(weight, solinas_mul_wide(left_0, right_0)));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                solinas_mul_wide(
                    solinas_sub(left_1, left_0),
                    solinas_sub(right_1, right_0))));
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_remainder_openings(
    device const ProductRemainderRow* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant ProductRemainderOpeningParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_REMAINDER_OPENINGS];
    for (uint column = 0u; column < PRODUCT_REMAINDER_OPENINGS; column++) {
        sums[column] = solinas_zero();
    }

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        ulong left_input = rows[row_index].words[0];
        ulong right_low = rows[row_index].words[1];
        ulong right_high = rows[row_index].words[2];
        ulong lookup_output = rows[row_index].words[3];
        ulong flags = rows[row_index].words[4];
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight, product_remainder_from_u64(left_input)));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                product_remainder_from_signed_u128(
                    right_low,
                    right_high,
                    product_remainder_flag(
                        flags, PRODUCT_REMAINDER_FLAG_RIGHT_NONNEGATIVE))));
        if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_JUMP)) {
            sums[2] = solinas_add(sums[2], weight);
        }
        if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_WRITE_LOOKUP)) {
            sums[3] = solinas_add(sums[3], weight);
        }
        sums[4] = solinas_add(
            sums[4],
            solinas_mul_wide(
                weight, product_remainder_from_u64(lookup_output)));
        if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_BRANCH)) {
            sums[5] = solinas_add(sums[5], weight);
        }
        if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_NEXT_IS_NOOP)) {
            sums[6] = solinas_add(sums[6], weight);
        }
        if (product_remainder_flag(flags, PRODUCT_REMAINDER_FLAG_VIRTUAL)) {
            sums[7] = solinas_add(sums[7], weight);
        }
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_OPENINGS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_remainder_stage1_openings(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant ProductRemainderOpeningParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 sums[PRODUCT_REMAINDER_OPENINGS];
    for (uint column = 0u; column < PRODUCT_REMAINDER_OPENINGS; column++) {
        sums[column] = solinas_zero();
    }

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        device const InstructionInputRow& compact = compact_rows[row_index];
        device const SpartanOuterSuccessorRow& residual = residual_rows[row_index];
        ulong flags = instruction_input_row_word(compact, 5u);
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight,
                product_remainder_from_u64(spartan_outer_successor_word(residual, 0u))));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                product_remainder_from_signed_u128(
                    spartan_outer_successor_word(residual, 1u),
                    spartan_outer_successor_word(residual, 2u),
                    product_remainder_flag(
                        flags,
                        SPARTAN_PRODUCT_FLAG_RIGHT_NONNEGATIVE))));
        if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_JUMP)) {
            sums[2] = solinas_add(sums[2], weight);
        }
        if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_WRITE_LOOKUP)) {
            sums[3] = solinas_add(sums[3], weight);
        }
        sums[4] = solinas_add(
            sums[4],
            solinas_mul_wide(
                weight,
                product_remainder_from_u64(spartan_outer_successor_word(residual, 13u))));
        if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_BRANCH)) {
            sums[5] = solinas_add(sums[5], weight);
        }
        if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_NEXT_IS_NOOP)) {
            sums[6] = solinas_add(sums[6], weight);
        }
        if (product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_VIRTUAL)) {
            sums[7] = solinas_add(sums[7], weight);
        }
    }

    product_remainder_finish_block(
        sums,
        PRODUCT_REMAINDER_OPENINGS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_product_remainder_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant ProductRemainderReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint column = 0u; column < params.columns; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[column * params.output_count + gid / 32u] = value;
        }
    }
}
