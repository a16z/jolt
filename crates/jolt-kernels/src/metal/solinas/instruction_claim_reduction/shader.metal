// Concatenate after offset-specialized fp128.metal and simd_reduce.metal.

#define INSTRUCTION_CLAIM_MESSAGE_COLUMNS 2u
#define INSTRUCTION_CLAIM_ALIASED_OPENINGS 2u
#define INSTRUCTION_CLAIM_CORE_OPENINGS 4u
#define INSTRUCTION_CLAIM_ALL_OPENINGS 5u

struct InstructionClaimRightLookup {
    ulong words[2];
};

struct InstructionClaimRightInput {
    ulong words[2];
};

struct InstructionClaimPhaseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

struct InstructionClaimReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

struct InstructionClaimBindRangeParams {
    uint source_offset;
    uint destination_offset;
    uint output_start;
    uint output_count;
};

struct InstructionClaimOpeningParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint columns;
};

inline SolinasFp128 instruction_claim_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb[0] = (uint)value;
    result.limb[1] = (uint)(value >> 32);
    return result;
}

inline SolinasFp128 instruction_claim_from_u128(ulong low, ulong high) {
    SolinasFp128 value;
    value.limb = uint4(
        (uint)low,
        (uint)(low >> 32),
        (uint)high,
        (uint)(high >> 32));
    SolinasCorrection corrected = solinas_add_offset(value);
    return solinas_select(corrected.carry != 0u, corrected.value, value);
}

inline SolinasFp128 instruction_claim_from_i128_twos(
    ulong low,
    ulong high)
{
    if ((high >> 63) == 0ul) {
        return instruction_claim_from_u128(low, high);
    }

    ulong magnitude_low = ~low + 1ul;
    ulong magnitude_high = ~high + (magnitude_low == 0ul ? 1ul : 0ul);
    SolinasFp128 magnitude = instruction_claim_from_u128(
        magnitude_low, magnitude_high);
    return solinas_sub(solinas_zero(), magnitude);
}

inline SolinasFp128 instruction_claim_bind(
    SolinasFp128 low,
    SolinasFp128 high,
    SolinasFp128 challenge)
{
    return solinas_add(
        low,
        solinas_mul_wide(challenge, solinas_sub(high, low)));
}

inline SolinasFp128 instruction_claim_combined(
    ulong lookup_output,
    ulong left_lookup_operand,
    device const InstructionClaimRightLookup& right_lookup_operand,
    ulong left_instruction_input,
    device const InstructionClaimRightInput& right_input,
    constant const SolinasFp128* gamma_powers)
{
    SolinasFp128 value = instruction_claim_from_u64(lookup_output);
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[0],
            instruction_claim_from_u64(left_lookup_operand)));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[1],
            instruction_claim_from_u128(
                right_lookup_operand.words[0], right_lookup_operand.words[1])));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[2],
            instruction_claim_from_u64(left_instruction_input)));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[3],
            instruction_claim_from_i128_twos(
                right_input.words[0], right_input.words[1])));
    return value;
}

inline void instruction_claim_accumulate_core_openings(
    ulong lookup_output,
    ulong left_lookup_operand,
    device const InstructionClaimRightLookup& right_lookup_operand,
    ulong left_instruction_input,
    SolinasFp128 weight,
    thread SolinasFp128* sums)
{
    sums[0] = solinas_add(
        sums[0],
        solinas_mul_wide(
            weight, instruction_claim_from_u64(lookup_output)));
    sums[1] = solinas_add(
        sums[1],
        solinas_mul_wide(
            weight, instruction_claim_from_u64(left_lookup_operand)));
    sums[2] = solinas_add(
        sums[2],
        solinas_mul_wide(
            weight,
            instruction_claim_from_u128(
                right_lookup_operand.words[0], right_lookup_operand.words[1])));
    sums[3] = solinas_add(
        sums[3],
        solinas_mul_wide(
            weight, instruction_claim_from_u64(left_instruction_input)));
}

inline void instruction_claim_finish_block(
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

kernel void solinas_instruction_claim_materialize_message(
    device const ulong* lookup_output [[buffer(0)]],
    device const ulong* left_lookup_operand [[buffer(1)]],
    device const InstructionClaimRightLookup* right_lookup_operand [[buffer(2)]],
    device const ulong* left_instruction_input [[buffer(3)]],
    device const InstructionClaimRightInput* right_instruction_input [[buffer(4)]],
    constant const SolinasFp128* gamma_powers [[buffer(5)]],
    device const SolinasFp128* e_in [[buffer(6)]],
    device const SolinasFp128* e_out [[buffer(7)]],
    device SolinasFp128* state [[buffer(8)]],
    device SolinasFp128* partials [[buffer(9)]],
    constant InstructionClaimPhaseParams& params [[buffer(10)]],
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
        SolinasFp128 low = instruction_claim_combined(
            lookup_output[low_index],
            left_lookup_operand[low_index],
            right_lookup_operand[low_index],
            left_instruction_input[low_index],
            right_instruction_input[low_index],
            gamma_powers);
        SolinasFp128 high = instruction_claim_combined(
            lookup_output[high_index],
            left_lookup_operand[high_index],
            right_lookup_operand[high_index],
            left_instruction_input[high_index],
            right_instruction_input[high_index],
            gamma_powers);
        state[low_index] = low;
        state[high_index] = high;

        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0], solinas_mul_wide(weight, low));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight, solinas_add(high, solinas_sub(high, low))));
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

kernel void solinas_instruction_claim_bind_message(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant InstructionClaimPhaseParams& params [[buffer(6)]],
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
        uint source_index = 4u * pair;
        uint destination_index = 2u * pair;
        SolinasFp128 low = instruction_claim_bind(
            source[source_index], source[source_index + 1u], challenge);
        SolinasFp128 high = instruction_claim_bind(
            source[source_index + 2u], source[source_index + 3u], challenge);
        destination[destination_index] = low;
        destination[destination_index + 1u] = high;

        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0], solinas_mul_wide(weight, low));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight, solinas_add(high, solinas_sub(high, low))));
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

kernel void solinas_instruction_claim_bind_range(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant InstructionClaimBindRangeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.output_count) {
        return;
    }
    uint output_index = params.output_start + gid;
    uint source_index = params.source_offset + 2u * output_index;
    destination[params.destination_offset + output_index] = instruction_claim_bind(
        source[source_index], source[source_index + 1u], challenge);
}

kernel void solinas_instruction_claim_copy_prefix(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        destination[gid] = source[gid];
    }
}

kernel void solinas_instruction_claim_bound_message(
    device const SolinasFp128* state [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant InstructionClaimPhaseParams& params [[buffer(4)]],
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
        uint index = 2u * pair;
        SolinasFp128 low = state[index];
        SolinasFp128 high = state[index + 1u];
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0], solinas_mul_wide(weight, low));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight, solinas_add(high, solinas_sub(high, low))));
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

kernel void solinas_instruction_claim_open_core(
    device const ulong* lookup_output [[buffer(0)]],
    device const ulong* left_lookup_operand [[buffer(1)]],
    device const InstructionClaimRightLookup* right_lookup_operand [[buffer(2)]],
    device const ulong* left_instruction_input [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* partials [[buffer(6)]],
    constant InstructionClaimOpeningParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length
        || params.columns != INSTRUCTION_CLAIM_CORE_OPENINGS)
    {
        return;
    }

    SolinasFp128 sums[INSTRUCTION_CLAIM_CORE_OPENINGS];
    for (uint column = 0u; column < INSTRUCTION_CLAIM_CORE_OPENINGS; column++) {
        sums[column] = solinas_zero();
    }

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        instruction_claim_accumulate_core_openings(
            lookup_output[row_index],
            left_lookup_operand[row_index],
            right_lookup_operand[row_index],
            left_instruction_input[row_index],
            e_in[x_in],
            sums);
    }

    instruction_claim_finish_block(
        sums,
        INSTRUCTION_CLAIM_CORE_OPENINGS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_instruction_claim_open_lookup_operands(
    device const ulong* left_lookup_operand [[buffer(0)]],
    device const InstructionClaimRightLookup* right_lookup_operand [[buffer(1)]],
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
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight, instruction_claim_from_u64(left_lookup_operand[row_index])));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                instruction_claim_from_u128(
                    right_lookup_operand[row_index].words[0],
                    right_lookup_operand[row_index].words[1])));
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

kernel void solinas_instruction_claim_open_all(
    device const ulong* lookup_output [[buffer(0)]],
    device const ulong* left_lookup_operand [[buffer(1)]],
    device const InstructionClaimRightLookup* right_lookup_operand [[buffer(2)]],
    device const ulong* left_instruction_input [[buffer(3)]],
    device const InstructionClaimRightInput* right_instruction_input [[buffer(4)]],
    device const SolinasFp128* e_in [[buffer(5)]],
    device const SolinasFp128* e_out [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant InstructionClaimOpeningParams& params [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length
        || params.columns != INSTRUCTION_CLAIM_ALL_OPENINGS)
    {
        return;
    }

    SolinasFp128 sums[INSTRUCTION_CLAIM_ALL_OPENINGS];
    for (uint column = 0u; column < INSTRUCTION_CLAIM_ALL_OPENINGS; column++) {
        sums[column] = solinas_zero();
    }

    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = x_out * params.e_in_length + x_in;
        SolinasFp128 weight = e_in[x_in];
        instruction_claim_accumulate_core_openings(
            lookup_output[row_index],
            left_lookup_operand[row_index],
            right_lookup_operand[row_index],
            left_instruction_input[row_index],
            weight,
            sums);
        sums[4] = solinas_add(
            sums[4],
            solinas_mul_wide(
                weight,
                instruction_claim_from_i128_twos(
                    right_instruction_input[row_index].words[0],
                    right_instruction_input[row_index].words[1])));
    }

    instruction_claim_finish_block(
        sums,
        INSTRUCTION_CLAIM_ALL_OPENINGS,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane,
        simdgroup,
        threads / 32u);
}

kernel void solinas_instruction_claim_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant InstructionClaimReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint column = 0u; column < params.columns; column++) {
        SolinasFp128 value = gid < params.input_count
            ? input[column * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u && gid / 32u < params.output_count) {
            output[column * params.output_count + gid / 32u] = value;
        }
    }
}
