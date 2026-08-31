#define SUCCESSOR_MESSAGE_COLUMNS 4u

struct ProductInstructionPhaseParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint reserved;
};

inline SolinasFp128 stage1_instruction_combined(
    device const InstructionInputRow& compact,
    device const SpartanOuterSuccessorRow& residual,
    constant const SolinasFp128* gamma_powers)
{
    ulong flags = instruction_input_row_word(compact, 5u);
    SolinasFp128 value = product_remainder_from_u64(
        spartan_outer_successor_word(residual, 13u));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[0],
            product_remainder_from_u64(
                spartan_outer_successor_word(residual, 8u))));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[1],
            instruction_claim_from_u128(
                spartan_outer_successor_word(residual, 9u),
                spartan_outer_successor_word(residual, 10u))));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[2],
            product_remainder_from_u64(
                spartan_outer_successor_word(residual, 0u))));
    value = solinas_add(
        value,
        solinas_mul_wide(
            gamma_powers[3],
            product_remainder_from_signed_u128(
                spartan_outer_successor_word(residual, 1u),
                spartan_outer_successor_word(residual, 2u),
                product_remainder_flag(
                    flags,
                    SPARTAN_PRODUCT_FLAG_RIGHT_NONNEGATIVE))));
    return value;
}

kernel void solinas_instruction_claim_materialize_stage1_rows(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
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
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
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
        device const SpartanOuterSuccessorRow& residual = residual_rows[row_index];
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(
                weight,
                instruction_claim_from_u64(
                    spartan_outer_successor_word(residual, 8u))));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(
                weight,
                instruction_claim_from_u128(
                    spartan_outer_successor_word(residual, 9u),
                    spartan_outer_successor_word(residual, 10u))));
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
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
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

#define TERMINAL_CACHE_FLAG_JUMP 0u
#define TERMINAL_CACHE_FLAG_WRITE_LOOKUP 1u
#define TERMINAL_CACHE_FLAG_BRANCH 2u
#define TERMINAL_CACHE_FLAG_NEXT_IS_NOOP 3u
#define TERMINAL_CACHE_FLAG_VIRTUAL 4u
#define TERMINAL_CACHE_LEFT_LOOKUP_TAG 0x80000000u
#define TERMINAL_CACHE_OVERFLOW 0x80000000u

inline uchar product_instruction_terminal_cache_flags(ulong flags) {
    return uchar(
        uint(product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_JUMP))
        | (uint(product_remainder_flag(
               flags, SPARTAN_PRODUCT_FLAG_WRITE_LOOKUP)) << 1u)
        | (uint(product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_BRANCH)) << 2u)
        | (uint(product_remainder_flag(
               flags, SPARTAN_PRODUCT_FLAG_NEXT_IS_NOOP)) << 3u)
        | (uint(product_remainder_flag(flags, SPARTAN_PRODUCT_FLAG_VIRTUAL)) << 4u));
}

inline void product_instruction_terminal_cache_exception(
    ulong value,
    uint local_row,
    uint tag,
    uint x_out,
    uint exceptions_per_group,
    device ulong* exceptions,
    threadgroup atomic_uint* cache_state)
{
    uint high = uint(value >> 32u);
    if (high == 0u) {
        return;
    }
    uint slot = atomic_fetch_add_explicit(
        &cache_state[0], 1u, memory_order_relaxed);
    if (slot < exceptions_per_group) {
        exceptions[x_out * exceptions_per_group + slot] =
            (ulong(high) << 32u) | ulong(local_row | tag);
    } else {
        atomic_store_explicit(&cache_state[1], 1u, memory_order_relaxed);
    }
}

kernel void solinas_product_instruction_materialize_stage1_message_cached(
    device const InstructionInputRow* compact_rows [[buffer(0)]],
    device const SpartanOuterSuccessorRow* residual_rows [[buffer(1)]],
    device const SolinasFp128* lagrange [[buffer(2)]],
    constant const SolinasFp128* gamma_powers [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device SolinasFp128* product_state [[buffer(6)]],
    device SolinasFp128* instruction_state [[buffer(7)]],
    device SolinasFp128* product_partials [[buffer(8)]],
    device SolinasFp128* instruction_partials [[buffer(9)]],
    device uint* lookup_low [[buffer(10)]],
    device uint* left_lookup_low [[buffer(11)]],
    device uchar* cache_flags [[buffer(12)]],
    device uint* exception_counts [[buffer(13)]],
    device ulong* exceptions [[buffer(14)]],
    constant ProductInstructionPhaseParams& params [[buffer(15)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    threadgroup atomic_uint* cache_state [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_out >= params.e_out_length) {
        return;
    }
    if (tid < 2u) {
        atomic_store_explicit(&cache_state[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

        ulong low_lookup = spartan_outer_successor_word(residual_rows[low_index], 13u);
        ulong high_lookup = spartan_outer_successor_word(residual_rows[high_index], 13u);
        ulong low_left_lookup = spartan_outer_successor_word(residual_rows[low_index], 8u);
        ulong high_left_lookup = spartan_outer_successor_word(residual_rows[high_index], 8u);
        lookup_low[low_index] = uint(low_lookup);
        lookup_low[high_index] = uint(high_lookup);
        left_lookup_low[low_index] = uint(low_left_lookup);
        left_lookup_low[high_index] = uint(high_left_lookup);
        cache_flags[low_index] = product_instruction_terminal_cache_flags(
            instruction_input_row_word(compact_rows[low_index], 5u));
        cache_flags[high_index] = product_instruction_terminal_cache_flags(
            instruction_input_row_word(compact_rows[high_index], 5u));
        uint local_low = 2u * x_in;
        uint local_high = local_low + 1u;
        product_instruction_terminal_cache_exception(
            low_lookup,
            local_low,
            0u,
            x_out,
            params.reserved,
            exceptions,
            cache_state);
        product_instruction_terminal_cache_exception(
            low_left_lookup,
            local_low,
            TERMINAL_CACHE_LEFT_LOOKUP_TAG,
            x_out,
            params.reserved,
            exceptions,
            cache_state);
        product_instruction_terminal_cache_exception(
            high_lookup,
            local_high,
            0u,
            x_out,
            params.reserved,
            exceptions,
            cache_state);
        product_instruction_terminal_cache_exception(
            high_left_lookup,
            local_high,
            TERMINAL_CACHE_LEFT_LOOKUP_TAG,
            x_out,
            params.reserved,
            exceptions,
            cache_state);

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

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint count = atomic_load_explicit(&cache_state[0], memory_order_relaxed);
        uint overflow = atomic_load_explicit(&cache_state[1], memory_order_relaxed);
        exception_counts[x_out] = min(count, params.reserved)
            | (overflow != 0u ? TERMINAL_CACHE_OVERFLOW : 0u);
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

kernel void solinas_product_instruction_terminal_cache_openings(
    device const uint* lookup_low [[buffer(0)]],
    device const uint* left_lookup_low [[buffer(1)]],
    device const uchar* cache_flags [[buffer(2)]],
    device const uint* exception_counts [[buffer(3)]],
    device const ulong* exceptions [[buffer(4)]],
    device const SolinasFp128* e_in [[buffer(5)]],
    device const SolinasFp128* e_out [[buffer(6)]],
    device SolinasFp128* partials [[buffer(7)]],
    constant ProductInstructionPhaseParams& params [[buffer(8)]],
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
    SolinasFp128 sums[PRODUCT_REMAINDER_OPENINGS];
    for (uint column = 0u; column < PRODUCT_REMAINDER_OPENINGS; column++) {
        sums[column] = solinas_zero();
    }
    uint group_begin = x_out * params.e_in_length;
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row = group_begin + x_in;
        SolinasFp128 weight = e_in[x_in];
        sums[0] = solinas_add(
            sums[0],
            solinas_mul_wide(weight, product_remainder_from_u64(lookup_low[row])));
        sums[1] = solinas_add(
            sums[1],
            solinas_mul_wide(weight, product_remainder_from_u64(left_lookup_low[row])));
        uchar flags = cache_flags[row];
        if ((flags & (1u << TERMINAL_CACHE_FLAG_JUMP)) != 0u) {
            sums[2] = solinas_add(sums[2], weight);
        }
        if ((flags & (1u << TERMINAL_CACHE_FLAG_WRITE_LOOKUP)) != 0u) {
            sums[3] = solinas_add(sums[3], weight);
        }
        if ((flags & (1u << TERMINAL_CACHE_FLAG_BRANCH)) != 0u) {
            sums[4] = solinas_add(sums[4], weight);
        }
        if ((flags & (1u << TERMINAL_CACHE_FLAG_NEXT_IS_NOOP)) != 0u) {
            sums[5] = solinas_add(sums[5], weight);
        }
        if ((flags & (1u << TERMINAL_CACHE_FLAG_VIRTUAL)) != 0u) {
            sums[6] = solinas_add(sums[6], weight);
        }
    }
    uint groups_per_cache = params.source_elements / params.e_in_length;
    uint cache_group = x_out / groups_per_cache;
    uint cache_subgroup = x_out % groups_per_cache;
    uint count = exception_counts[cache_group] & ~TERMINAL_CACHE_OVERFLOW;
    uint exception_begin = cache_group * params.reserved;
    for (uint index = tid; index < count; index += threads) {
        ulong entry = exceptions[exception_begin + index];
        uint tagged_row = uint(entry);
        uint local_row = tagged_row & ~TERMINAL_CACHE_LEFT_LOOKUP_TAG;
        if (local_row / params.e_in_length != cache_subgroup) {
            continue;
        }
        uint high = uint(entry >> 32u);
        SolinasFp128 contribution = solinas_mul_wide(
            e_in[local_row % params.e_in_length],
            product_remainder_from_u64(ulong(high) << 32u));
        uint column = (tagged_row & TERMINAL_CACHE_LEFT_LOOKUP_TAG) != 0u ? 1u : 0u;
        sums[column] = solinas_add(sums[column], contribution);
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
