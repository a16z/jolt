#define BYTECODE_ROW_STAGES 9u
#define BYTECODE_ROW_BASE_STAGES 5u

struct BytecodeRowParams {
    uint rows;
    uint lo_length;
    uint hi_length;
    uint reserved;
};

struct BytecodeRowRootBindParams {
    uint source_length;
    uint output_length;
    uint2 reserved;
};

struct BytecodeRowTail {
    SolinasFp128 fused_inc;
    SolinasFp128 ra_zero;
    SolinasFp128 ra_one;
};

inline SolinasFp128 bytecode_row_signed_magnitude(ulong magnitude, bool negative)
{
    SolinasFp128 value = solinas_zero();
    value.limb[0] = (uint)magnitude;
    value.limb[1] = (uint)(magnitude >> 32);
    return negative ? solinas_sub(solinas_zero(), value) : value;
}

inline BytecodeRowTail bytecode_row_tail(
    device const ulong* rows,
    uint row_count,
    uint index,
    device const SolinasFp128* ra_zero,
    device const SolinasFp128* ra_one)
{
    ulong magnitude = booleanity_row_word(rows, row_count, 3u, index);
    ulong flags = booleanity_row_word(rows, row_count, 4u, index);
    BytecodeRowTail result;
    result.fused_inc = bytecode_row_signed_magnitude(
        magnitude,
        (flags >> 63) != 0ul);

    ulong pc_plus_one = flags & 0x00ffFFFFFFFFFFFFul;
    if (pc_plus_one == 0ul) {
        result.ra_zero = solinas_zero();
        result.ra_one = solinas_zero();
    } else {
        ulong mapped_pc = pc_plus_one - 1ul;
        result.ra_zero = ra_zero[(uint)(mapped_pc >> 8) & 0xffu];
        result.ra_one = ra_one[(uint)mapped_pc & 0xffu];
    }
    return result;
}

inline SolinasFp128 bytecode_row_bind(
    SolinasFp128 lo,
    SolinasFp128 hi,
    SolinasFp128 challenge)
{
    return solinas_add(
        lo,
        solinas_mul_wide(challenge, solinas_sub(hi, lo)));
}

inline void bytecode_row_coefficients(
    uint x_lo,
    uint lo_length,
    device const SolinasFp128* eq_lo,
    threadgroup const SolinasFp128* weighted_hi,
    thread SolinasFp128* factors)
{
    SolinasFp128 combined = solinas_zero();
    for (uint stage = 0u; stage < BYTECODE_ROW_BASE_STAGES; stage++) {
        combined = solinas_add(
            combined,
            solinas_mul_wide(
                weighted_hi[stage],
                eq_lo[stage * lo_length + x_lo]));
    }
    SolinasFp128 fused_combined = solinas_zero();
    for (uint stage = BYTECODE_ROW_BASE_STAGES; stage < BYTECODE_ROW_STAGES; stage++) {
        fused_combined = solinas_add(
            fused_combined,
            solinas_mul_wide(
                weighted_hi[stage],
                eq_lo[stage * lo_length + x_lo]));
    }
    factors[0] = combined;
    factors[1] = fused_combined;
}

kernel void solinas_bytecode_row_first_message(
    device const ulong* rows [[buffer(0)]],
    device const SolinasFp128* eq_lo [[buffer(1)]],
    device const SolinasFp128* weighted_eq_hi [[buffer(2)]],
    device const SolinasFp128* ra_zero [[buffer(3)]],
    device const SolinasFp128* ra_one [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& entry_weight [[buffer(6)]],
    constant BytecodeRowParams& params [[buffer(7)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_index [[thread_index_in_threadgroup]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (local_index < BYTECODE_ROW_STAGES) {
        shared[local_index] = weighted_eq_hi[
            local_index * params.hi_length + x_hi];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasFp128 lanes[BYTECODE_CYCLE_SAMPLES];
    for (uint sample = 0u; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pairs = params.lo_length / 2u;
    for (uint x_pair = local_index; x_pair < pairs; x_pair += threads) {
        uint x_lo = 2u * x_pair;
        uint row_zero = x_hi * params.lo_length + x_lo;
        SolinasFp128 lo[BYTECODE_CYCLE_TABLES];
        SolinasFp128 hi[BYTECODE_CYCLE_TABLES];
        bytecode_row_coefficients(x_lo, params.lo_length, eq_lo, shared, lo);
        bytecode_row_coefficients(x_lo + 1u, params.lo_length, eq_lo, shared, hi);
        if (row_zero == 0u) {
            lo[0] = solinas_add(lo[0], entry_weight);
        }
        BytecodeRowTail lo_tail = bytecode_row_tail(
            rows, params.rows, row_zero, ra_zero, ra_one);
        BytecodeRowTail hi_tail = bytecode_row_tail(
            rows, params.rows, row_zero + 1u, ra_zero, ra_one);
        lo[2] = lo_tail.fused_inc;
        lo[3] = lo_tail.ra_zero;
        lo[4] = lo_tail.ra_one;
        hi[2] = hi_tail.fused_inc;
        hi[3] = hi_tail.ra_zero;
        hi[4] = hi_tail.ra_one;

        SolinasFp128 q[BYTECODE_CYCLE_SAMPLES];
        bytecode_cycle_q10(lo, hi, q);
        for (uint sample = 0u; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            lanes[sample] = solinas_add(lanes[sample], q[sample]);
        }
    }

    bytecode_cycle_finish_block(
        lanes,
        partials,
        shared + BYTECODE_ROW_STAGES,
        x_hi,
        params.hi_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_bytecode_row_bind_lo_roots(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant BytecodeRowRootBindParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint elements = BYTECODE_ROW_STAGES * params.output_length;
    if (gid >= elements) {
        return;
    }
    uint stage = gid / params.output_length;
    uint output = gid - stage * params.output_length;
    uint input = stage * params.source_length + 2u * output;
    destination[gid] = bytecode_row_bind(source[input], source[input + 1u], challenge);
}

kernel void solinas_bytecode_row_first_bind_message(
    device const ulong* rows [[buffer(0)]],
    device const SolinasFp128* bound_eq_lo [[buffer(1)]],
    device const SolinasFp128* weighted_eq_hi [[buffer(2)]],
    device const SolinasFp128* ra_zero [[buffer(3)]],
    device const SolinasFp128* ra_one [[buffer(4)]],
    device SolinasFp128* bound_zero [[buffer(5)]],
    device SolinasFp128* bound_one [[buffer(6)]],
    device SolinasFp128* bound_two [[buffer(7)]],
    device SolinasFp128* bound_three [[buffer(8)]],
    device SolinasFp128* bound_four [[buffer(9)]],
    device SolinasFp128* partials [[buffer(10)]],
    constant SolinasFp128& challenge [[buffer(11)]],
    constant SolinasFp128& bound_entry_weight [[buffer(12)]],
    constant BytecodeRowParams& params [[buffer(13)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_index [[thread_index_in_threadgroup]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (local_index < BYTECODE_ROW_STAGES) {
        shared[local_index] = weighted_eq_hi[
            local_index * params.hi_length + x_hi];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasFp128 lanes[BYTECODE_CYCLE_SAMPLES];
    for (uint sample = 0u; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint bound_lo_length = params.lo_length / 2u;
    uint pairs = bound_lo_length / 2u;
    for (uint x_pair = local_index; x_pair < pairs; x_pair += threads) {
        uint destination = x_hi * bound_lo_length + 2u * x_pair;
        uint source = x_hi * params.lo_length + 4u * x_pair;
        SolinasFp128 lo[BYTECODE_CYCLE_TABLES];
        SolinasFp128 hi[BYTECODE_CYCLE_TABLES];
        bytecode_row_coefficients(
            2u * x_pair, bound_lo_length, bound_eq_lo, shared, lo);
        bytecode_row_coefficients(
            2u * x_pair + 1u, bound_lo_length, bound_eq_lo, shared, hi);
        if (destination == 0u) {
            lo[0] = solinas_add(lo[0], bound_entry_weight);
        }

        BytecodeRowTail row_zero = bytecode_row_tail(
            rows, params.rows, source, ra_zero, ra_one);
        BytecodeRowTail row_one = bytecode_row_tail(
            rows, params.rows, source + 1u, ra_zero, ra_one);
        BytecodeRowTail row_two = bytecode_row_tail(
            rows, params.rows, source + 2u, ra_zero, ra_one);
        BytecodeRowTail row_three = bytecode_row_tail(
            rows, params.rows, source + 3u, ra_zero, ra_one);
        lo[2] = bytecode_row_bind(row_zero.fused_inc, row_one.fused_inc, challenge);
        lo[3] = bytecode_row_bind(row_zero.ra_zero, row_one.ra_zero, challenge);
        lo[4] = bytecode_row_bind(row_zero.ra_one, row_one.ra_one, challenge);
        hi[2] = bytecode_row_bind(row_two.fused_inc, row_three.fused_inc, challenge);
        hi[3] = bytecode_row_bind(row_two.ra_zero, row_three.ra_zero, challenge);
        hi[4] = bytecode_row_bind(row_two.ra_one, row_three.ra_one, challenge);

        for (uint table = 0u; table < BYTECODE_CYCLE_TABLES; table++) {
            bytecode_cycle_store(
                table,
                destination,
                lo[table],
                bound_zero,
                bound_one,
                bound_two,
                bound_three,
                bound_four);
            bytecode_cycle_store(
                table,
                destination + 1u,
                hi[table],
                bound_zero,
                bound_one,
                bound_two,
                bound_three,
                bound_four);
        }

        SolinasFp128 q[BYTECODE_CYCLE_SAMPLES];
        bytecode_cycle_q10(lo, hi, q);
        for (uint sample = 0u; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            lanes[sample] = solinas_add(lanes[sample], q[sample]);
        }
    }

    bytecode_cycle_finish_block(
        lanes,
        partials,
        shared + BYTECODE_ROW_STAGES,
        x_hi,
        params.hi_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}
