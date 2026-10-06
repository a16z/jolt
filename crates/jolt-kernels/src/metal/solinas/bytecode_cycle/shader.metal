constant uint BYTECODE_CYCLE_TABLES [[function_constant(25)]];
#define BYTECODE_CYCLE_SAMPLES (BYTECODE_CYCLE_TABLES - 1u)

struct BytecodeCycleParams {
    uint source_elements;
    uint message_pairs;
    uint threadgroups;
    uint reserved;
};

inline void bytecode_cycle_grid_from_anchors(
    SolinasFp128 at_zero,
    SolinasFp128 at_one,
    SolinasFp128 leading,
    thread SolinasFp128* grid)
{
    SolinasFp128 second_difference = solinas_add(leading, leading);
    SolinasFp128 delta_two = solinas_add(
        solinas_sub(at_one, at_zero),
        second_difference);
    grid[0] = at_zero;
    grid[1] = solinas_add(at_one, delta_two);
    SolinasFp128 delta_three = solinas_add(delta_two, second_difference);
    grid[2] = solinas_add(grid[1], delta_three);
    grid[3] = solinas_add(solinas_add(grid[2], delta_three), second_difference);
}

inline void bytecode_cycle_evals(
    thread const SolinasFp128* lo,
    thread const SolinasFp128* hi,
    thread SolinasFp128* q)
{
    if (BYTECODE_CYCLE_TABLES == 8u) {
        SolinasFp128 values[8];
        SolinasFp128 delta[8];
        for (uint table = 0; table < 8u; table++) {
            values[table] = lo[table];
            delta[table] = solinas_sub(hi[table], lo[table]);
        }
        for (uint sample = 0; sample < 7u; sample++) {
            SolinasFp128 value = solinas_add(values[0], solinas_mul_wide(values[1], values[2]));
            for (uint table = 3; table < 8u; table++) {
                value = solinas_mul_wide(value, values[table]);
            }
            q[sample] = value;
            for (uint table = 0; table < 8u; table++) {
                values[table] = solinas_add(values[table], delta[table]);
                if (sample == 0u) values[table] = solinas_add(values[table], delta[table]);
            }
        }
        return;
    }
    SolinasFp128 ra_grid[7];
    SolinasFp128 ra_leading = solinas_mul_wide(
        solinas_sub(hi[3], lo[3]),
        solinas_sub(hi[4], lo[4]));
    bytecode_cycle_grid_from_anchors(
        solinas_mul_wide(lo[3], lo[4]),
        solinas_mul_wide(hi[3], hi[4]),
        ra_leading,
        ra_grid);

    SolinasFp128 coefficient_grid[7];
    SolinasFp128 coefficient_zero = solinas_add(
        lo[0],
        solinas_mul_wide(lo[2], lo[1]));
    SolinasFp128 coefficient_one = solinas_add(
        hi[0],
        solinas_mul_wide(hi[2], hi[1]));
    SolinasFp128 coefficient_leading = solinas_mul_wide(
        solinas_sub(hi[2], lo[2]),
        solinas_sub(hi[1], lo[1]));
    bytecode_cycle_grid_from_anchors(
        coefficient_zero,
        coefficient_one,
        coefficient_leading,
        coefficient_grid);

    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        q[sample] = solinas_mul_wide(ra_grid[sample], coefficient_grid[sample]);
    }
}

inline void bytecode_cycle_finish_block(
    thread SolinasFp128* lanes,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint group,
    uint groups,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane_in_simd == 0) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[sample * groups + group] = sum;
            }
        }
    }
}

inline SolinasFp128 bytecode_cycle_load(
    uint table,
    uint index,
    device const SolinasFp128* table_zero,
    device const SolinasFp128* table_one,
    device const SolinasFp128* table_two,
    device const SolinasFp128* table_three,
    device const SolinasFp128* table_four, device const SolinasFp128* table_five, device const SolinasFp128* table_six, device const SolinasFp128* table_seven)
{
    switch (table) {
        case 0u: return table_zero[index];
        case 1u: return table_one[index];
        case 2u: return table_two[index];
        case 3u: return table_three[index];
        case 4u: return table_four[index];
        case 5u: return table_five[index];
        case 6u: return table_six[index];
        default: return table_seven[index];
    }
}

inline void bytecode_cycle_store(
    uint table,
    uint index,
    SolinasFp128 value,
    device SolinasFp128* table_zero,
    device SolinasFp128* table_one,
    device SolinasFp128* table_two,
    device SolinasFp128* table_three,
    device SolinasFp128* table_four, device SolinasFp128* table_five, device SolinasFp128* table_six, device SolinasFp128* table_seven)
{
    switch (table) {
        case 0u: table_zero[index] = value; break;
        case 1u: table_one[index] = value; break;
        case 2u: table_two[index] = value; break;
        case 3u: table_three[index] = value; break;
        case 4u: table_four[index] = value; break;
        case 5u: table_five[index] = value; break;
        case 6u: table_six[index] = value; break;
        default: table_seven[index] = value; break;
    }
}

kernel void solinas_bytecode_cycle_message(
    device const SolinasFp128* table_zero [[buffer(0)]],
    device const SolinasFp128* table_one [[buffer(1)]],
    device const SolinasFp128* table_two [[buffer(2)]],
    device const SolinasFp128* table_three [[buffer(3)]],
    device const SolinasFp128* table_four [[buffer(4)]],
    device const SolinasFp128* table_five [[buffer(5)]],
    device const SolinasFp128* table_six [[buffer(6)]],
    device const SolinasFp128* table_seven [[buffer(7)]],
    device SolinasFp128* partials [[buffer(8)]],
    constant BytecodeCycleParams& params [[buffer(9)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[7];
    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pair = group * threads + local_index;
    uint stride = params.threadgroups * threads;
    for (; pair < params.message_pairs; pair += stride) {
        SolinasFp128 lo[8];
        SolinasFp128 hi[8];
        for (uint table = 0; table < BYTECODE_CYCLE_TABLES; table++) {
            uint source = 2u * pair;
            lo[table] = bytecode_cycle_load(
                table, source, table_zero, table_one, table_two, table_three, table_four, table_five, table_six, table_seven);
            hi[table] = bytecode_cycle_load(
                table, source + 1u, table_zero, table_one, table_two, table_three, table_four, table_five, table_six, table_seven);
        }
        SolinasFp128 q[7];
        bytecode_cycle_evals(lo, hi, q);
        for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            lanes[sample] = solinas_add(lanes[sample], q[sample]);
        }
    }

    bytecode_cycle_finish_block(
        lanes,
        partials,
        shared,
        group,
        params.threadgroups,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_bytecode_cycle_transition(
    device const SolinasFp128* source_zero [[buffer(0)]],
    device const SolinasFp128* source_one [[buffer(1)]],
    device const SolinasFp128* source_two [[buffer(2)]],
    device const SolinasFp128* source_three [[buffer(3)]],
    device const SolinasFp128* source_four [[buffer(4)]],
    device const SolinasFp128* source_five [[buffer(5)]],
    device const SolinasFp128* source_six [[buffer(6)]],
    device const SolinasFp128* source_seven [[buffer(7)]],
    device SolinasFp128* bound_zero [[buffer(8)]],
    device SolinasFp128* bound_one [[buffer(9)]],
    device SolinasFp128* bound_two [[buffer(10)]],
    device SolinasFp128* bound_three [[buffer(11)]],
    device SolinasFp128* bound_four [[buffer(12)]],
    device SolinasFp128* bound_five [[buffer(13)]],
    device SolinasFp128* bound_six [[buffer(14)]],
    device SolinasFp128* bound_seven [[buffer(15)]],
    device SolinasFp128* partials [[buffer(16)]],
    constant SolinasFp128& challenge [[buffer(17)]],
    constant BytecodeCycleParams& params [[buffer(18)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[7];
    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pair = group * threads + local_index;
    uint stride = params.threadgroups * threads;
    for (; pair < params.message_pairs; pair += stride) {
        SolinasFp128 lo[8];
        SolinasFp128 hi[8];
        for (uint table = 0; table < BYTECODE_CYCLE_TABLES; table++) {
            uint source = 4u * pair;
            SolinasFp128 value_zero = bytecode_cycle_load(
                table, source, source_zero, source_one, source_two, source_three, source_four, source_five, source_six, source_seven);
            SolinasFp128 value_one = bytecode_cycle_load(
                table, source + 1u, source_zero, source_one, source_two, source_three, source_four, source_five, source_six, source_seven);
            SolinasFp128 value_two = bytecode_cycle_load(
                table, source + 2u, source_zero, source_one, source_two, source_three, source_four, source_five, source_six, source_seven);
            SolinasFp128 value_three = bytecode_cycle_load(
                table, source + 3u, source_zero, source_one, source_two, source_three, source_four, source_five, source_six, source_seven);
            lo[table] = solinas_add(
                value_zero,
                solinas_mul_wide(challenge, solinas_sub(value_one, value_zero)));
            hi[table] = solinas_add(
                value_two,
                solinas_mul_wide(challenge, solinas_sub(value_three, value_two)));
            uint destination = 2u * pair;
            bytecode_cycle_store(
                table,
                destination,
                lo[table],
                bound_zero,
                bound_one,
                bound_two,
                bound_three,
                bound_four, bound_five, bound_six, bound_seven);
            bytecode_cycle_store(
                table,
                destination + 1u,
                hi[table],
                bound_zero,
                bound_one,
                bound_two,
                bound_three,
                bound_four, bound_five, bound_six, bound_seven);
        }
        SolinasFp128 q[7];
        bytecode_cycle_evals(lo, hi, q);
        for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            lanes[sample] = solinas_add(lanes[sample], q[sample]);
        }
    }

    bytecode_cycle_finish_block(
        lanes,
        partials,
        shared,
        group,
        params.threadgroups,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}
