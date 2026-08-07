// Append after fp128, simd_reduce, and the promoted half-width primitive.

struct SpartanShiftSuccessorFlagWord {
    uint is_virtual;
    uint is_first_in_sequence;
    uint is_noop;
};

struct SpartanShiftSuccessorPartialParams {
    uint rows;
    uint prefix_elements;
    uint suffix_elements;
    uint high_tile_elements;
    uint high_tiles;
    uint output_columns;
    uint2 reserved;
};

struct SpartanShiftSuccessorReductionParams {
    uint prefix_elements;
    uint high_tiles;
    uint columns;
    uint reserved;
};

struct SpartanShiftSuccessorFoldParams {
    uint rows;
    uint prefix_elements;
    uint suffix_elements;
    uint reserved;
};

inline bool spartan_shift_successor_flag(
    device const SpartanShiftSuccessorFlagWord* flags,
    uint row,
    uint plane)
{
    SpartanShiftSuccessorFlagWord word = flags[row >> 5u];
    uint mask = 1u << (row & 31u);
    if (plane == 0u) {
        return (word.is_virtual & mask) != 0u;
    }
    if (plane == 1u) {
        return (word.is_first_in_sequence & mask) != 0u;
    }
    return (word.is_noop & mask) != 0u;
}

inline bool spartan_shift_successor_masked_flag(uint word, uint mask) {
    return (word & mask) != 0u;
}

inline uint spartan_shift_successor_low_groups(
    constant SpartanShiftSuccessorPartialParams& params,
    uint threads)
{
    return (params.prefix_elements + threads - 1u) / threads;
}

kernel void solinas_spartan_shift_successor_outer_numeric(
    device const ulong* unexpanded_pc [[buffer(0)]],
    device const ulong* pc [[buffer(1)]],
    device const SolinasFp128* high_weights [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant SpartanShiftSuccessorPartialParams& params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint low_groups = spartan_shift_successor_low_groups(params, threads);
    uint high_tile = group / low_groups;
    uint low_group = group - high_tile * low_groups;
    uint x_lo = low_group * threads + tid;
    if (high_tile >= params.high_tiles || x_lo >= params.prefix_elements) {
        return;
    }

    uint high_start = high_tile * params.high_tile_elements;
    uint high_end = min(
        high_start + params.high_tile_elements,
        params.suffix_elements);
    SolinasFp128 upc_current = solinas_zero();
    SolinasFp128 upc_successor = solinas_zero();
    SolinasFp128 pc_current = solinas_zero();
    SolinasFp128 pc_successor = solinas_zero();

    for (uint high = high_start; high < high_end; high++) {
        uint row = high * params.prefix_elements + x_lo;
        SolinasFp128 current_weight = high_weights[high];
        ulong upc_value = unexpanded_pc[row];
        ulong pc_value = pc[row];
        upc_current = solinas_add(
            upc_current,
            solinas_half_width_mul_u64(current_weight, upc_value));
        pc_current = solinas_add(
            pc_current,
            solinas_half_width_mul_u64(current_weight, pc_value));
        if (high != 0u) {
            SolinasFp128 successor_weight = high_weights[high - 1u];
            upc_successor = solinas_add(
                upc_successor,
                solinas_half_width_mul_u64(successor_weight, upc_value));
            pc_successor = solinas_add(
                pc_successor,
                solinas_half_width_mul_u64(successor_weight, pc_value));
        }
    }

    uint partial_count = params.prefix_elements * params.high_tiles;
    uint partial = high_tile * params.prefix_elements + x_lo;
    partials[partial] = upc_current;
    partials[partial_count + partial] = upc_successor;
    partials[2u * partial_count + partial] = pc_current;
    partials[3u * partial_count + partial] = pc_successor;
}

kernel void solinas_spartan_shift_successor_outer_flags(
    device const SpartanShiftSuccessorFlagWord* flags [[buffer(0)]],
    device const SolinasFp128* high_weights [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant SpartanShiftSuccessorPartialParams& params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint low_groups = spartan_shift_successor_low_groups(params, threads);
    uint high_tile = group / low_groups;
    uint low_group = group - high_tile * low_groups;
    uint x_lo = low_group * threads + tid;
    if (high_tile >= params.high_tiles || x_lo >= params.prefix_elements) {
        return;
    }

    uint high_start = high_tile * params.high_tile_elements;
    uint high_end = min(
        high_start + params.high_tile_elements,
        params.suffix_elements);
    SolinasFp128 virtual_current = solinas_zero();
    SolinasFp128 virtual_successor = solinas_zero();
    SolinasFp128 first_current = solinas_zero();
    SolinasFp128 first_successor = solinas_zero();

    for (uint high = high_start; high < high_end; high++) {
        uint row = high * params.prefix_elements + x_lo;
        SolinasFp128 current_weight = high_weights[high];
        SpartanShiftSuccessorFlagWord flag_word = flags[row >> 5u];
        uint flag_mask = 1u << (row & 31u);
        bool is_virtual = spartan_shift_successor_masked_flag(
            flag_word.is_virtual, flag_mask);
        bool is_first = spartan_shift_successor_masked_flag(
            flag_word.is_first_in_sequence, flag_mask);
        if (is_virtual) {
            virtual_current = solinas_add(virtual_current, current_weight);
        }
        if (is_first) {
            first_current = solinas_add(first_current, current_weight);
        }
        if (high != 0u) {
            SolinasFp128 successor_weight = high_weights[high - 1u];
            if (is_virtual) {
                virtual_successor = solinas_add(
                    virtual_successor, successor_weight);
            }
            if (is_first) {
                first_successor = solinas_add(first_successor, successor_weight);
            }
        }
    }

    uint partial_count = params.prefix_elements * params.high_tiles;
    uint partial = high_tile * params.prefix_elements + x_lo;
    partials[partial] = virtual_current;
    partials[partial_count + partial] = virtual_successor;
    partials[2u * partial_count + partial] = first_current;
    partials[3u * partial_count + partial] = first_successor;
}

kernel void solinas_spartan_shift_successor_product_flags(
    device const SpartanShiftSuccessorFlagWord* flags [[buffer(0)]],
    device const SolinasFp128* high_weights [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant SpartanShiftSuccessorPartialParams& params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint low_groups = spartan_shift_successor_low_groups(params, threads);
    uint high_tile = group / low_groups;
    uint low_group = group - high_tile * low_groups;
    uint x_lo = low_group * threads + tid;
    if (high_tile >= params.high_tiles || x_lo >= params.prefix_elements) {
        return;
    }

    uint high_start = high_tile * params.high_tile_elements;
    uint high_end = min(
        high_start + params.high_tile_elements,
        params.suffix_elements);
    SolinasFp128 nonnoop_current = solinas_zero();
    SolinasFp128 nonnoop_successor = solinas_zero();
    for (uint high = high_start; high < high_end; high++) {
        uint row = high * params.prefix_elements + x_lo;
        if (!spartan_shift_successor_flag(flags, row, 2u)) {
            nonnoop_current = solinas_add(nonnoop_current, high_weights[high]);
            if (high != 0u) {
                nonnoop_successor = solinas_add(
                    nonnoop_successor, high_weights[high - 1u]);
            }
        }
    }

    uint partial_count = params.prefix_elements * params.high_tiles;
    uint partial = high_tile * params.prefix_elements + x_lo;
    partials[partial] = nonnoop_current;
    partials[partial_count + partial] = nonnoop_successor;
}

kernel void solinas_spartan_shift_successor_reduce_partials(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* tables [[buffer(1)]],
    constant SpartanShiftSuccessorReductionParams& params [[buffer(2)]],
    uint x_lo [[thread_position_in_grid]])
{
    if (x_lo >= params.prefix_elements) {
        return;
    }
    uint partial_count = params.prefix_elements * params.high_tiles;
    for (uint column = 0u; column < params.columns; column++) {
        SolinasFp128 sum = solinas_zero();
        uint column_start = column * partial_count;
        for (uint tile = 0u; tile < params.high_tiles; tile++) {
            sum = solinas_add(
                sum,
                partials[column_start + tile * params.prefix_elements + x_lo]);
        }
        tables[column * params.prefix_elements + x_lo] = sum;
    }
}

kernel void solinas_spartan_shift_successor_fold_residual(
    device const ulong* pc [[buffer(0)]],
    device const SpartanShiftSuccessorFlagWord* flags [[buffer(1)]],
    device const SolinasFp128* low_weights [[buffer(2)]],
    device SolinasFp128* outputs [[buffer(3)]],
    constant SpartanShiftSuccessorFoldParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_hi >= params.suffix_elements) {
        return;
    }
    SolinasFp128 pc_sum = solinas_zero();
    SolinasFp128 virtual_sum = solinas_zero();
    SolinasFp128 first_sum = solinas_zero();
    SolinasFp128 noop_sum = solinas_zero();
    uint row_start = x_hi * params.prefix_elements;
    for (uint x_lo = tid; x_lo < params.prefix_elements; x_lo += threads) {
        uint row = row_start + x_lo;
        SolinasFp128 weight = low_weights[x_lo];
        SpartanShiftSuccessorFlagWord flag_word = flags[row >> 5u];
        uint flag_mask = 1u << (row & 31u);
        pc_sum = solinas_add(
            pc_sum,
            solinas_half_width_mul_u64(weight, pc[row]));
        if (spartan_shift_successor_masked_flag(
                flag_word.is_virtual, flag_mask)) {
            virtual_sum = solinas_add(virtual_sum, weight);
        }
        if (spartan_shift_successor_masked_flag(
                flag_word.is_first_in_sequence, flag_mask)) {
            first_sum = solinas_add(first_sum, weight);
        }
        if (spartan_shift_successor_masked_flag(flag_word.is_noop, flag_mask)) {
            noop_sum = solinas_add(noop_sum, weight);
        }
    }

    pc_sum = solinas_simd_sum_32(pc_sum);
    virtual_sum = solinas_simd_sum_32(virtual_sum);
    first_sum = solinas_simd_sum_32(first_sum);
    noop_sum = solinas_simd_sum_32(noop_sum);
    uint simdgroups = threads / 32u;
    if (lane == 0u) {
        shared[simdgroup] = pc_sum;
        shared[simdgroups + simdgroup] = virtual_sum;
        shared[2u * simdgroups + simdgroup] = first_sum;
        shared[3u * simdgroups + simdgroup] = noop_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        SolinasFp128 pc_total = solinas_zero();
        SolinasFp128 virtual_total = solinas_zero();
        SolinasFp128 first_total = solinas_zero();
        SolinasFp128 noop_total = solinas_zero();
        for (uint group = 0u; group < simdgroups; group++) {
            pc_total = solinas_add(pc_total, shared[group]);
            virtual_total = solinas_add(
                virtual_total, shared[simdgroups + group]);
            first_total = solinas_add(
                first_total, shared[2u * simdgroups + group]);
            noop_total = solinas_add(
                noop_total, shared[3u * simdgroups + group]);
        }
        outputs[x_hi] = pc_total;
        outputs[params.suffix_elements + x_hi] = virtual_total;
        outputs[2u * params.suffix_elements + x_hi] = first_total;
        outputs[3u * params.suffix_elements + x_hi] = noop_total;
    }
}

kernel void solinas_spartan_shift_successor_fold_full(
    device const ulong* unexpanded_pc [[buffer(0)]],
    device const ulong* pc [[buffer(1)]],
    device const SpartanShiftSuccessorFlagWord* flags [[buffer(2)]],
    device const SolinasFp128* low_weights [[buffer(3)]],
    device SolinasFp128* outputs [[buffer(4)]],
    constant SpartanShiftSuccessorFoldParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (x_hi >= params.suffix_elements) {
        return;
    }
    SolinasFp128 upc_sum = solinas_zero();
    SolinasFp128 pc_sum = solinas_zero();
    SolinasFp128 virtual_sum = solinas_zero();
    SolinasFp128 first_sum = solinas_zero();
    SolinasFp128 noop_sum = solinas_zero();
    uint row_start = x_hi * params.prefix_elements;
    for (uint x_lo = tid; x_lo < params.prefix_elements; x_lo += threads) {
        uint row = row_start + x_lo;
        SolinasFp128 weight = low_weights[x_lo];
        SpartanShiftSuccessorFlagWord flag_word = flags[row >> 5u];
        uint flag_mask = 1u << (row & 31u);
        upc_sum = solinas_add(
            upc_sum,
            solinas_half_width_mul_u64(weight, unexpanded_pc[row]));
        pc_sum = solinas_add(
            pc_sum,
            solinas_half_width_mul_u64(weight, pc[row]));
        if (spartan_shift_successor_masked_flag(
                flag_word.is_virtual, flag_mask)) {
            virtual_sum = solinas_add(virtual_sum, weight);
        }
        if (spartan_shift_successor_masked_flag(
                flag_word.is_first_in_sequence, flag_mask)) {
            first_sum = solinas_add(first_sum, weight);
        }
        if (spartan_shift_successor_masked_flag(flag_word.is_noop, flag_mask)) {
            noop_sum = solinas_add(noop_sum, weight);
        }
    }

    upc_sum = solinas_simd_sum_32(upc_sum);
    pc_sum = solinas_simd_sum_32(pc_sum);
    virtual_sum = solinas_simd_sum_32(virtual_sum);
    first_sum = solinas_simd_sum_32(first_sum);
    noop_sum = solinas_simd_sum_32(noop_sum);
    uint simdgroups = threads / 32u;
    if (lane == 0u) {
        shared[simdgroup] = upc_sum;
        shared[simdgroups + simdgroup] = pc_sum;
        shared[2u * simdgroups + simdgroup] = virtual_sum;
        shared[3u * simdgroups + simdgroup] = first_sum;
        shared[4u * simdgroups + simdgroup] = noop_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        SolinasFp128 upc_total = solinas_zero();
        SolinasFp128 pc_total = solinas_zero();
        SolinasFp128 virtual_total = solinas_zero();
        SolinasFp128 first_total = solinas_zero();
        SolinasFp128 noop_total = solinas_zero();
        for (uint group = 0u; group < simdgroups; group++) {
            upc_total = solinas_add(upc_total, shared[group]);
            pc_total = solinas_add(pc_total, shared[simdgroups + group]);
            virtual_total = solinas_add(
                virtual_total, shared[2u * simdgroups + group]);
            first_total = solinas_add(
                first_total, shared[3u * simdgroups + group]);
            noop_total = solinas_add(
                noop_total, shared[4u * simdgroups + group]);
        }
        outputs[x_hi] = upc_total;
        outputs[params.suffix_elements + x_hi] = pc_total;
        outputs[2u * params.suffix_elements + x_hi] = virtual_total;
        outputs[3u * params.suffix_elements + x_hi] = first_total;
        outputs[4u * params.suffix_elements + x_hi] = noop_total;
    }
}
