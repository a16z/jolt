// Concatenate after fp128.metal.

#define HAMMING_WEIGHT_SELECTORS 29u
#define HAMMING_WEIGHT_BINS 256u
#define HAMMING_WEIGHT_RETAINED_BINS 255u
#define HAMMING_WEIGHT_SIMD_WIDTH 32u
#define HAMMING_WEIGHT_THREADS 928u
#define HAMMING_WEIGHT_STAGE_ROWS 512u
#define HAMMING_WEIGHT_LOADER_SIMDGROUPS 16u
#define HAMMING_WEIGHT_HOT_BYTES 14848u
#define HAMMING_WEIGHT_WEIGHT_BYTES 8192u
#define HAMMING_WEIGHT_AUDIT_WORDS 48u
#define HAMMING_WEIGHT_PC_MASK 0x00fffffffffffffful

struct HammingWeightResidentRow {
    ulong lookup_lo;
    ulong lookup_hi;
    ulong ram_address_plus_one;
    ulong fused_inc_magnitude;
    ulong packed_pc_and_flags;
};

inline HammingWeightResidentRow hamming_weight_load_row(
    device const ulong* rows,
    uint row_count,
    uint index)
{
    BooleanityRow source = booleanity_row_load(rows, row_count, index);
    HammingWeightResidentRow row;
    row.lookup_lo = source.lookup_lo;
    row.lookup_hi = source.lookup_hi;
    row.ram_address_plus_one = source.ram_address_plus_one;
    row.fused_inc_magnitude = source.fused_inc_magnitude;
    row.packed_pc_and_flags = source.packed_pc_and_flags;
    return row;
}

struct HammingWeightHistogramParams {
    uint rows;
    uint inner_length;
    uint outer_length;
    uint selectors;
    uint bins;
    uint stage_rows;
    uint simd_width;
    uint threads;
    ulong inc_bias;
    uint2 reserved;
};

struct HammingWeightAuditRow {
    uint rows_seen;
    uint pc_present;
    uint ram_present;
    uint retained_nonzero_contributions;
    uint occupied_outer_bins;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

struct HammingWeightBins {
    SolinasFp128 value[8];
};

inline bool hamming_weight_supported(
    constant HammingWeightHistogramParams& params,
    uint3 threads_per_group)
{
    return params.rows != 0u
        && params.inner_length != 0u
        && params.outer_length != 0u
        && (ulong)params.inner_length * (ulong)params.outer_length
            == (ulong)params.rows
        && (params.inner_length % HAMMING_WEIGHT_STAGE_ROWS) == 0u
        && params.selectors == HAMMING_WEIGHT_SELECTORS
        && params.bins == HAMMING_WEIGHT_BINS
        && params.stage_rows == HAMMING_WEIGHT_STAGE_ROWS
        && params.simd_width == HAMMING_WEIGHT_SIMD_WIDTH
        && params.threads == HAMMING_WEIGHT_THREADS
        && threads_per_group.x == HAMMING_WEIGHT_THREADS
        && threads_per_group.y == 1u
        && threads_per_group.z == 1u;
}

inline uint hamming_weight_simd_sum_u32(uint value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value += simd_shuffle_down(value, offset);
    }
    return value;
}

inline SolinasFp128 hamming_weight_simd_broadcast_zero(SolinasFp128 value) {
    value.limb = simd_broadcast(value.limb, 0u);
    return value;
}

inline bool hamming_weight_is_zero(SolinasFp128 value) {
    return all(value.limb == uint4(0u));
}

inline void hamming_weight_bins_zero(thread HammingWeightBins& bins) {
    for (uint bucket = 0u; bucket < 8u; bucket++) {
        bins.value[bucket] = solinas_zero();
    }
}

inline void hamming_weight_bins_add(
    thread HammingWeightBins& bins,
    uint bucket,
    SolinasFp128 value)
{
    switch (bucket) {
        case 0u: bins.value[0] = solinas_add(bins.value[0], value); break;
        case 1u: bins.value[1] = solinas_add(bins.value[1], value); break;
        case 2u: bins.value[2] = solinas_add(bins.value[2], value); break;
        case 3u: bins.value[3] = solinas_add(bins.value[3], value); break;
        case 4u: bins.value[4] = solinas_add(bins.value[4], value); break;
        case 5u: bins.value[5] = solinas_add(bins.value[5], value); break;
        case 6u: bins.value[6] = solinas_add(bins.value[6], value); break;
        default: bins.value[7] = solinas_add(bins.value[7], value); break;
    }
}

inline void hamming_weight_store_hot(
    threadgroup uchar* stage_hot,
    uint selector,
    uint local,
    uint hot,
    thread uint& retained)
{
    stage_hot[selector * HAMMING_WEIGHT_STAGE_ROWS + local] = (uchar)hot;
    retained += uint(hot != 0u);
}

inline void hamming_weight_decode_row(
    HammingWeightResidentRow row,
    threadgroup uchar* stage_hot,
    uint local,
    constant HammingWeightHistogramParams& params,
    thread uint& pc_present,
    thread uint& ram_present,
    thread uint& retained)
{
    for (uint selector = 0u; selector < 8u; selector++) {
        hamming_weight_store_hot(
            stage_hot,
            selector,
            local,
            (uint)(row.lookup_hi >> (8u * (7u - selector))) & 0xffu,
            retained);
    }
    for (uint selector = 8u; selector < 16u; selector++) {
        hamming_weight_store_hot(
            stage_hot,
            selector,
            local,
            (uint)(row.lookup_lo >> (8u * (15u - selector))) & 0xffu,
            retained);
    }

    ulong pc_plus_one = row.packed_pc_and_flags & HAMMING_WEIGHT_PC_MASK;
    pc_present = uint(pc_plus_one != 0ul);
    ulong pc = pc_plus_one == 0ul ? 0ul : pc_plus_one - 1ul;
    hamming_weight_store_hot(
        stage_hot, 16u, local, (uint)(pc >> 8u) & 0xffu, retained);
    hamming_weight_store_hot(
        stage_hot, 17u, local, (uint)pc & 0xffu, retained);

    ram_present = uint(row.ram_address_plus_one != 0ul);
    ulong ram = row.ram_address_plus_one == 0ul
        ? 0ul
        : row.ram_address_plus_one - 1ul;
    hamming_weight_store_hot(
        stage_hot, 18u, local, (uint)(ram >> 8u) & 0xffu, retained);
    hamming_weight_store_hot(
        stage_hot, 19u, local, (uint)ram & 0xffu, retained);

    bool negative = (row.packed_pc_and_flags >> 63) != 0ul;
    ulong biased;
    int carry;
    if (negative) {
        biased = params.inc_bias - row.fused_inc_magnitude;
        carry = row.fused_inc_magnitude > params.inc_bias ? -1 : 0;
    } else {
        biased = params.inc_bias + row.fused_inc_magnitude;
        carry = biased < params.inc_bias ? 1 : 0;
    }
    for (uint index = 0u; index < 8u; index++) {
        uint standard = (uint)(biased >> (8u * index)) & 0xffu;
        hamming_weight_store_hot(
            stage_hot,
            20u + index,
            local,
            (standard + 128u) & 0xffu,
            retained);
    }
    hamming_weight_store_hot(
        stage_hot, 28u, local, (uint)carry & 0xffu, retained);
}

inline SolinasFp128 hamming_weight_stage_weight(
    threadgroup uint* words,
    uint local)
{
    uint base = 4u * local;
    SolinasFp128 value;
    value.limb = uint4(
        words[base],
        words[base + 1u],
        words[base + 2u],
        words[base + 3u]);
    return value;
}

kernel void solinas_hamming_weight_register_histogram(
    device const ulong* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    device HammingWeightAuditRow* audits [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant HammingWeightHistogramParams& params [[buffer(6)]],
    threadgroup uchar* scratch [[threadgroup(0)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    uint outer = group_position.x;
    if (!hamming_weight_supported(params, threads_per_group)) {
        if (outer == 0u && tid == 0u) {
            atomic_fetch_add_explicit(&status[0], 1u, memory_order_relaxed);
        }
        return;
    }
    if (outer >= params.outer_length) {
        return;
    }

    threadgroup uchar* stage_hot = scratch;
    threadgroup uint* stage_weights =
        (threadgroup uint*)(scratch + HAMMING_WEIGHT_HOT_BYTES);
    threadgroup uint* stage_audit =
        (threadgroup uint*)(scratch
            + HAMMING_WEIGHT_HOT_BYTES
            + HAMMING_WEIGHT_WEIGHT_BYTES);

    HammingWeightBins bins;
    hamming_weight_bins_zero(bins);
    uint group_pc_present = 0u;
    uint group_ram_present = 0u;
    uint group_retained = 0u;
    uint row_base = outer * params.inner_length;

    for (uint inner_base = 0u;
         inner_base < params.inner_length;
         inner_base += HAMMING_WEIGHT_STAGE_ROWS) {
        uint local_pc_present = 0u;
        uint local_ram_present = 0u;
        uint local_retained = 0u;
        if (tid < HAMMING_WEIGHT_STAGE_ROWS) {
            uint inner = inner_base + tid;
            HammingWeightResidentRow row = hamming_weight_load_row(
                rows, params.rows, row_base + inner);
            hamming_weight_decode_row(
                row,
                stage_hot,
                tid,
                params,
                local_pc_present,
                local_ram_present,
                local_retained);
            SolinasFp128 weight = e_in[inner];
            uint weight_base = 4u * tid;
            stage_weights[weight_base] = weight.limb[0];
            stage_weights[weight_base + 1u] = weight.limb[1];
            stage_weights[weight_base + 2u] = weight.limb[2];
            stage_weights[weight_base + 3u] = weight.limb[3];
        }

        uint pc_sum = hamming_weight_simd_sum_u32(local_pc_present);
        uint ram_sum = hamming_weight_simd_sum_u32(local_ram_present);
        uint retained_sum = hamming_weight_simd_sum_u32(local_retained);
        if (lane == 0u && simdgroup < HAMMING_WEIGHT_LOADER_SIMDGROUPS) {
            stage_audit[simdgroup] = pc_sum;
            stage_audit[HAMMING_WEIGHT_LOADER_SIMDGROUPS + simdgroup] = ram_sum;
            stage_audit[2u * HAMMING_WEIGHT_LOADER_SIMDGROUPS + simdgroup] = retained_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local = 0u; local < HAMMING_WEIGHT_STAGE_ROWS; local++) {
            uint hot = lane == 0u
                ? (uint)stage_hot[simdgroup * HAMMING_WEIGHT_STAGE_ROWS + local]
                : 0u;
            hot = simd_broadcast(hot, 0u);
            if (hot != 0u && lane == (hot & 31u)) {
                hamming_weight_bins_add(
                    bins,
                    hot >> 5u,
                    hamming_weight_stage_weight(stage_weights, local));
            }
        }

        if (tid == 0u) {
            for (uint group = 0u; group < HAMMING_WEIGHT_LOADER_SIMDGROUPS; group++) {
                group_pc_present += stage_audit[group];
                group_ram_present +=
                    stage_audit[HAMMING_WEIGHT_LOADER_SIMDGROUPS + group];
                group_retained +=
                    stage_audit[2u * HAMMING_WEIGHT_LOADER_SIMDGROUPS + group];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    SolinasFp128 outer_weight = lane == 0u ? e_out[outer] : solinas_zero();
    outer_weight = hamming_weight_simd_broadcast_zero(outer_weight);
    uint occupied = 0u;
    for (uint bucket = 0u; bucket < 8u; bucket++) {
        uint bin = bucket * HAMMING_WEIGHT_SIMD_WIDTH + lane;
        if (bin == 0u) {
            continue;
        }
        SolinasFp128 value = bins.value[bucket];
        bool present = !hamming_weight_is_zero(value);
        occupied += uint(present);
        uint output = (outer * HAMMING_WEIGHT_SELECTORS + simdgroup)
            * HAMMING_WEIGHT_RETAINED_BINS
            + bin - 1u;
        partials[output] = present
            ? solinas_mul_wide(outer_weight, value)
            : solinas_zero();
    }
    occupied = hamming_weight_simd_sum_u32(occupied);
    if (lane == 0u) {
        stage_audit[simdgroup] = occupied;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint group_occupied = 0u;
        for (uint group = 0u; group < HAMMING_WEIGHT_SELECTORS; group++) {
            group_occupied += stage_audit[group];
        }
        HammingWeightAuditRow audit;
        audit.rows_seen = params.inner_length;
        audit.pc_present = group_pc_present;
        audit.ram_present = group_ram_present;
        audit.retained_nonzero_contributions = group_retained;
        audit.occupied_outer_bins = group_occupied;
        audit.reserved_0 = 0u;
        audit.reserved_1 = 0u;
        audit.reserved_2 = 0u;
        audits[outer] = audit;
    }
}

kernel void solinas_hamming_weight_register_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant HammingWeightHistogramParams& params [[buffer(3)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint bin [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    uint selector = group_position.x;
    bool supported = params.selectors == HAMMING_WEIGHT_SELECTORS
        && params.bins == HAMMING_WEIGHT_BINS
        && threads_per_group.x == HAMMING_WEIGHT_BINS
        && threads_per_group.y == 1u
        && threads_per_group.z == 1u;
    if (!supported) {
        if (selector == 0u && bin == 0u) {
            atomic_fetch_add_explicit(&status[0], 1u, memory_order_relaxed);
        }
        return;
    }
    if (selector >= HAMMING_WEIGHT_SELECTORS) {
        return;
    }
    uint output_index = selector * HAMMING_WEIGHT_BINS + bin;
    if (bin == 0u) {
        output[output_index] = solinas_zero();
        return;
    }

    SolinasFp128 sum = solinas_zero();
    for (uint outer = 0u; outer < params.outer_length; outer++) {
        uint input = (outer * HAMMING_WEIGHT_SELECTORS + selector)
            * HAMMING_WEIGHT_RETAINED_BINS
            + bin - 1u;
        sum = solinas_add(sum, partials[input]);
    }
    output[output_index] = sum;
}
