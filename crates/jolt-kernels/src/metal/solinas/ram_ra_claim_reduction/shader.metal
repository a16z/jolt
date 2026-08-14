#define RAM_RA_CLAIM_TERMS 3u
#define RAM_RA_CLAIM_ADDRESS_DOMAIN 8192u
#define RAM_RA_CLAIM_NO_ACCESS 0xffffffffu
#define RAM_RA_CLAIM_THREADS 32u
#define RAM_RA_CLAIM_MAX_Q_PARTITIONS 16u
#define RAM_RA_CLAIM_Q_ACCESSED 0u
#define RAM_RA_CLAIM_Q_INVALID 1u
#define RAM_RA_CLAIM_GATHER_INVALID 2u
#define RAM_RA_CLAIM_UNSUPPORTED 3u

struct RamRaClaimParams {
    uint rows;
    uint address_limit;
    uint prefix_length;
    uint suffix_length;
    uint terms;
    uint no_access;
    uint q_partitions;
    uint threads;
};

inline bool ram_ra_claim_supported(
    constant RamRaClaimParams& params,
    uint3 threads_per_group)
{
    return params.rows != 0u
        && params.address_limit == RAM_RA_CLAIM_ADDRESS_DOMAIN
        && params.prefix_length >= RAM_RA_CLAIM_THREADS
        && (params.prefix_length % RAM_RA_CLAIM_THREADS) == 0u
        && params.suffix_length != 0u
        && (ulong)params.prefix_length * (ulong)params.suffix_length
            == (ulong)params.rows
        && params.terms == RAM_RA_CLAIM_TERMS
        && params.no_access == RAM_RA_CLAIM_NO_ACCESS
        && params.q_partitions != 0u
        && params.q_partitions <= RAM_RA_CLAIM_MAX_Q_PARTITIONS
        && (params.q_partitions & (params.q_partitions - 1u)) == 0u
        && (params.suffix_length % params.q_partitions) == 0u
        && params.threads == RAM_RA_CLAIM_THREADS
        && threads_per_group.x == RAM_RA_CLAIM_THREADS
        && threads_per_group.y == 1u
        && threads_per_group.z == 1u;
}

inline uint ram_ra_claim_simd_sum_u32(uint value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value += simd_shuffle_down(value, offset);
    }
    return value;
}

kernel void solinas_ram_ra_claim_build_q_partials_compact(
    device const uint* compact_entries [[buffer(0)]],
    device const uint* compact_offsets [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* eq_hi [[buffer(3)]],
    device SolinasFp128* q_partials [[buffer(4)]],
    device atomic_uint* counters [[buffer(5)]],
    constant RamRaClaimParams& params [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    if (!ram_ra_claim_supported(params, threads_per_group)) {
        if (group.x == 0u && lane == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_UNSUPPORTED],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    uint producer_threads = params.prefix_length * params.q_partitions;
    if (gid.x >= producer_threads) {
        return;
    }
    uint partition = gid.x / params.prefix_length;
    uint lo = gid.x - partition * params.prefix_length;
    ulong begin = (ulong)compact_offsets[lo];
    ulong end = (ulong)compact_offsets[lo + 1u];
    ulong count = end - begin;
    ulong segment_begin = begin
        + count * (ulong)partition / (ulong)params.q_partitions;
    ulong segment_end = begin
        + count * (ulong)(partition + 1u) / (ulong)params.q_partitions;

    SolinasFp128 sum0 = solinas_zero();
    SolinasFp128 sum1 = solinas_zero();
    SolinasFp128 sum2 = solinas_zero();
    uint lane_invalid = 0u;
    for (ulong index = segment_begin; index < segment_end; index++) {
        uint entry = compact_entries[index];
        uint address = entry & (RAM_RA_CLAIM_ADDRESS_DOMAIN - 1u);
        uint hi = entry >> 13u;
        if (hi >= params.suffix_length) {
            lane_invalid += 1u;
            continue;
        }
        SolinasFp128 h = eq_address[address];
        sum0 = solinas_add(sum0, solinas_mul_wide(h, eq_hi[hi]));
        sum1 = solinas_add(
            sum1,
            solinas_mul_wide(h, eq_hi[params.suffix_length + hi]));
        sum2 = solinas_add(
            sum2,
            solinas_mul_wide(h, eq_hi[2u * params.suffix_length + hi]));
    }

    uint partial_base = partition * params.prefix_length + lo;
    uint term_stride = params.q_partitions * params.prefix_length;
    q_partials[partial_base] = sum0;
    q_partials[term_stride + partial_base] = sum1;
    q_partials[2u * term_stride + partial_base] = sum2;
    uint group_accessed = ram_ra_claim_simd_sum_u32(
        (uint)(segment_end - segment_begin));
    uint group_invalid = ram_ra_claim_simd_sum_u32(lane_invalid);
    if (lane == 0u) {
        if (group_accessed != 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_Q_ACCESSED],
                group_accessed,
                memory_order_relaxed);
        }
        if (group_invalid != 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_Q_INVALID],
                group_invalid,
                memory_order_relaxed);
        }
    }
}

kernel void solinas_ram_ra_claim_reduce_q(
    device const SolinasFp128* q_partials [[buffer(0)]],
    device SolinasFp128* q [[buffer(1)]],
    device atomic_uint* counters [[buffer(2)]],
    constant RamRaClaimParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    if (!ram_ra_claim_supported(params, threads_per_group)) {
        if (group.x == 0u && lane == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_UNSUPPORTED],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    if (gid.x >= params.prefix_length) {
        return;
    }

    for (uint term = 0u; term < RAM_RA_CLAIM_TERMS; term++) {
        SolinasFp128 sum = solinas_zero();
        for (uint partition = 0u;
             partition < params.q_partitions;
             partition++) {
            uint partial = term * params.q_partitions + partition;
            sum = solinas_add(
                sum,
                q_partials[partial * params.prefix_length + gid.x]);
        }
        q[term * params.prefix_length + gid.x] = sum;
    }
}

kernel void solinas_ram_ra_claim_gather_h_compact(
    device const uint* compact_entries [[buffer(0)]],
    device const uint* compact_offsets [[buffer(1)]],
    device const SolinasFp128* eq_address [[buffer(2)]],
    device const SolinasFp128* eq_prefix [[buffer(3)]],
    device SolinasFp128* h_prime [[buffer(4)]],
    device atomic_uint* counters [[buffer(5)]],
    constant RamRaClaimParams& params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]])
{
    if (!ram_ra_claim_supported(params, threads_per_group)) {
        if (group.x == 0u && lane == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_UNSUPPORTED],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    uint hi = group.x;
    if (hi >= params.suffix_length) {
        return;
    }

    uint begin = compact_offsets[hi];
    uint end = compact_offsets[hi + 1u];
    SolinasFp128 sum = solinas_zero();
    uint lane_invalid = 0u;
    for (uint index = begin + tid; index < end; index += RAM_RA_CLAIM_THREADS) {
        uint entry = compact_entries[index];
        uint address = entry & (RAM_RA_CLAIM_ADDRESS_DOMAIN - 1u);
        uint lo = entry >> 13u;
        if (lo >= params.prefix_length) {
            lane_invalid += 1u;
            continue;
        }
        sum = solinas_add(
            sum,
            solinas_mul_wide(eq_address[address], eq_prefix[lo]));
    }

    sum = solinas_simd_sum_32(sum);
    uint group_invalid = ram_ra_claim_simd_sum_u32(lane_invalid);
    if (lane == 0u) {
        h_prime[hi] = sum;
        uint accessed = end - begin;
        if (accessed != 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_Q_ACCESSED],
                accessed,
                memory_order_relaxed);
        }
        if (group_invalid != 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RA_CLAIM_GATHER_INVALID],
                group_invalid,
                memory_order_relaxed);
        }
    }
}
