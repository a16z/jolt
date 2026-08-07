#define HAMMING_RETAINED_BINS 256u
#define HAMMING_RETAINED_DEFERRED_WORDS 5u

struct HammingWeightRetainedParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint selector_offset;
    uint selectors_in_tile;
    uint bins;
    uint2 reserved;
};

struct HammingRetainedLocalSum {
    SolinasFp128 low;
    uint overflow;
};

inline HammingRetainedLocalSum hamming_retained_local_zero()
{
    HammingRetainedLocalSum result;
    result.low = solinas_zero();
    result.overflow = 0u;
    return result;
}

inline void hamming_retained_local_add(
    thread HammingRetainedLocalSum& sum,
    SolinasFp128 value)
{
    ulong carry = 0ul;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong word = (ulong)sum.low.limb[limb]
            + (ulong)value.limb[limb]
            + carry;
        sum.low.limb[limb] = (uint)word;
        carry = word >> 32;
    }
    sum.overflow += (uint)carry;
}

inline void hamming_retained_atomic_add(
    threadgroup atomic_uint* sums,
    uint local_selector,
    uint hot,
    SolinasFp128 weight)
{
    uint field = local_selector * HAMMING_RETAINED_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, weight);
}

inline void hamming_retained_flush_local(
    threadgroup atomic_uint* sums,
    uint local_selector,
    uint hot,
    HammingRetainedLocalSum value)
{
    uint field = local_selector * HAMMING_RETAINED_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, value.low);
    if (value.overflow != 0u) {
        atomic_fetch_add_explicit(
            &sums[field * HAMMING_RETAINED_DEFERRED_WORDS + 4u],
            value.overflow,
            memory_order_relaxed);
    }
}

inline ulong hamming_retained_hot_index(
    uint selector,
    uint row,
    constant HammingWeightRetainedParams& params)
{
    return (ulong)selector * (ulong)params.rows + (ulong)row;
}

template <uint selector, uint local_selector>
inline void hamming_retained_add_selector(
    device const uchar* hot_rows,
    uint row,
    threadgroup atomic_uint* sums,
    constant HammingWeightRetainedParams& params,
    SolinasFp128 weight)
{
    uint hot = (uint)hot_rows[hamming_retained_hot_index(selector, row, params)];
    if (hot != 0u) {
        hamming_retained_atomic_add(sums, local_selector, hot, weight);
    }
}

template <uint selector_offset, uint selector_count, bool aggregate_carry>
inline void hamming_retained_tile_impl(
    device const uchar* hot_rows,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    device SolinasFp128* partials,
    constant HammingWeightRetainedParams& params,
    threadgroup atomic_uint* sums,
    uint x_out,
    uint tid,
    uint threads)
{
    uint fields = selector_count * HAMMING_RETAINED_BINS;
    uint counters = fields * HAMMING_RETAINED_DEFERRED_WORDS;
    for (uint counter = tid; counter < counters; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    HammingRetainedLocalSum negative_carry = hamming_retained_local_zero();
    HammingRetainedLocalSum positive_carry = hamming_retained_local_zero();
    uint row_base = x_out * params.e_in_length;
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row = row_base + x_in;
        SolinasFp128 weight = e_in[x_in];
        if (selector_count > 0u) {
            hamming_retained_add_selector<selector_offset, 0u>(
                hot_rows, row, sums, params, weight);
        }
        if (selector_count > 1u) {
            hamming_retained_add_selector<selector_offset + 1u, 1u>(
                hot_rows, row, sums, params, weight);
        }
        if (selector_count > 2u) {
            hamming_retained_add_selector<selector_offset + 2u, 2u>(
                hot_rows, row, sums, params, weight);
        }
        if (selector_count > 3u) {
            hamming_retained_add_selector<selector_offset + 3u, 3u>(
                hot_rows, row, sums, params, weight);
        }
        if (selector_count > 4u) {
            if (aggregate_carry) {
                uint hot = (uint)hot_rows[
                    hamming_retained_hot_index(selector_offset + 4u, row, params)];
                if (hot == 255u) {
                    hamming_retained_local_add(negative_carry, weight);
                } else if (hot == 1u) {
                    hamming_retained_local_add(positive_carry, weight);
                } else if (hot != 0u) {
                    hamming_retained_atomic_add(sums, 4u, hot, weight);
                }
            } else {
                hamming_retained_add_selector<selector_offset + 4u, 4u>(
                    hot_rows, row, sums, params, weight);
            }
        }
        if (selector_count > 5u) {
            hamming_retained_add_selector<selector_offset + 5u, 5u>(
                hot_rows, row, sums, params, weight);
        }
    }
    if (aggregate_carry) {
        hamming_retained_flush_local(sums, 4u, 255u, negative_carry);
        hamming_retained_flush_local(sums, 4u, 1u, positive_carry);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasFp128 outer = e_out[x_out];
    for (uint field = tid; field < fields; field += threads) {
        SolinasFp128 value = solinas_deferred_atomic_reduce_5(sums, field);
        partials[(ulong)x_out * (ulong)fields + (ulong)field] =
            solinas_mul_wide(outer, value);
    }
}

#define HAMMING_RETAINED_TILE_ENTRY(name, offset, count, aggregate_carry)         \
kernel void name(                                                               \
    device const uchar* hot_rows [[buffer(0)]],                                  \
    device const SolinasFp128* e_in [[buffer(1)]],                               \
    device const SolinasFp128* e_out [[buffer(2)]],                              \
    device SolinasFp128* partials [[buffer(3)]],                                 \
    constant HammingWeightRetainedParams& params [[buffer(4)]],                  \
    threadgroup atomic_uint* sums [[threadgroup(0)]],                            \
    uint x_out [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                    \
    uint threads [[threads_per_threadgroup]])                                    \
{                                                                               \
    hamming_retained_tile_impl<offset, count, aggregate_carry>(                  \
        hot_rows, e_in, e_out, partials, params, sums, x_out, tid, threads);     \
}

HAMMING_RETAINED_TILE_ENTRY(solinas_hamming_retained_tile_0, 0u, 6u, false)
HAMMING_RETAINED_TILE_ENTRY(solinas_hamming_retained_tile_1, 6u, 6u, false)
HAMMING_RETAINED_TILE_ENTRY(solinas_hamming_retained_tile_2, 12u, 6u, false)
HAMMING_RETAINED_TILE_ENTRY(solinas_hamming_retained_tile_3, 18u, 6u, false)
HAMMING_RETAINED_TILE_ENTRY(solinas_hamming_retained_tile_4, 24u, 5u, true)

#undef HAMMING_RETAINED_TILE_ENTRY

kernel void solinas_hamming_retained_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant HammingWeightRetainedParams& params [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_selector [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint bin = tid & (HAMMING_RETAINED_BINS - 1u);
    uint shard = tid / HAMMING_RETAINED_BINS;
    uint shards = threads / HAMMING_RETAINED_BINS;
    uint fields = params.selectors_in_tile * HAMMING_RETAINED_BINS;
    SolinasFp128 sum = solinas_zero();
    for (uint x_out = shard; x_out < params.e_out_length; x_out += shards) {
        ulong index = (ulong)x_out * (ulong)fields
            + (ulong)local_selector * (ulong)HAMMING_RETAINED_BINS
            + (ulong)bin;
        sum = solinas_add(sum, partials[index]);
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (shard == 0u) {
        for (uint other = 1u; other < shards; other++) {
            sum = solinas_add(
                sum,
                shared[other * HAMMING_RETAINED_BINS + bin]);
        }
        uint selector = params.selector_offset + local_selector;
        output[selector * HAMMING_RETAINED_BINS + bin] = sum;
    }
}
