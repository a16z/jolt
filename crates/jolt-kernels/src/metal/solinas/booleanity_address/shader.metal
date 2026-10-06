#define BOOLEANITY_ADDRESS_BINS 256u

struct BooleanityAddressParams {
    uint rows;
    uint polys;
    uint k;
    uint e_in_length;
    uint e_out_length;
    uint selector_offset;
    uint selectors_in_tile;
    uint chunk_bits;
    ulong inc_bias;
};

inline void booleanity_address_add(
    threadgroup atomic_uint* sums,
    uint local,
    uint hot,
    SolinasFp128 weight)
{
    uint field = local * BOOLEANITY_ADDRESS_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, weight);
}

inline void booleanity_address_inc(
    ulong magnitude,
    ulong packed_pc_and_flags,
    ulong bias,
    thread ulong& biased,
    thread int& carry)
{
    bool negative = (packed_pc_and_flags >> 63) != 0ul;
    if (negative) {
        biased = bias - magnitude;
        carry = magnitude > bias ? -1 : 0;
    } else {
        biased = bias + magnitude;
        carry = biased < bias ? 1 : 0;
    }
}

inline uint booleanity_address_inc_bin(ulong biased, uint shift) {
    uint standard = (uint)(biased >> shift) & (BOOLEANITY_ADDRESS_BINS - 1u);
    return (standard + BOOLEANITY_ADDRESS_BINS / 2u) & (BOOLEANITY_ADDRESS_BINS - 1u);
}

inline uint booleanity_address_byte_bin(ulong word, uint shift) {
    return (uint)(word >> shift) & (BOOLEANITY_ADDRESS_BINS - 1u);
}

inline bool booleanity_address_offset_bin(ulong plus_one, uint shift, thread uint& hot) {
    plus_one &= 0x00ffFFFFFFFFFFFFul;
    hot = (uint)((plus_one - 1ul) >> shift) & (BOOLEANITY_ADDRESS_BINS - 1u);
    return plus_one != 0ul;
}

// Production selector layout: 0..7 lookup_hi bytes, 8..15 lookup_lo bytes,
// 16..17 bytecode, then two (three with `three_ram`) RAM bytes, eight
// increment bytes and the increment carry. Keep in sync with
// `production_selector_schedule` (booleanity_address/mod.rs).
template <uint selector, bool three_ram>
inline bool booleanity_address_production_hot(
    BooleanityRow row,
    constant BooleanityAddressParams& params,
    thread uint& hot)
{
    const uint ram_end = three_ram ? 21u : 20u;
    if (selector < 8u) {
        hot = booleanity_address_byte_bin(row.lookup_hi, 8u * (7u - selector));
        return true;
    }
    if (selector < 16u) {
        hot = booleanity_address_byte_bin(row.lookup_lo, 8u * (15u - selector));
        return true;
    }
    if (selector < 18u) {
        return booleanity_address_offset_bin(
            row.packed_pc_and_flags, 8u * (17u - selector), hot);
    }
    if (selector < ram_end) {
        return booleanity_address_offset_bin(
            row.ram_address_plus_one, 8u * (ram_end - 1u - selector), hot);
    }
    ulong biased;
    int carry;
    booleanity_address_inc(
        row.fused_inc_magnitude,
        row.packed_pc_and_flags,
        params.inc_bias,
        biased,
        carry);
    hot = selector < ram_end + 8u
        ? booleanity_address_inc_bin(biased, 8u * (selector - ram_end))
        : (uint)carry & (BOOLEANITY_ADDRESS_BINS - 1u);
    return true;
}

// The bin of a zero source value, which most rows hit (padding rows, short
// lookups and pcs, zero increments).
inline uint booleanity_address_dominant_bin(
    uint selector,
    bool three_ram,
    constant BooleanityAddressParams& params)
{
    uint ram_end = three_ram ? 21u : 20u;
    return selector >= ram_end && selector < ram_end + 8u
        ? booleanity_address_inc_bin(params.inc_bias, 8u * (selector - ram_end))
        : 0u;
}

template <uint selector, bool three_ram>
inline void booleanity_address_add_production(
    BooleanityRow row,
    threadgroup atomic_uint* sums,
    uint local,
    constant BooleanityAddressParams& params,
    SolinasFp128 weight,
    thread SolinasLazySum& dominant)
{
    uint hot;
    if (!booleanity_address_production_hot<selector, three_ram>(row, params, hot)) {
        return;
    }
    if (hot == booleanity_address_dominant_bin(selector, three_ram, params)) {
        solinas_lazy_add(dominant, weight);
    } else {
        booleanity_address_add(sums, local, hot, weight);
    }
}

template <
    uint production_offset,
    uint production_count,
    bool aggregate_inc,
    bool three_ram>
inline void booleanity_address_tile_impl(
    device const ulong* rows,
    device const BooleanitySelector* selectors,
    device const SolinasFp128* e_in,
    device const SolinasFp128* e_out,
    device SolinasFp128* partials,
    constant BooleanityAddressParams& params,
    threadgroup atomic_uint* sums,
    uint x_out,
    uint tid,
    uint lane,
    uint threads)
{
    uint fields = params.selectors_in_tile * BOOLEANITY_ADDRESS_BINS;
    uint counters = fields * SOLINAS_DEFERRED_SUM_WORDS;
    for (uint counter = tid; counter < counters; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasLazySum common_inc_sum = solinas_lazy_zero();
    SolinasLazySum negative_carry_sum = solinas_lazy_zero();
    SolinasLazySum zero_carry_sum = solinas_lazy_zero();
    SolinasLazySum positive_carry_sum = solinas_lazy_zero();
    SolinasLazySum dominant[6];
    for (uint local = 0u; local < 6u; local++) {
        dominant[local] = solinas_lazy_zero();
    }
    uint row_base = x_out * params.e_in_length;
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = row_base + x_in;
        SolinasFp128 weight = e_in[x_in];
        BooleanityRow row = booleanity_row_load(rows, params.rows, row_index);
        if (production_count == 0u) {
            for (uint local = 0u; local < params.selectors_in_tile; local++) {
                uint hot = params.k;
                BooleanitySelector selector = selectors[params.selector_offset + local];
                if (booleanity_hot_index(
                        row, selector, params.chunk_bits, params.inc_bias, hot)) {
                    booleanity_address_add(sums, local, hot, weight);
                }
            }
        } else if (aggregate_inc) {
            ulong biased;
            int carry;
            booleanity_address_inc(
                row.fused_inc_magnitude,
                row.packed_pc_and_flags,
                params.inc_bias,
                biased,
                carry);
            uint hot_24 = booleanity_address_inc_bin(biased, 24u);
            uint hot_32 = booleanity_address_inc_bin(biased, 32u);
            uint hot_40 = booleanity_address_inc_bin(biased, 40u);
            uint hot_48 = booleanity_address_inc_bin(biased, 48u);
            if ((!three_ram || hot_24 == 0u)
                && hot_32 == 0u
                && hot_40 == 0u
                && hot_48 == 0u) {
                solinas_lazy_add(common_inc_sum, weight);
            } else {
                if (three_ram) {
                    booleanity_address_add(sums, 0u, hot_24, weight);
                    booleanity_address_add(sums, 1u, hot_32, weight);
                    booleanity_address_add(sums, 2u, hot_40, weight);
                    booleanity_address_add(sums, 3u, hot_48, weight);
                } else {
                    booleanity_address_add(sums, 0u, hot_32, weight);
                    booleanity_address_add(sums, 1u, hot_40, weight);
                    booleanity_address_add(sums, 2u, hot_48, weight);
                }
            }
            booleanity_address_add(
                sums, three_ram ? 4u : 3u, booleanity_address_inc_bin(biased, 56u), weight);
            if (carry < 0) {
                solinas_lazy_add(negative_carry_sum, weight);
            } else if (carry > 0) {
                solinas_lazy_add(positive_carry_sum, weight);
            } else {
                solinas_lazy_add(zero_carry_sum, weight);
            }
        } else {
            if (production_count > 0u) {
                booleanity_address_add_production<production_offset, three_ram>(
                    row, sums, 0u, params, weight, dominant[0]);
            }
            if (production_count > 1u) {
                booleanity_address_add_production<production_offset + 1u, three_ram>(
                    row, sums, 1u, params, weight, dominant[1]);
            }
            if (production_count > 2u) {
                booleanity_address_add_production<production_offset + 2u, three_ram>(
                    row, sums, 2u, params, weight, dominant[2]);
            }
            if (production_count > 3u) {
                booleanity_address_add_production<production_offset + 3u, three_ram>(
                    row, sums, 3u, params, weight, dominant[3]);
            }
            if (production_count > 4u) {
                booleanity_address_add_production<production_offset + 4u, three_ram>(
                    row, sums, 4u, params, weight, dominant[4]);
            }
            if (production_count > 5u) {
                booleanity_address_add_production<production_offset + 5u, three_ram>(
                    row, sums, 5u, params, weight, dominant[5]);
            }
        }
    }
    if (aggregate_inc) {
        uint carry_field = (three_ram ? 5u : 4u) * BOOLEANITY_ADDRESS_BINS;
        for (uint local = 0u; local < (three_ram ? 4u : 3u); local++) {
            solinas_deferred_atomic_flush_simd(
                sums, local * BOOLEANITY_ADDRESS_BINS, common_inc_sum, lane);
        }
        solinas_deferred_atomic_flush_simd(
            sums, carry_field + BOOLEANITY_ADDRESS_BINS - 1u, negative_carry_sum, lane);
        solinas_deferred_atomic_flush_simd(sums, carry_field, zero_carry_sum, lane);
        solinas_deferred_atomic_flush_simd(sums, carry_field + 1u, positive_carry_sum, lane);
    } else {
        for (uint local = 0u; local < production_count; local++) {
            uint bin = booleanity_address_dominant_bin(production_offset + local, three_ram, params);
            solinas_deferred_atomic_flush_simd(
                sums, local * BOOLEANITY_ADDRESS_BINS + bin, dominant[local], lane);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasFp128 outer = e_out[x_out];
    uint output_base = x_out * fields;
    for (uint field = tid; field < fields; field += threads) {
        SolinasFp128 value = solinas_deferred_atomic_reduce_5(sums, field);
        partials[output_base + field] = solinas_mul_wide(outer, value);
    }
}

#define BOOLEANITY_ADDRESS_TILE_ENTRY(name, offset, count, aggregate_inc, three_ram) \
kernel void name(                                                                 \
    device const ulong* rows [[buffer(0)]],                                       \
    device const BooleanitySelector* selectors [[buffer(1)]],                    \
    device const SolinasFp128* e_in [[buffer(2)]],                                \
    device const SolinasFp128* e_out [[buffer(3)]],                               \
    device SolinasFp128* partials [[buffer(4)]],                                  \
    constant BooleanityAddressParams& params [[buffer(5)]],                       \
    threadgroup atomic_uint* sums [[threadgroup(0)]],                             \
    uint x_out [[threadgroup_position_in_grid]],                                  \
    uint tid [[thread_index_in_threadgroup]],                                     \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint threads [[threads_per_threadgroup]])                                     \
{                                                                                 \
    booleanity_address_tile_impl<offset, count, aggregate_inc, three_ram>(        \
        rows, selectors, e_in, e_out, partials, params, sums, x_out, tid, lane,   \
        threads);                                                                 \
}

BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile, 0u, 0u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_0, 0u, 6u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_1, 6u, 6u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_2, 12u, 6u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3, 18u, 6u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_4, 24u, 5u, true, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_ram3_3, 18u, 6u, false, true)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_ram3_4, 24u, 6u, true, true)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_0, 0u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_1, 3u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_2, 6u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_3, 9u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_4, 12u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_5, 15u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_6, 18u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_7, 21u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_8, 24u, 3u, false, false)
BOOLEANITY_ADDRESS_TILE_ENTRY(solinas_booleanity_address_tile_3_9, 27u, 2u, false, false)

#undef BOOLEANITY_ADDRESS_TILE_ENTRY

kernel void solinas_booleanity_address_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant BooleanityAddressParams& params [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_selector [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint bin = tid & (BOOLEANITY_ADDRESS_BINS - 1u);
    uint shard = tid / BOOLEANITY_ADDRESS_BINS;
    uint shards = threads / BOOLEANITY_ADDRESS_BINS;
    uint fields = params.selectors_in_tile * BOOLEANITY_ADDRESS_BINS;
    SolinasFp128 sum = solinas_zero();
    for (uint x_out = shard; x_out < params.e_out_length; x_out += shards) {
        uint index = x_out * fields
            + local_selector * BOOLEANITY_ADDRESS_BINS
            + bin;
        sum = solinas_add(sum, partials[index]);
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (shard == 0u) {
        for (uint other = 1u; other < shards; other++) {
            sum = solinas_add(
                sum,
                shared[other * BOOLEANITY_ADDRESS_BINS + bin]);
        }
        uint selector = params.selector_offset + local_selector;
        output[selector * BOOLEANITY_ADDRESS_BINS + bin] = sum;
    }
}
