#define BOOLEANITY_ADDRESS_BINS 256u
#define BOOLEANITY_ADDRESS_ACCUMULATOR_WORDS 5u

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

struct BooleanityAddressLocalSum {
    SolinasFp128 low;
    uint overflow;
};

inline BooleanityAddressLocalSum booleanity_address_local_zero() {
    BooleanityAddressLocalSum result;
    result.low = solinas_zero();
    result.overflow = 0u;
    return result;
}

inline void booleanity_address_local_add(
    thread BooleanityAddressLocalSum& sum,
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

inline void booleanity_address_flush_local(
    threadgroup atomic_uint* sums,
    uint local,
    uint hot,
    BooleanityAddressLocalSum value)
{
    uint field = local * BOOLEANITY_ADDRESS_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, value.low);
    if (value.overflow != 0u) {
        atomic_fetch_add_explicit(
            &sums[field * BOOLEANITY_ADDRESS_ACCUMULATOR_WORDS + 4u],
            value.overflow,
            memory_order_relaxed);
    }
}

inline void booleanity_address_add(
    threadgroup atomic_uint* sums,
    uint local,
    uint hot,
    SolinasFp128 weight)
{
    uint field = local * BOOLEANITY_ADDRESS_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, weight);
}

inline void booleanity_address_add_lookup_word(
    threadgroup atomic_uint* sums,
    ulong word,
    uint local,
    uint word_shift,
    SolinasFp128 weight)
{
    booleanity_address_add(
        sums,
        local,
        (uint)(word >> word_shift) & (BOOLEANITY_ADDRESS_BINS - 1u),
        weight);
}

inline void booleanity_address_add_bytecode(
    threadgroup atomic_uint* sums,
    ulong packed_pc_and_flags,
    uint local,
    uint shift,
    SolinasFp128 weight)
{
    ulong plus_one = packed_pc_and_flags & 0x00ffFFFFFFFFFFFFul;
    if (plus_one != 0ul) {
        booleanity_address_add(
            sums,
            local,
            (uint)((plus_one - 1ul) >> shift)
                & (BOOLEANITY_ADDRESS_BINS - 1u),
            weight);
    }
}

inline void booleanity_address_add_ram(
    threadgroup atomic_uint* sums,
    ulong ram_address_plus_one,
    uint local,
    uint shift,
    SolinasFp128 weight)
{
    ulong plus_one = ram_address_plus_one & 0x00ffFFFFFFFFFFFFul;
    if (plus_one != 0ul) {
        booleanity_address_add(
            sums,
            local,
            (uint)((plus_one - 1ul) >> shift)
                & (BOOLEANITY_ADDRESS_BINS - 1u),
            weight);
    }
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

inline void booleanity_address_add_inc(
    threadgroup atomic_uint* sums,
    ulong biased,
    uint local,
    uint shift,
    SolinasFp128 weight)
{
    uint standard = (uint)(biased >> shift) & (BOOLEANITY_ADDRESS_BINS - 1u);
    booleanity_address_add(
        sums,
        local,
        (standard + BOOLEANITY_ADDRESS_BINS / 2u)
            & (BOOLEANITY_ADDRESS_BINS - 1u),
        weight);
}

template <uint selector>
inline void booleanity_address_add_production_selector(
    BooleanityRow row,
    threadgroup atomic_uint* sums,
    uint local,
    constant BooleanityAddressParams& params,
    SolinasFp128 weight)
{
    if (selector < 8u) {
        booleanity_address_add_lookup_word(
            sums,
            row.lookup_hi,
            local,
            8u * (7u - selector),
            weight);
    } else if (selector < 16u) {
        booleanity_address_add_lookup_word(
            sums,
            row.lookup_lo,
            local,
            8u * (15u - selector),
            weight);
    } else if (selector < 18u) {
        booleanity_address_add_bytecode(
            sums,
            row.packed_pc_and_flags,
            local,
            8u * (17u - selector),
            weight);
    } else if (selector < 20u) {
        booleanity_address_add_ram(
            sums,
            row.ram_address_plus_one,
            local,
            8u * (19u - selector),
            weight);
    } else {
        ulong biased;
        int carry;
        booleanity_address_inc(
            row.fused_inc_magnitude,
            row.packed_pc_and_flags,
            params.inc_bias,
            biased,
            carry);
        if (selector < 28u) {
            booleanity_address_add_inc(
                sums, biased, local, 8u * (selector - 20u), weight);
        } else {
            booleanity_address_add(
                sums,
                local,
                (uint)carry & (BOOLEANITY_ADDRESS_BINS - 1u),
                weight);
        }
    }
}

template <uint selector>
inline void booleanity_address_add_three_ram_production_selector(
    BooleanityRow row,
    threadgroup atomic_uint* sums,
    uint local,
    constant BooleanityAddressParams& params,
    SolinasFp128 weight)
{
    if (selector < 8u) {
        booleanity_address_add_lookup_word(
            sums,
            row.lookup_hi,
            local,
            8u * (7u - selector),
            weight);
    } else if (selector < 16u) {
        booleanity_address_add_lookup_word(
            sums,
            row.lookup_lo,
            local,
            8u * (15u - selector),
            weight);
    } else if (selector < 18u) {
        booleanity_address_add_bytecode(
            sums,
            row.packed_pc_and_flags,
            local,
            8u * (17u - selector),
            weight);
    } else if (selector < 21u) {
        booleanity_address_add_ram(
            sums,
            row.ram_address_plus_one,
            local,
            8u * (20u - selector),
            weight);
    } else {
        ulong biased;
        int carry;
        booleanity_address_inc(
            row.fused_inc_magnitude,
            row.packed_pc_and_flags,
            params.inc_bias,
            biased,
            carry);
        if (selector < 29u) {
            booleanity_address_add_inc(
                sums, biased, local, 8u * (selector - 21u), weight);
        } else {
            booleanity_address_add(
                sums,
                local,
                (uint)carry & (BOOLEANITY_ADDRESS_BINS - 1u),
                weight);
        }
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
    uint threads)
{
    uint fields = params.selectors_in_tile * BOOLEANITY_ADDRESS_BINS;
    uint counters = fields * BOOLEANITY_ADDRESS_ACCUMULATOR_WORDS;
    for (uint counter = tid; counter < counters; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    BooleanityAddressLocalSum common_inc_sum = booleanity_address_local_zero();
    BooleanityAddressLocalSum negative_carry_sum = booleanity_address_local_zero();
    BooleanityAddressLocalSum zero_carry_sum = booleanity_address_local_zero();
    BooleanityAddressLocalSum positive_carry_sum = booleanity_address_local_zero();
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
            uint hot_24 = ((uint)(biased >> 24) + BOOLEANITY_ADDRESS_BINS / 2u)
                & (BOOLEANITY_ADDRESS_BINS - 1u);
            uint hot_32 = ((uint)(biased >> 32) + BOOLEANITY_ADDRESS_BINS / 2u)
                & (BOOLEANITY_ADDRESS_BINS - 1u);
            uint hot_40 = ((uint)(biased >> 40) + BOOLEANITY_ADDRESS_BINS / 2u)
                & (BOOLEANITY_ADDRESS_BINS - 1u);
            uint hot_48 = ((uint)(biased >> 48) + BOOLEANITY_ADDRESS_BINS / 2u)
                & (BOOLEANITY_ADDRESS_BINS - 1u);
            if ((!three_ram || hot_24 == 0u)
                && hot_32 == 0u
                && hot_40 == 0u
                && hot_48 == 0u) {
                booleanity_address_local_add(common_inc_sum, weight);
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
            booleanity_address_add_inc(
                sums, biased, three_ram ? 4u : 3u, 56u, weight);
            if (carry < 0) {
                booleanity_address_local_add(negative_carry_sum, weight);
            } else if (carry > 0) {
                booleanity_address_local_add(positive_carry_sum, weight);
            } else {
                booleanity_address_local_add(zero_carry_sum, weight);
            }
        } else {
            if (production_count > 0u) {
                if (three_ram) {
                    booleanity_address_add_three_ram_production_selector<production_offset>(
                        row, sums, 0u, params, weight);
                } else {
                    booleanity_address_add_production_selector<production_offset>(
                        row, sums, 0u, params, weight);
                }
            }
            if (production_count > 1u) {
                if (three_ram) {
                    booleanity_address_add_three_ram_production_selector<production_offset + 1u>(
                        row, sums, 1u, params, weight);
                } else {
                    booleanity_address_add_production_selector<production_offset + 1u>(
                        row, sums, 1u, params, weight);
                }
            }
            if (production_count > 2u) {
                if (three_ram) {
                    booleanity_address_add_three_ram_production_selector<production_offset + 2u>(
                        row, sums, 2u, params, weight);
                } else {
                    booleanity_address_add_production_selector<production_offset + 2u>(
                        row, sums, 2u, params, weight);
                }
            }
            if (production_count > 3u) {
                if (three_ram) {
                    booleanity_address_add_three_ram_production_selector<production_offset + 3u>(
                        row, sums, 3u, params, weight);
                } else {
                    booleanity_address_add_production_selector<production_offset + 3u>(
                        row, sums, 3u, params, weight);
                }
            }
            if (production_count > 4u) {
                if (three_ram) {
                    booleanity_address_add_three_ram_production_selector<production_offset + 4u>(
                        row, sums, 4u, params, weight);
                } else {
                    booleanity_address_add_production_selector<production_offset + 4u>(
                        row, sums, 4u, params, weight);
                }
            }
            if (production_count > 5u) {
                if (three_ram) {
                    booleanity_address_add_three_ram_production_selector<production_offset + 5u>(
                        row, sums, 5u, params, weight);
                } else {
                    booleanity_address_add_production_selector<production_offset + 5u>(
                        row, sums, 5u, params, weight);
                }
            }
        }
    }
    if (aggregate_inc) {
        booleanity_address_flush_local(sums, 0u, 0u, common_inc_sum);
        booleanity_address_flush_local(sums, 1u, 0u, common_inc_sum);
        booleanity_address_flush_local(sums, 2u, 0u, common_inc_sum);
        if (three_ram) {
            booleanity_address_flush_local(sums, 3u, 0u, common_inc_sum);
        }
        uint carry_local = three_ram ? 5u : 4u;
        booleanity_address_flush_local(
            sums, carry_local, BOOLEANITY_ADDRESS_BINS - 1u, negative_carry_sum);
        booleanity_address_flush_local(sums, carry_local, 0u, zero_carry_sum);
        booleanity_address_flush_local(sums, carry_local, 1u, positive_carry_sum);
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
    uint threads [[threads_per_threadgroup]])                                     \
{                                                                                 \
    booleanity_address_tile_impl<offset, count, aggregate_inc, three_ram>(        \
        rows, selectors, e_in, e_out, partials, params, sums, x_out, tid, threads); \
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
