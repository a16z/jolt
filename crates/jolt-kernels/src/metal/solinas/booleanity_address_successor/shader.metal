#define BOOLEANITY_ADDRESS_SUCCESSOR_BINS 256u
#define BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS 29u
#define BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_SELECTORS 6u
#define BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_BASE 6u
#define BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_SELECTORS 23u
#define BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_PLANES 29u
#define BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES 4u
#define BOOLEANITY_ADDRESS_SUCCESSOR_DEFERRED_WORDS 5u
#define BOOLEANITY_ADDRESS_SUCCESSOR_BYTECODE_PRESENT 1u
#define BOOLEANITY_ADDRESS_SUCCESSOR_RAM_PRESENT 2u

struct BooleanityAddressSuccessorParams {
    uint rows;
    uint e_in_length;
    uint e_out_length;
    uint selector_count;
    ulong inc_bias;
    uint packed_selector_base;
    uint packed_planes;
    uint remaining_tiles;
    uint reserved;
};

struct BooleanityAddressSuccessorLocalSum {
    SolinasFp128 low;
    uint overflow;
};

inline BooleanityAddressSuccessorLocalSum
booleanity_address_successor_local_zero()
{
    BooleanityAddressSuccessorLocalSum result;
    result.low = solinas_zero();
    result.overflow = 0u;
    return result;
}

inline void booleanity_address_successor_local_add(
    thread BooleanityAddressSuccessorLocalSum& sum,
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

inline void booleanity_address_successor_atomic_add(
    threadgroup atomic_uint* sums,
    uint local_selector,
    uint hot,
    SolinasFp128 weight)
{
    uint field = local_selector * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, weight);
}

inline void booleanity_address_successor_flush_local(
    threadgroup atomic_uint* sums,
    uint local_selector,
    uint hot,
    BooleanityAddressSuccessorLocalSum value)
{
    uint field = local_selector * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + hot;
    solinas_deferred_atomic_add_5(sums, field, value.low);
    if (value.overflow != 0u) {
        atomic_fetch_add_explicit(
            &sums[field * BOOLEANITY_ADDRESS_SUCCESSOR_DEFERRED_WORDS + 4u],
            value.overflow,
            memory_order_relaxed);
    }
}

inline ulong booleanity_address_successor_packed_index(
    uint plane,
    uint row,
    constant BooleanityAddressSuccessorParams& params)
{
    return (ulong)plane * (ulong)params.rows + (ulong)row;
}

inline void booleanity_address_successor_store_hot(
    device uchar* hot_rows,
    uint selector,
    uint row,
    uint hot,
    constant BooleanityAddressSuccessorParams& params)
{
    hot_rows[booleanity_address_successor_packed_index(selector, row, params)] =
        (uchar)hot;
}

inline uint booleanity_address_successor_load_hot(
    device const uchar* hot_rows,
    uint selector,
    uint row,
    constant BooleanityAddressSuccessorParams& params)
{
    return (uint)hot_rows[
        booleanity_address_successor_packed_index(selector, row, params)];
}

inline uint booleanity_address_successor_load_flags(
    device const uchar* validity,
    uint row)
{
    return (uint)validity[row];
}

inline void booleanity_address_successor_increment(
    BooleanityRow row,
    constant BooleanityAddressSuccessorParams& params,
    thread ulong& biased,
    thread int& carry)
{
    if ((row.packed_pc_and_flags >> 63) != 0ul) {
        biased = params.inc_bias - row.fused_inc_magnitude;
        carry = row.fused_inc_magnitude > params.inc_bias ? -1 : 0;
    } else {
        biased = params.inc_bias + row.fused_inc_magnitude;
        carry = biased < params.inc_bias ? 1 : 0;
    }
}

inline uint booleanity_address_successor_recentered_byte(
    ulong biased,
    uint shift)
{
    return ((uint)(biased >> shift) + 128u)
        & (BOOLEANITY_ADDRESS_SUCCESSOR_BINS - 1u);
}

inline void booleanity_address_successor_pack_row(
    device uchar* hot_rows,
    device uchar* validity,
    uint row_index,
    BooleanityRow row,
    constant BooleanityAddressSuccessorParams& params)
{
    booleanity_address_successor_store_hot(
        hot_rows, 0u, row_index, (uint)(row.lookup_hi >> 56u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 1u, row_index, (uint)(row.lookup_hi >> 48u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 2u, row_index, (uint)(row.lookup_hi >> 40u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 3u, row_index, (uint)(row.lookup_hi >> 32u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 4u, row_index, (uint)(row.lookup_hi >> 24u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 5u, row_index, (uint)(row.lookup_hi >> 16u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 6u, row_index, (uint)(row.lookup_hi >> 8u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 7u, row_index, (uint)row.lookup_hi & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 8u, row_index, (uint)(row.lookup_lo >> 56u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 9u, row_index, (uint)(row.lookup_lo >> 48u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 10u, row_index, (uint)(row.lookup_lo >> 40u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 11u, row_index, (uint)(row.lookup_lo >> 32u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 12u, row_index, (uint)(row.lookup_lo >> 24u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 13u, row_index, (uint)(row.lookup_lo >> 16u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 14u, row_index, (uint)(row.lookup_lo >> 8u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 15u, row_index, (uint)row.lookup_lo & 255u, params);

    uint flags = 0u;
    ulong pc_plus_one = row.packed_pc_and_flags & 0x00ffFFFFFFFFFFFFul;
    ulong pc = 0ul;
    if (pc_plus_one != 0ul) {
        flags |= BOOLEANITY_ADDRESS_SUCCESSOR_BYTECODE_PRESENT;
        pc = pc_plus_one - 1ul;
    }
    booleanity_address_successor_store_hot(
        hot_rows, 16u, row_index, (uint)(pc >> 8u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 17u, row_index, (uint)pc & 255u, params);

    ulong ram = 0ul;
    if (row.ram_address_plus_one != 0ul) {
        flags |= BOOLEANITY_ADDRESS_SUCCESSOR_RAM_PRESENT;
        ram = row.ram_address_plus_one - 1ul;
    }
    booleanity_address_successor_store_hot(
        hot_rows, 18u, row_index, (uint)(ram >> 8u) & 255u, params);
    booleanity_address_successor_store_hot(
        hot_rows, 19u, row_index, (uint)ram & 255u, params);

    ulong biased;
    int carry;
    booleanity_address_successor_increment(row, params, biased, carry);
    booleanity_address_successor_store_hot(
        hot_rows,
        20u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 0u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        21u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 8u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        22u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 16u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        23u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 24u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        24u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 32u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        25u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 40u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        26u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 48u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows,
        27u,
        row_index,
        booleanity_address_successor_recentered_byte(biased, 56u),
        params);
    booleanity_address_successor_store_hot(
        hot_rows, 28u, row_index, (uint)carry & 255u, params);
    validity[row_index] = (uchar)flags;
}

inline void booleanity_address_successor_clear(
    threadgroup atomic_uint* sums,
    uint fields,
    uint tid,
    uint threads)
{
    uint counters = fields * BOOLEANITY_ADDRESS_SUCCESSOR_DEFERRED_WORDS;
    for (uint counter = tid; counter < counters; counter += threads) {
        atomic_store_explicit(&sums[counter], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline ulong booleanity_address_successor_partial_index(
    uint x_out,
    uint selector,
    uint bin)
{
    return ((ulong)x_out * (ulong)BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS
            + (ulong)selector)
        * (ulong)BOOLEANITY_ADDRESS_SUCCESSOR_BINS
        + (ulong)bin;
}

kernel void solinas_booleanity_address_successor_pack_and_first(
    device const ulong* rows [[buffer(0)]],
    device const SolinasFp128* e_in [[buffer(1)]],
    device const SolinasFp128* e_out [[buffer(2)]],
    device uchar* hot_rows [[buffer(3)]],
    device uchar* validity [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant BooleanityAddressSuccessorParams& params [[buffer(6)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint x_out [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    booleanity_address_successor_clear(
        sums,
        BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_SELECTORS
            * BOOLEANITY_ADDRESS_SUCCESSOR_BINS,
        tid,
        threads);

    uint row_base = x_out * params.e_in_length;
    for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
        uint row_index = row_base + x_in;
        BooleanityRow row = booleanity_row_load(rows, params.rows, row_index);
        SolinasFp128 weight = e_in[x_in];
        for (uint local = 0u;
             local < BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_SELECTORS;
             local++) {
            uint hot = (uint)(row.lookup_hi >> (8u * (7u - local))) & 255u;
            booleanity_address_successor_atomic_add(
                sums, local, hot, weight);
        }
        booleanity_address_successor_pack_row(
            hot_rows, validity, row_index, row, params);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasFp128 outer = e_out[x_out];
    uint fields = BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_SELECTORS
        * BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
    for (uint field = tid; field < fields; field += threads) {
        uint local = field / BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
        uint bin = field & (BOOLEANITY_ADDRESS_SUCCESSOR_BINS - 1u);
        SolinasFp128 value = solinas_deferred_atomic_reduce_5(sums, field);
        partials[booleanity_address_successor_partial_index(x_out, local, bin)] =
            solinas_mul_wide(outer, value);
    }
}

kernel void solinas_booleanity_address_successor_packed_tiles(
    device const uchar* hot_rows [[buffer(0)]],
    device const uchar* validity [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant BooleanityAddressSuccessorParams& params [[buffer(5)]],
    threadgroup atomic_uint* sums [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint tile = group % BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES;
    uint x_out = group / BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES;
    uint selector_base = BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_BASE + 6u * tile;
    uint selectors_in_tile = tile == 3u ? 5u : 6u;
    booleanity_address_successor_clear(
        sums,
        selectors_in_tile * BOOLEANITY_ADDRESS_SUCCESSOR_BINS,
        tid,
        threads);

    uint row_base = x_out * params.e_in_length;
    if (tile == 3u) {
        BooleanityAddressSuccessorLocalSum common_inc_sum =
            booleanity_address_successor_local_zero();
        BooleanityAddressSuccessorLocalSum negative_carry_sum =
            booleanity_address_successor_local_zero();
        BooleanityAddressSuccessorLocalSum zero_carry_sum =
            booleanity_address_successor_local_zero();
        BooleanityAddressSuccessorLocalSum positive_carry_sum =
            booleanity_address_successor_local_zero();
        for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
            uint row_index = row_base + x_in;
            SolinasFp128 weight = e_in[x_in];
            uint hot_32 = booleanity_address_successor_load_hot(
                hot_rows, 24u, row_index, params);
            uint hot_40 = booleanity_address_successor_load_hot(
                hot_rows, 25u, row_index, params);
            uint hot_48 = booleanity_address_successor_load_hot(
                hot_rows, 26u, row_index, params);
            if (hot_32 == 0u && hot_40 == 0u && hot_48 == 0u) {
                booleanity_address_successor_local_add(common_inc_sum, weight);
            } else {
                booleanity_address_successor_atomic_add(
                    sums, 0u, hot_32, weight);
                booleanity_address_successor_atomic_add(
                    sums, 1u, hot_40, weight);
                booleanity_address_successor_atomic_add(
                    sums, 2u, hot_48, weight);
            }
            uint hot_56 = booleanity_address_successor_load_hot(
                hot_rows, 27u, row_index, params);
            booleanity_address_successor_atomic_add(
                sums, 3u, hot_56, weight);
            uint carry = booleanity_address_successor_load_hot(
                hot_rows, 28u, row_index, params);
            if (carry == 255u) {
                booleanity_address_successor_local_add(negative_carry_sum, weight);
            } else if (carry == 1u) {
                booleanity_address_successor_local_add(positive_carry_sum, weight);
            } else {
                booleanity_address_successor_local_add(zero_carry_sum, weight);
            }
        }
        booleanity_address_successor_flush_local(
            sums, 0u, 0u, common_inc_sum);
        booleanity_address_successor_flush_local(
            sums, 1u, 0u, common_inc_sum);
        booleanity_address_successor_flush_local(
            sums, 2u, 0u, common_inc_sum);
        booleanity_address_successor_flush_local(
            sums, 4u, 255u, negative_carry_sum);
        booleanity_address_successor_flush_local(
            sums, 4u, 0u, zero_carry_sum);
        booleanity_address_successor_flush_local(
            sums, 4u, 1u, positive_carry_sum);
    } else {
        for (uint x_in = tid; x_in < params.e_in_length; x_in += threads) {
            uint row_index = row_base + x_in;
            SolinasFp128 weight = e_in[x_in];
            uint flags = 0u;
            if (tile == 1u || tile == 2u) {
                flags = booleanity_address_successor_load_flags(
                    validity, row_index);
            }
            for (uint local = 0u; local < 6u; local++) {
                uint selector = selector_base + local;
                bool present = true;
                if (selector == 16u || selector == 17u) {
                    present =
                        (flags & BOOLEANITY_ADDRESS_SUCCESSOR_BYTECODE_PRESENT) != 0u;
                } else if (selector == 18u || selector == 19u) {
                    present =
                        (flags & BOOLEANITY_ADDRESS_SUCCESSOR_RAM_PRESENT) != 0u;
                }
                if (present) {
                    uint hot = booleanity_address_successor_load_hot(
                        hot_rows, selector, row_index, params);
                    booleanity_address_successor_atomic_add(
                        sums, local, hot, weight);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    SolinasFp128 outer = e_out[x_out];
    uint fields = selectors_in_tile * BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
    for (uint field = tid; field < fields; field += threads) {
        uint local = field / BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
        uint bin = field & (BOOLEANITY_ADDRESS_SUCCESSOR_BINS - 1u);
        uint selector = selector_base + local;
        SolinasFp128 value = solinas_deferred_atomic_reduce_5(sums, field);
        partials[booleanity_address_successor_partial_index(
            x_out, selector, bin)] = solinas_mul_wide(outer, value);
    }
}

kernel void solinas_booleanity_address_successor_finalize(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant BooleanityAddressSuccessorParams& params [[buffer(2)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint selector [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    uint bin = tid & (BOOLEANITY_ADDRESS_SUCCESSOR_BINS - 1u);
    uint shard = tid / BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
    uint shards = threads / BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
    SolinasFp128 sum = solinas_zero();
    for (uint x_out = shard; x_out < params.e_out_length; x_out += shards) {
        sum = solinas_add(
            sum,
            partials[booleanity_address_successor_partial_index(
                x_out, selector, bin)]);
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (shard == 0u) {
        for (uint other = 1u; other < shards; other++) {
            sum = solinas_add(
                sum,
                shared[other * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + bin]);
        }
        output[selector * BOOLEANITY_ADDRESS_SUCCESSOR_BINS + bin] = sum;
    }
}
