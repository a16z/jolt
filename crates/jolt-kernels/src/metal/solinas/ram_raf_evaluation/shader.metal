#define RAM_RAF_ADDRESS_DOMAIN 8192u
#define RAM_RAF_INNER_LENGTH 32768u
#define RAM_RAF_TILE_COUNT 6u
#define RAM_RAF_THREADS 1024u
#define RAM_RAF_ACCUMULATOR_WORDS 5u
#define RAM_RAF_NONZERO_COUNTER 0u
#define RAM_RAF_INVALID_COUNTER 1u
#define RAM_RAF_ACCESSED_COUNTER 2u
#define RAM_RAF_UNSUPPORTED_COUNTER 3u

struct RamRafFoldParams {
    uint rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint tile_addresses;
    uint tiles;
    uint accumulator_words;
    uint no_access;
};

struct RamRafSegmentedParams {
    uint rows;
    uint addresses;
    uint accesses;
    uint inner_length;
    uint outer_length;
    uint cold_segment_threshold;
    uint hot_message_chunk_size;
    uint bounded_address_count;
    uint hot_address_count;
    uint hot_message_chunk_count;
};

struct RamRafSegment {
    uint offset;
    uint length;
    uint capacity;
    uint aux_offset;
};

struct RamRafHotSegment {
    uint segment_index;
    uint first_chunk;
    uint chunk_count;
    uint aux_offset;
};

struct RamRafHotChunk {
    uint hot_index;
    uint local_offset;
};

inline bool ram_raf_is_zero(SolinasFp128 value) {
    return all(value.limb == uint4(0u));
}

inline uint ram_raf_simd_sum_u32(uint value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value += simd_shuffle_down(value, offset);
    }
    return value;
}

inline void ram_raf_threadgroup_atomic_add_5(
    threadgroup atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * RAM_RAF_ACCUMULATOR_WORDS;
    uint carry = 0u;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong addend = (ulong)value.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(
            &sums[base + limb],
            low,
            memory_order_relaxed);
        carry = (uint)(addend >> 32u) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry != 0u) {
        atomic_fetch_add_explicit(
            &sums[base + 4u],
            carry,
            memory_order_relaxed);
    }
}

inline SolinasFp128 ram_raf_threadgroup_atomic_reduce_5(
    threadgroup atomic_uint* sums,
    uint field)
{
    uint base = field * RAM_RAF_ACCUMULATOR_WORDS;
    SolinasFp128 low;
    for (uint limb = 0u; limb < 4u; limb++) {
        low.limb[limb] = atomic_load_explicit(
            &sums[base + limb],
            memory_order_relaxed);
    }
    uint overflow = atomic_load_explicit(
        &sums[base + 4u],
        memory_order_relaxed);

    SolinasCorrection canonical = solinas_add_offset(low);
    low = solinas_select(canonical.carry != 0u, canonical.value, low);
    ulong correction_word = (ulong)overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return solinas_add(low, correction);
}

inline void ram_raf_device_atomic_add_5(
    device atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * RAM_RAF_ACCUMULATOR_WORDS;
    uint carry = 0u;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong addend = (ulong)value.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(
            &sums[base + limb],
            low,
            memory_order_relaxed);
        carry = (uint)(addend >> 32u) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry != 0u) {
        atomic_fetch_add_explicit(
            &sums[base + 4u],
            carry,
            memory_order_relaxed);
    }
}

inline SolinasFp128 ram_raf_device_atomic_reduce_5(
    device atomic_uint* sums,
    uint field)
{
    uint base = field * RAM_RAF_ACCUMULATOR_WORDS;
    SolinasFp128 low;
    for (uint limb = 0u; limb < 4u; limb++) {
        low.limb[limb] = atomic_load_explicit(
            &sums[base + limb],
            memory_order_relaxed);
    }
    uint overflow = atomic_load_explicit(
        &sums[base + 4u],
        memory_order_relaxed);

    SolinasCorrection canonical = solinas_add_offset(low);
    low = solinas_select(canonical.carry != 0u, canonical.value, low);
    ulong correction_word = (ulong)overflow * (ulong)SOLINAS_OFFSET;
    SolinasFp128 correction = solinas_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return solinas_add(low, correction);
}

kernel void solinas_ram_raf_fold_tiles(
    device const uint* cycle_addresses [[buffer(0)]],
    device const SolinasFp128* e_lo [[buffer(1)]],
    device const SolinasFp128* e_hi [[buffer(2)]],
    device atomic_uint* output [[buffer(3)]],
    device atomic_uint* counters [[buffer(4)]],
    constant RamRafFoldParams& params [[buffer(5)]],
    threadgroup atomic_uint* local_sums [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group_threads [[threads_per_threadgroup]])
{
    uint outer = group.x;
    uint tile = group.y;
    uint threads = group_threads.x;
    bool supported = (threads == 512u || threads == RAM_RAF_THREADS)
        && group_threads.y == 1u
        && group_threads.z == 1u
        && params.rows == params.inner_length * params.outer_length
        && params.addresses == RAM_RAF_ADDRESS_DOMAIN
        && params.inner_length == RAM_RAF_INNER_LENGTH
        && params.tiles == RAM_RAF_TILE_COUNT
        && params.tile_addresses >= 1376u
        && params.tile_addresses <= 1632u
        && params.accumulator_words == RAM_RAF_ACCUMULATOR_WORDS
        && params.no_access == 0xffffffffu
        && outer < params.outer_length
        && tile < params.tiles
        && group.z == 0u;
    if (!supported) {
        if (tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_UNSUPPORTED_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }

    uint tile_start = tile * params.tile_addresses;
    if (tile_start >= params.addresses) {
        return;
    }
    uint active = min(params.tile_addresses, params.addresses - tile_start);
    uint local_words = active * RAM_RAF_ACCUMULATOR_WORDS;
    for (uint word = tid; word < local_words; word += threads) {
        atomic_store_explicit(&local_sums[word], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint outer_base = outer * params.inner_length;
    uint lane_accessed = 0u;
    uint lane_invalid = 0u;
    for (uint inner = tid; inner < params.inner_length; inner += threads) {
        uint address = cycle_addresses[outer_base + inner];
        if (address == params.no_access) {
            continue;
        }
        if (address >= params.addresses) {
            if (tile == 0u) {
                lane_invalid += 1u;
            }
            continue;
        }
        if (tile == 0u) {
            lane_accessed += 1u;
        }
        if (address >= tile_start && address < tile_start + active) {
            ram_raf_threadgroup_atomic_add_5(
                local_sums,
                address - tile_start,
                e_lo[inner]);
        }
    }
    if (tile == 0u) {
        uint simd_accessed = ram_raf_simd_sum_u32(lane_accessed);
        uint simd_invalid = ram_raf_simd_sum_u32(lane_invalid);
        if (lane == 0u) {
            if (simd_accessed != 0u) {
                atomic_fetch_add_explicit(
                    &counters[RAM_RAF_ACCESSED_COUNTER],
                    simd_accessed,
                    memory_order_relaxed);
            }
            if (simd_invalid != 0u) {
                atomic_fetch_add_explicit(
                    &counters[RAM_RAF_INVALID_COUNTER],
                    simd_invalid,
                    memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane_nonzero = 0u;
    for (uint local = tid; local < active; local += threads) {
        SolinasFp128 subtotal = ram_raf_threadgroup_atomic_reduce_5(
            local_sums,
            local);
        if (!ram_raf_is_zero(subtotal)) {
            SolinasFp128 weighted = solinas_mul_wide(subtotal, e_hi[outer]);
            ram_raf_device_atomic_add_5(output, tile_start + local, weighted);
            lane_nonzero += 1u;
        }
    }
    uint simd_nonzero = ram_raf_simd_sum_u32(lane_nonzero);
    if (lane == 0u && simd_nonzero != 0u) {
        atomic_fetch_add_explicit(
            &counters[RAM_RAF_NONZERO_COUNTER],
            simd_nonzero,
            memory_order_relaxed);
    }
}

kernel void solinas_ram_raf_finalize(
    device atomic_uint* sums [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamRafFoldParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.addresses
        || params.addresses != RAM_RAF_ADDRESS_DOMAIN
        || params.accumulator_words != RAM_RAF_ACCUMULATOR_WORDS) {
        return;
    }
    output[gid] = ram_raf_device_atomic_reduce_5(sums, gid);
}

inline bool ram_raf_segmented_supported(
    constant RamRafSegmentedParams& params)
{
    return params.rows == params.inner_length * params.outer_length
        && params.inner_length == RAM_RAF_INNER_LENGTH
        && params.addresses != 0u
        && params.accesses != 0u
        && params.cold_segment_threshold != 0u
        && params.hot_message_chunk_size == 4096u;
}

inline SolinasFp128 ram_raf_simd_sum_fp128(
    SolinasFp128 value,
    uint lane)
{
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        SolinasFp128 other;
        other.limb[0] = simd_shuffle_down(value.limb[0], offset);
        other.limb[1] = simd_shuffle_down(value.limb[1], offset);
        other.limb[2] = simd_shuffle_down(value.limb[2], offset);
        other.limb[3] = simd_shuffle_down(value.limb[3], offset);
        if (lane < offset) {
            value = solinas_add(value, other);
        }
    }
    return value;
}

inline SolinasFp128 ram_raf_threadgroup_sum_fp128(
    SolinasFp128 value,
    threadgroup SolinasFp128* partials,
    uint lane,
    uint simd_index,
    uint simdgroups)
{
    value = ram_raf_simd_sum_fp128(value, lane);
    if (lane == 0u) {
        partials[simd_index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = solinas_zero();
    if (simd_index == 0u && lane < simdgroups) {
        value = partials[lane];
    }
    if (simd_index == 0u) {
        value = ram_raf_simd_sum_fp128(value, lane);
    }
    return value;
}

inline SolinasFp128 ram_raf_segmented_sum_range(
    device const uint* blocks,
    device const SolinasFp128* e_lo,
    device const SolinasFp128* e_hi,
    device atomic_uint* counters,
    constant RamRafSegmentedParams& params,
    uint offset,
    uint length,
    uint tid,
    uint threads)
{
    SolinasFp128 sum = solinas_zero();
    for (uint local = tid; local < length; local += threads) {
        uint cycle = blocks[offset + local] & 0x7fffffffu;
        if (cycle >= params.rows) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_INVALID_COUNTER],
                1u,
                memory_order_relaxed);
            continue;
        }
        uint inner = cycle & (params.inner_length - 1u);
        uint outer = cycle / params.inner_length;
        SolinasFp128 weight = solinas_mul_wide(e_lo[inner], e_hi[outer]);
        sum = solinas_add(sum, weight);
    }
    return sum;
}

kernel void solinas_ram_raf_segmented_cold(
    device const RamRafSegment* segments [[buffer(0)]],
    device const uint* blocks [[buffer(1)]],
    device const SolinasFp128* e_lo [[buffer(2)]],
    device const SolinasFp128* e_hi [[buffer(3)]],
    device SolinasFp128* output [[buffer(4)]],
    device atomic_uint* counters [[buffer(5)]],
    constant RamRafSegmentedParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (!ram_raf_segmented_supported(params)) {
        if (gid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_UNSUPPORTED_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    if (gid >= params.addresses) {
        return;
    }
    RamRafSegment segment = segments[gid];
    output[gid] = solinas_zero();
    if (segment.length > params.cold_segment_threshold) {
        return;
    }
    if ((ulong)segment.offset + (ulong)segment.length > (ulong)params.accesses) {
        atomic_fetch_add_explicit(
            &counters[RAM_RAF_INVALID_COUNTER],
            1u,
            memory_order_relaxed);
        return;
    }
    output[gid] = ram_raf_segmented_sum_range(
        blocks,
        e_lo,
        e_hi,
        counters,
        params,
        segment.offset,
        segment.length,
        0u,
        1u);
}

kernel void solinas_ram_raf_segmented_bounded(
    device const RamRafSegment* segments [[buffer(0)]],
    device const uint* blocks [[buffer(1)]],
    device const uint* bounded_segments [[buffer(2)]],
    device const SolinasFp128* e_lo [[buffer(3)]],
    device const SolinasFp128* e_hi [[buffer(4)]],
    device SolinasFp128* output [[buffer(5)]],
    device atomic_uint* counters [[buffer(6)]],
    constant RamRafSegmentedParams& params [[buffer(7)]],
    threadgroup SolinasFp128* partials [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint3 group_threads [[threads_per_threadgroup]])
{
    uint item = group.x;
    bool supported = ram_raf_segmented_supported(params)
        && group_threads.x == 256u
        && group_threads.y == 1u
        && group_threads.z == 1u;
    if (!supported) {
        if (item == 0u && tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_UNSUPPORTED_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    if (item >= params.bounded_address_count) {
        return;
    }
    uint address = bounded_segments[item];
    if (address >= params.addresses) {
        if (tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_INVALID_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    RamRafSegment segment = segments[address];
    bool valid = segment.length > params.cold_segment_threshold
        && segment.length <= params.hot_message_chunk_size
        && (ulong)segment.offset + (ulong)segment.length <= (ulong)params.accesses;
    if (!valid) {
        if (tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_INVALID_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    SolinasFp128 sum = ram_raf_segmented_sum_range(
        blocks,
        e_lo,
        e_hi,
        counters,
        params,
        segment.offset,
        segment.length,
        tid,
        group_threads.x);
    sum = ram_raf_threadgroup_sum_fp128(
        sum,
        partials,
        lane,
        simd_index,
        simdgroups);
    if (tid == 0u) {
        output[address] = sum;
    }
}

kernel void solinas_ram_raf_segmented_hot_chunk(
    device const RamRafSegment* segments [[buffer(0)]],
    device const uint* blocks [[buffer(1)]],
    device const RamRafHotSegment* hot_segments [[buffer(2)]],
    device const RamRafHotChunk* hot_chunks [[buffer(3)]],
    device const SolinasFp128* e_lo [[buffer(4)]],
    device const SolinasFp128* e_hi [[buffer(5)]],
    device SolinasFp128* hot_partials [[buffer(6)]],
    device atomic_uint* counters [[buffer(7)]],
    constant RamRafSegmentedParams& params [[buffer(8)]],
    threadgroup SolinasFp128* partials [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint3 group_threads [[threads_per_threadgroup]])
{
    uint chunk_index = group.x;
    bool supported = ram_raf_segmented_supported(params)
        && group_threads.x == 256u
        && group_threads.y == 1u
        && group_threads.z == 1u;
    if (!supported) {
        if (chunk_index == 0u && tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_UNSUPPORTED_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    if (chunk_index >= params.hot_message_chunk_count) {
        return;
    }
    RamRafHotChunk chunk = hot_chunks[chunk_index];
    if (chunk.hot_index >= params.hot_address_count) {
        if (tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_INVALID_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    RamRafHotSegment hot = hot_segments[chunk.hot_index];
    if (hot.segment_index >= params.addresses) {
        if (tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_INVALID_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    RamRafSegment segment = segments[hot.segment_index];
    uint length = min(
        params.hot_message_chunk_size,
        segment.length - min(segment.length, chunk.local_offset));
    bool valid = segment.length > params.hot_message_chunk_size
        && chunk.local_offset < segment.length
        && (ulong)segment.offset + (ulong)chunk.local_offset + (ulong)length
            <= (ulong)params.accesses;
    if (!valid) {
        if (tid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_INVALID_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    SolinasFp128 sum = ram_raf_segmented_sum_range(
        blocks,
        e_lo,
        e_hi,
        counters,
        params,
        segment.offset + chunk.local_offset,
        length,
        tid,
        group_threads.x);
    sum = ram_raf_threadgroup_sum_fp128(
        sum,
        partials,
        lane,
        simd_index,
        simdgroups);
    if (tid == 0u) {
        hot_partials[chunk_index] = sum;
    }
}

kernel void solinas_ram_raf_segmented_hot_finalize(
    device const RamRafHotSegment* hot_segments [[buffer(0)]],
    device const SolinasFp128* hot_partials [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    device atomic_uint* counters [[buffer(3)]],
    constant RamRafSegmentedParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (!ram_raf_segmented_supported(params)) {
        if (gid == 0u) {
            atomic_fetch_add_explicit(
                &counters[RAM_RAF_UNSUPPORTED_COUNTER],
                1u,
                memory_order_relaxed);
        }
        return;
    }
    if (gid >= params.hot_address_count) {
        return;
    }
    RamRafHotSegment hot = hot_segments[gid];
    if (hot.segment_index >= params.addresses
        || (ulong)hot.first_chunk + (ulong)hot.chunk_count
            > (ulong)params.hot_message_chunk_count) {
        atomic_fetch_add_explicit(
            &counters[RAM_RAF_INVALID_COUNTER],
            1u,
            memory_order_relaxed);
        return;
    }
    SolinasFp128 sum = solinas_zero();
    for (uint local = 0u; local < hot.chunk_count; local++) {
        sum = solinas_add(sum, hot_partials[hot.first_chunk + local]);
    }
    output[hot.segment_index] = sum;
}
