#include <metal_stdlib>
using namespace metal;

constant uint LOOKUP_TABLES = 40u;
constant uint GROUPED_SEGMENTS = 82u;
constant uint GROUPED_SEGMENT_OFFSETS = 83u;
constant uint PRODUCER_CHUNK_ROWS = 4096u;
constant uint PRODUCER_THREADS = 1024u;
constant uint MAX_TOTAL_ROWS = 1u << 28;
constant uint MAX_SHARD_ROWS = 1u << 26;

constant uint STATUS_INVALID_GEOMETRY = 1u << 0;
constant uint STATUS_INVALID_SELECTOR = 1u << 1;
constant uint STATUS_INVALID_LAYOUT = 1u << 2;
constant uint STATUS_OUT_OF_BOUNDS = 1u << 3;
constant uint STATUS_COUNT_MISMATCH = 1u << 4;

struct ScatterParams {
    uint total_rows;
    uint shard_row_start;
    uint shard_rows;
    uint chunks;
    uint segments;
    uint chunk_rows;
    uint lookup_lo_elements;
    uint lookup_hi_elements;
    uint cycle_claim_elements;
    uint grouped_lo_elements;
    uint grouped_hi_elements;
    uint cycle_to_grouped_elements;
    uint chunk_base_elements;
    uint segment_offset_elements;
    uint status_elements;
    uint reserved;
};

kernel void instruction_read_raf_producer_scatter_4096(
    device const ulong* cycle_lookup_lo [[buffer(0)]],
    device const ulong* cycle_lookup_hi [[buffer(1)]],
    device const uchar* cycle_claims [[buffer(2)]],
    device const uint* chunk_segment_bases [[buffer(3)]],
    device const uint* segment_offsets [[buffer(4)]],
    device ulong* grouped_lookup_lo [[buffer(5)]],
    device ulong* grouped_lookup_hi [[buffer(6)]],
    device uint* cycle_to_grouped_local [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]],
    constant ScatterParams& params [[buffer(9)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    threadgroup atomic_uint local_counts[82];

    // A real owner guarantees the status allocation before encoding. If that
    // prerequisite is absent, no diagnostic buffer is safe to touch.
    if (params.status_elements < 1u) {
        return;
    }

    bool absolute_bounds = false;
    if (params.shard_row_start <= params.total_rows) {
        absolute_bounds = params.shard_rows <= params.total_rows - params.shard_row_start;
    }
    uint expected_chunks = params.shard_rows == 0u
        ? 0u
        : 1u + (params.shard_rows - 1u) / PRODUCER_CHUNK_ROWS;
    bool base_product_safe = params.chunks <= 0xffffffffu / GROUPED_SEGMENTS;
    uint required_chunk_bases = base_product_safe
        ? params.chunks * GROUPED_SEGMENTS
        : 0u;
    bool capacities = params.lookup_lo_elements >= params.shard_rows
        && params.lookup_hi_elements >= params.shard_rows
        && params.cycle_claim_elements >= params.shard_rows
        && params.grouped_lo_elements >= params.shard_rows
        && params.grouped_hi_elements >= params.shard_rows
        && params.cycle_to_grouped_elements >= params.shard_rows
        && params.segment_offset_elements >= GROUPED_SEGMENT_OFFSETS
        && base_product_safe
        && params.chunk_base_elements >= required_chunk_bases;
    bool geometry = params.total_rows > 0u
        && params.total_rows <= MAX_TOTAL_ROWS
        && params.shard_rows > 0u
        && params.shard_rows <= MAX_SHARD_ROWS
        && absolute_bounds
        && params.chunks == expected_chunks
        && params.segments == GROUPED_SEGMENTS
        && params.chunk_rows == PRODUCER_CHUNK_ROWS
        && params.reserved == 0u
        && threads == PRODUCER_THREADS;
    if (!geometry || !capacities || chunk >= params.chunks) {
        if (lane == 0u) {
            atomic_fetch_or_explicit(status, STATUS_INVALID_GEOMETRY, memory_order_relaxed);
        }
        return;
    }

    if (lane < GROUPED_SEGMENTS) {
        atomic_store_explicit(&local_counts[lane], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_begin = chunk * PRODUCER_CHUNK_ROWS;
    uint local_end = min(local_begin + PRODUCER_CHUNK_ROWS, params.shard_rows);
    for (uint local_cycle = local_begin + lane;
         local_cycle < local_end;
         local_cycle += threads) {
        uint absolute_cycle = params.shard_row_start + local_cycle;
        if (absolute_cycle >= params.total_rows
            || local_cycle >= params.lookup_lo_elements
            || local_cycle >= params.lookup_hi_elements
            || local_cycle >= params.cycle_claim_elements
            || local_cycle >= params.cycle_to_grouped_elements) {
            atomic_fetch_or_explicit(status, STATUS_OUT_OF_BOUNDS, memory_order_relaxed);
            continue;
        }

        uchar claim = cycle_claims[local_cycle];
        uint table_plus_one = uint(claim & 0x7fu);
        uint raf = uint(claim >> 7);
        if (table_plus_one > LOOKUP_TABLES) {
            atomic_fetch_or_explicit(status, STATUS_INVALID_SELECTOR, memory_order_relaxed);
            continue;
        }

        uint segment = 2u * table_plus_one + raf;
        uint base_index = chunk * GROUPED_SEGMENTS + segment;
        uint end_index = chunk + 1u < params.chunks
            ? base_index + GROUPED_SEGMENTS
            : segment + 1u;
        if (base_index >= params.chunk_base_elements
            || (chunk + 1u < params.chunks && end_index >= params.chunk_base_elements)
            || (chunk + 1u == params.chunks && end_index >= params.segment_offset_elements)) {
            atomic_fetch_or_explicit(status, STATUS_OUT_OF_BOUNDS, memory_order_relaxed);
            continue;
        }

        uint base = chunk_segment_bases[base_index];
        uint range_end = chunk + 1u < params.chunks
            ? chunk_segment_bases[end_index]
            : segment_offsets[end_index];
        uint local_rank = atomic_fetch_add_explicit(
            &local_counts[segment], 1u, memory_order_relaxed);
        if (base > range_end || local_rank > 0xffffffffu - base) {
            atomic_fetch_or_explicit(status, STATUS_INVALID_LAYOUT, memory_order_relaxed);
            continue;
        }
        uint grouped = base + local_rank;
        if (grouped >= range_end
            || grouped >= params.shard_rows
            || grouped >= params.grouped_lo_elements
            || grouped >= params.grouped_hi_elements) {
            atomic_fetch_or_explicit(status, STATUS_OUT_OF_BOUNDS, memory_order_relaxed);
            continue;
        }

        // Atomic rank assignment intentionally leaves intra-range order free.
        grouped_lookup_lo[grouped] = cycle_lookup_lo[local_cycle];
        grouped_lookup_hi[grouped] = cycle_lookup_hi[local_cycle];
        cycle_to_grouped_local[local_cycle] = grouped;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < GROUPED_SEGMENTS) {
        uint base_index = chunk * GROUPED_SEGMENTS + lane;
        uint base = chunk_segment_bases[base_index];
        uint range_end = chunk + 1u < params.chunks
            ? chunk_segment_bases[base_index + GROUPED_SEGMENTS]
            : segment_offsets[lane + 1u];
        uint observed = atomic_load_explicit(&local_counts[lane], memory_order_relaxed);
        if (base > range_end || observed != range_end - base) {
            atomic_fetch_or_explicit(status, STATUS_COUNT_MISMATCH, memory_order_relaxed);
        }
    }
}
