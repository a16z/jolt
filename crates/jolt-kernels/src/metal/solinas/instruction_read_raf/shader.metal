#include <metal_stdlib>
using namespace metal;

constant uint INSTRUCTION_READ_RAF_TABLES = 40u;
constant uint INSTRUCTION_READ_RAF_SEGMENTS = 82u;
constant uint INSTRUCTION_READ_RAF_CHUNK_ROWS = 4096u;

constant uint INSTRUCTION_READ_RAF_STATUS_GEOMETRY = 1u << 0;
constant uint INSTRUCTION_READ_RAF_STATUS_SELECTOR = 1u << 1;
constant uint INSTRUCTION_READ_RAF_STATUS_RANGE = 1u << 2;
constant uint INSTRUCTION_READ_RAF_STATUS_COUNT = 1u << 3;

struct InstructionReadRafSourceRow {
    ulong lookup_lo;
    ulong lookup_hi;
    ulong ram_address_plus_one;
    ulong fused_inc_magnitude;
    ulong packed_pc_and_flags;
};

struct InstructionReadRafLookup {
    ulong lo;
    ulong hi;
};

struct InstructionReadRafScatterParams {
    uint rows;
    uint chunks;
    uint chunk_rows;
    uint segments;
    uint e_in_length;
    uint e_out_length;
    uint packed_rows_elements;
    uint lookup_elements;
    uint inverse_elements;
    uint weight_elements;
    uint status_elements;
    uint e_in_log2;
};

kernel void solinas_instruction_read_raf_compatibility_scatter(
    device const InstructionReadRafSourceRow* rows [[buffer(0)]],
    device const uchar* claims [[buffer(1)]],
    device const uint* chunk_segment_bases [[buffer(2)]],
    device const uint* segment_offsets [[buffer(3)]],
    device const SolinasFp128* e_in [[buffer(4)]],
    device const SolinasFp128* e_out [[buffer(5)]],
    device InstructionReadRafLookup* lookups [[buffer(6)]],
    device uchar* packed_rows [[buffer(7)]],
    device uint* cycle_to_table_major [[buffer(8)]],
    device SolinasFp128* weights [[buffer(9)]],
    device atomic_uint* status [[buffer(10)]],
    constant InstructionReadRafScatterParams& params [[buffer(11)]],
    threadgroup atomic_uint* local_counts [[threadgroup(0)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (params.status_elements == 0u) {
        return;
    }
    bool powers_of_two = params.e_in_length != 0u
        && (params.e_in_length & (params.e_in_length - 1u)) == 0u
        && params.e_out_length != 0u
        && (params.e_out_length & (params.e_out_length - 1u)) == 0u;
    bool geometry = params.rows != 0u
        && params.chunks == (params.rows + INSTRUCTION_READ_RAF_CHUNK_ROWS - 1u)
            / INSTRUCTION_READ_RAF_CHUNK_ROWS
        && params.chunk_rows == INSTRUCTION_READ_RAF_CHUNK_ROWS
        && params.segments == INSTRUCTION_READ_RAF_SEGMENTS
        && params.packed_rows_elements == params.rows
        && params.lookup_elements == params.rows
        && params.inverse_elements == params.rows
        && params.weight_elements == params.rows
        && params.e_in_length <= params.rows
        && params.e_out_length == params.rows / params.e_in_length
        && params.e_in_log2 < 31u
        && (1u << params.e_in_log2) == params.e_in_length
        && powers_of_two;
    if (!geometry || chunk >= params.chunks) {
        if (lane == 0u) {
            atomic_fetch_or_explicit(
                status,
                INSTRUCTION_READ_RAF_STATUS_GEOMETRY,
                memory_order_relaxed);
        }
        return;
    }

    if (lane < INSTRUCTION_READ_RAF_SEGMENTS) {
        atomic_store_explicit(&local_counts[lane], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint begin = chunk * INSTRUCTION_READ_RAF_CHUNK_ROWS;
    uint end = min(begin + INSTRUCTION_READ_RAF_CHUNK_ROWS, params.rows);
    for (uint cycle = begin + lane; cycle < end; cycle += threads) {
        uchar claim = claims[cycle];
        uint table_plus_one = uint(claim & 0x7fu);
        uint raf = uint(claim >> 7);
        if (table_plus_one > INSTRUCTION_READ_RAF_TABLES || raf > 1u) {
            atomic_fetch_or_explicit(
                status,
                INSTRUCTION_READ_RAF_STATUS_SELECTOR,
                memory_order_relaxed);
            continue;
        }
        uint logical = 2u * table_plus_one + raf;
        uint physical = logical < 2u
            ? logical + 2u * INSTRUCTION_READ_RAF_TABLES
            : logical - 2u;
        uint base_index = chunk * INSTRUCTION_READ_RAF_SEGMENTS + physical;
        uint base = chunk_segment_bases[base_index];
        uint range_end = chunk + 1u < params.chunks
            ? chunk_segment_bases[base_index + INSTRUCTION_READ_RAF_SEGMENTS]
            : segment_offsets[physical + 1u];
        uint local_rank = atomic_fetch_add_explicit(
            &local_counts[physical], 1u, memory_order_relaxed);
        uint grouped = base + local_rank;
        if (base > range_end || grouped < base || grouped >= range_end || grouped >= params.rows) {
            atomic_fetch_or_explicit(
                status,
                INSTRUCTION_READ_RAF_STATUS_RANGE,
                memory_order_relaxed);
            continue;
        }

        InstructionReadRafSourceRow row = rows[cycle];
        InstructionReadRafLookup lookup;
        lookup.lo = row.lookup_lo;
        lookup.hi = row.lookup_hi;
        lookups[grouped] = lookup;
        packed_rows[grouped] = claim;
        cycle_to_table_major[cycle] = grouped;
        uint x_in = cycle & (params.e_in_length - 1u);
        uint x_out = cycle >> params.e_in_log2;
        weights[grouped] = solinas_mul_wide(e_out[x_out], e_in[x_in]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < INSTRUCTION_READ_RAF_SEGMENTS) {
        uint base_index = chunk * INSTRUCTION_READ_RAF_SEGMENTS + lane;
        uint base = chunk_segment_bases[base_index];
        uint range_end = chunk + 1u < params.chunks
            ? chunk_segment_bases[base_index + INSTRUCTION_READ_RAF_SEGMENTS]
            : segment_offsets[lane + 1u];
        uint observed = atomic_load_explicit(&local_counts[lane], memory_order_relaxed);
        if (base > range_end || observed != range_end - base) {
            atomic_fetch_or_explicit(
                status,
                INSTRUCTION_READ_RAF_STATUS_COUNT,
                memory_order_relaxed);
        }
    }
}
