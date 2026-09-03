#include <metal_stdlib>
using namespace metal;

// Table count injected by the host at library assembly (`source.rs`) from
// `LookupTableKind::COUNT`, so a new lookup table cannot silently desync the
// segment layout below.
constant uint INSTRUCTION_READ_RAF_TABLES = INSTRUCTION_READ_RAF_TABLE_COUNT;
constant uint INSTRUCTION_READ_RAF_SEGMENTS = 2u * INSTRUCTION_READ_RAF_TABLES + 2u;
constant uint INSTRUCTION_READ_RAF_CHUNK_ROWS = 4096u;

constant uint INSTRUCTION_READ_RAF_STATUS_GEOMETRY = 1u << 0;
constant uint INSTRUCTION_READ_RAF_STATUS_SELECTOR = 1u << 1;
constant uint INSTRUCTION_READ_RAF_STATUS_RANGE = 1u << 2;
constant uint INSTRUCTION_READ_RAF_STATUS_COUNT = 1u << 3;
constant uint INSTRUCTION_READ_RAF_STATUS_BYTECODE_TOPOLOGY = 1u << 4;

constant uint IRRAF_BYTECODE_ADDRESS_INNER_LOG2 = 15u;
constant uint IRRAF_BYTECODE_ADDRESS_INNER_LENGTH = 1u << IRRAF_BYTECODE_ADDRESS_INNER_LOG2;
constant uint BYTECODE_ADDRESS_COUNT = 1u << 13u;
constant uint BYTECODE_ADDRESS_DESCRIPTOR_PIVOT_START_MASK = 0x000fffffu;
constant ulong BYTECODE_ADDRESS_PC_MASK = (1ul << 56u) - 1ul;

struct InstructionReadRafSourcePrimerParams {
    ulong word_counts[2];
    uint page_words;
    uint total_threads;
};

kernel void solinas_instruction_read_raf_source_primer(
    device const uint* rows [[buffer(0)]],
    device const uint* claims [[buffer(1)]],
    device uint* checksums [[buffer(2)]],
    constant InstructionReadRafSourcePrimerParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.total_threads) {
        return;
    }
    const ulong row_pages = (params.word_counts[0] + params.page_words - 1u)
        / params.page_words;
    const ulong claim_pages = (params.word_counts[1] + params.page_words - 1u)
        / params.page_words;
    const ulong total_pages = row_pages + claim_pages;
    uint checksum = 0x9e3779b9u ^ gid;
    for (ulong page = gid; page < total_pages; page += params.total_threads) {
        const uint value = page < row_pages
            ? rows[page * params.page_words]
            : claims[(page - row_pages) * params.page_words];
        checksum ^= value ^ (uint)page;
        checksum = ((checksum << 5u) | (checksum >> 27u)) * 0x85ebca6bu;
    }
    checksums[gid] = checksum;
}

struct InstructionReadRafLookup {
    ulong lo;
    ulong hi;
};

struct BytecodeAddressChunkDescriptor {
    ushort address;
    ushort base;
    uint packed_count_and_pivot_start;
};

inline uint bytecode_address_descriptor_pivot_start(
    BytecodeAddressChunkDescriptor descriptor)
{
    return descriptor.packed_count_and_pivot_start
        & BYTECODE_ADDRESS_DESCRIPTOR_PIVOT_START_MASK;
}

inline uint bytecode_address_descriptor_count(
    BytecodeAddressChunkDescriptor descriptor)
{
    return (descriptor.packed_count_and_pivot_start >> 20u) + 1u;
}

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
    uint bytecode_enabled;
    uint bytecode_physical_rows;
    uint bytecode_descriptor_elements;
    uint bytecode_pivot_elements;
    uint bytecode_chunk_offset_elements;
    uint bytecode_occurrence_elements;
    uint bytecode_magnitude_elements;
    uint bytecode_inner_log2;
    uint bytecode_max_descriptors_per_chunk;
    uint bytecode_max_pivots_per_chunk;
};

kernel void solinas_instruction_read_raf_compatibility_scatter(
    device const ulong* rows [[buffer(0)]],
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
    device const BytecodeAddressChunkDescriptor* bytecode_descriptors [[buffer(12)]],
    device const ushort* bytecode_pivots [[buffer(13)]],
    device const uint* bytecode_chunk_offsets [[buffer(14)]],
    device ushort* bytecode_occurrences [[buffer(15)]],
    device ulong* bytecode_magnitudes [[buffer(16)]],
    threadgroup atomic_uint* local_counts [[threadgroup(0)]],
    threadgroup BytecodeAddressChunkDescriptor* local_bytecode_descriptors [[threadgroup(1)]],
    threadgroup ushort* local_bytecode_pivots [[threadgroup(2)]],
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
    bool bytecode_geometry = params.bytecode_enabled <= 1u
        && (params.bytecode_enabled == 0u
            || (params.bytecode_physical_rows != 0u
                && params.bytecode_physical_rows <= params.rows
                && params.bytecode_descriptor_elements != 0u
                && params.bytecode_chunk_offset_elements
                    == 2u * ((params.bytecode_physical_rows
                        + INSTRUCTION_READ_RAF_CHUNK_ROWS - 1u)
                        / INSTRUCTION_READ_RAF_CHUNK_ROWS)
                && params.bytecode_occurrence_elements == params.bytecode_physical_rows
                && params.bytecode_magnitude_elements == params.bytecode_physical_rows
                && params.bytecode_inner_log2 == IRRAF_BYTECODE_ADDRESS_INNER_LOG2
                && params.bytecode_max_descriptors_per_chunk != 0u));
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
        && powers_of_two
        && bytecode_geometry;
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

    uint bytecode_chunks = (params.bytecode_physical_rows
        + INSTRUCTION_READ_RAF_CHUNK_ROWS - 1u) / INSTRUCTION_READ_RAF_CHUNK_ROWS;
    uint bytecode_descriptor_begin = 0u;
    uint bytecode_descriptor_end = 0u;
    uint bytecode_pivot_begin = 0u;
    uint bytecode_pivot_end = 0u;
    bool bytecode_chunk_valid = params.bytecode_enabled == 0u || chunk >= bytecode_chunks;
    if (params.bytecode_enabled != 0u && chunk < bytecode_chunks) {
        bytecode_descriptor_begin = bytecode_chunk_offsets[2u * chunk];
        bytecode_descriptor_end = bytecode_chunk_offsets[2u * chunk + 1u];
        uint descriptor_count = bytecode_descriptor_end - bytecode_descriptor_begin;
        bytecode_chunk_valid = bytecode_descriptor_begin < bytecode_descriptor_end
            && bytecode_descriptor_end < params.bytecode_descriptor_elements
            && descriptor_count <= params.bytecode_max_descriptors_per_chunk;
        if (bytecode_chunk_valid) {
            BytecodeAddressChunkDescriptor sentinel =
                bytecode_descriptors[bytecode_descriptor_end];
            bytecode_pivot_begin = bytecode_address_descriptor_pivot_start(
                bytecode_descriptors[bytecode_descriptor_begin]);
            bytecode_pivot_end = bytecode_address_descriptor_pivot_start(sentinel);
            uint outer = (chunk * INSTRUCTION_READ_RAF_CHUNK_ROWS)
                >> IRRAF_BYTECODE_ADDRESS_INNER_LOG2;
            uint outer_begin = outer * IRRAF_BYTECODE_ADDRESS_INNER_LENGTH;
            uint outer_rows = min(
                IRRAF_BYTECODE_ADDRESS_INNER_LENGTH,
                params.bytecode_physical_rows - outer_begin);
            bytecode_chunk_valid = bytecode_pivot_begin <= bytecode_pivot_end
                && bytecode_pivot_end <= params.bytecode_pivot_elements
                && bytecode_pivot_end - bytecode_pivot_begin
                    <= params.bytecode_max_pivots_per_chunk
                && uint(sentinel.address) == 0xffffu
                && uint(sentinel.base) == outer_rows;
        }
        if (bytecode_chunk_valid) {
            uint descriptor_count = bytecode_descriptor_end - bytecode_descriptor_begin;
            for (uint index = lane; index <= descriptor_count; index += threads) {
                local_bytecode_descriptors[index] =
                    bytecode_descriptors[bytecode_descriptor_begin + index];
            }
            uint pivot_count = bytecode_pivot_end - bytecode_pivot_begin;
            for (uint index = lane; index < pivot_count; index += threads) {
                local_bytecode_pivots[index] = bytecode_pivots[bytecode_pivot_begin + index];
            }
        } else if (lane == 0u) {
            atomic_fetch_or_explicit(
                status,
                INSTRUCTION_READ_RAF_STATUS_BYTECODE_TOPOLOGY,
                memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint begin = chunk * INSTRUCTION_READ_RAF_CHUNK_ROWS;
    uint end = min(begin + INSTRUCTION_READ_RAF_CHUNK_ROWS, params.rows);
    for (uint cycle = begin + lane; cycle < end; cycle += threads) {
        uchar claim = claims[cycle];
        uint table_plus_one = uint(claim & 0x3fu);
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

        BooleanityRow row = booleanity_row_load(rows, params.rows, cycle);
        InstructionReadRafLookup lookup;
        lookup.lo = row.lookup_lo;
        lookup.hi = row.lookup_hi;
        lookups[grouped] = lookup;
        packed_rows[grouped] = claim & 0xbfu;
        cycle_to_table_major[cycle] = grouped;
        uint x_in = cycle & (params.e_in_length - 1u);
        uint x_out = cycle >> params.e_in_log2;
        weights[grouped] = solinas_mul_wide(e_out[x_out], e_in[x_in]);

        if (params.bytecode_enabled != 0u && cycle < params.bytecode_physical_rows) {
            if (!bytecode_chunk_valid) {
                continue;
            }
            uint descriptor_count = bytecode_descriptor_end - bytecode_descriptor_begin;
            ulong pc_plus_one = row.packed_pc_and_flags & BYTECODE_ADDRESS_PC_MASK;
            uint address = pc_plus_one == 0ul ? 0u : uint(pc_plus_one - 1ul);
            bool valid_topology = address < BYTECODE_ADDRESS_COUNT
                && descriptor_count != 0u;
            if (!valid_topology) {
                atomic_fetch_or_explicit(
                    status,
                    INSTRUCTION_READ_RAF_STATUS_BYTECODE_TOPOLOGY,
                    memory_order_relaxed);
                continue;
            }

            uint descriptor_lo = 0u;
            uint descriptor_hi = descriptor_count;
            while (descriptor_lo < descriptor_hi) {
                uint midpoint = descriptor_lo + (descriptor_hi - descriptor_lo) / 2u;
                if (uint(local_bytecode_descriptors[midpoint].address) < address) {
                    descriptor_lo = midpoint + 1u;
                } else {
                    descriptor_hi = midpoint;
                }
            }
            if (descriptor_lo >= descriptor_count
                || uint(local_bytecode_descriptors[descriptor_lo].address) != address) {
                atomic_fetch_or_explicit(
                    status,
                    INSTRUCTION_READ_RAF_STATUS_BYTECODE_TOPOLOGY,
                    memory_order_relaxed);
                continue;
            }

            BytecodeAddressChunkDescriptor descriptor =
                local_bytecode_descriptors[descriptor_lo];
            BytecodeAddressChunkDescriptor next_descriptor =
                local_bytecode_descriptors[descriptor_lo + 1u];
            uint descriptor_pivot_start =
                bytecode_address_descriptor_pivot_start(descriptor);
            uint next_pivot_start =
                bytecode_address_descriptor_pivot_start(next_descriptor);
            uint pivot_begin = descriptor_pivot_start - bytecode_pivot_begin;
            uint pivot_end = next_pivot_start - bytecode_pivot_begin;
            uint staged_pivots = bytecode_pivot_end - bytecode_pivot_begin;
            uint descriptor_base = uint(descriptor.base);
            uint descriptor_count_rows = bytecode_address_descriptor_count(descriptor);
            uint descriptor_end = descriptor_base + descriptor_count_rows;
            if (descriptor_pivot_start < bytecode_pivot_begin
                || descriptor_pivot_start > next_pivot_start
                || next_pivot_start > bytecode_pivot_end
                || pivot_end > staged_pivots) {
                atomic_fetch_or_explicit(
                    status,
                    INSTRUCTION_READ_RAF_STATUS_BYTECODE_TOPOLOGY,
                    memory_order_relaxed);
                continue;
            }
            uint cycle_in_chunk = cycle & (INSTRUCTION_READ_RAF_CHUNK_ROWS - 1u);
            uint pivot_lo = pivot_begin;
            uint pivot_hi = pivot_end;
            while (pivot_lo < pivot_hi) {
                uint midpoint = pivot_lo + (pivot_hi - pivot_lo) / 2u;
                if (uint(local_bytecode_pivots[midpoint]) <= cycle_in_chunk) {
                    pivot_lo = midpoint + 1u;
                } else {
                    pivot_hi = midpoint;
                }
            }
            uint rank_low = uint((row.packed_pc_and_flags >> 56u) & 0x7ful)
                | (uint(claim & 0x40u) << 1u);
            uint rank = rank_low | ((pivot_lo - pivot_begin) << 8u);
            uint outer = cycle >> IRRAF_BYTECODE_ADDRESS_INNER_LOG2;
            uint inner = cycle & (IRRAF_BYTECODE_ADDRESS_INNER_LENGTH - 1u);
            uint outer_base = outer * IRRAF_BYTECODE_ADDRESS_INNER_LENGTH;
            uint cell_begin = outer_base + descriptor_base;
            uint cell_end = outer_base + descriptor_end;
            uint destination = cell_begin + rank;
            if (rank >= INSTRUCTION_READ_RAF_CHUNK_ROWS
                || rank >= descriptor_count_rows
                || descriptor_end < descriptor_base
                || descriptor_end > uint(next_descriptor.base)
                || destination < cell_begin
                || destination >= cell_end
                || destination >= params.bytecode_physical_rows
                || destination >> IRRAF_BYTECODE_ADDRESS_INNER_LOG2 != outer) {
                atomic_fetch_or_explicit(
                    status,
                    INSTRUCTION_READ_RAF_STATUS_BYTECODE_TOPOLOGY,
                    memory_order_relaxed);
                continue;
            }
            uint negative = uint(row.packed_pc_and_flags >> 63u);
            bytecode_occurrences[destination] = ushort(inner | (negative << 15u));
            bytecode_magnitudes[destination] = row.fused_inc_magnitude;
        }
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
