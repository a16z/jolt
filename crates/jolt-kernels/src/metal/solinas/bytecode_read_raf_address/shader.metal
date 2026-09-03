#if SOLINAS_OFFSET != 0xffffa7f7u
#error "bytecode address worker requires the Akita Solinas offset"
#endif

#define BYTECODE_ADDRESS_WORKER_STAGES 9u
#define BYTECODE_ADDRESS_WORKER_BASE_STAGES 5u
#define BYTECODE_ADDRESS_WORKER_SIMD_WIDTH 32u
#define BYTECODE_ADDRESS_PACKED_ITEMS_PER_GROUP 4u

struct BytecodeAddressSparseWorkItem {
    ushort address;
    ushort outer;
    ushort start;
    ushort count;
};

struct BytecodeAddressSparseParams {
    uint physical_rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint work_items;
    uint stages;
    uint base_stages;
    uint reserved;
};

#define BYTECODE_ADDRESS_RESIDENT_WORK_ITEM_ROWS 4096u
#define BYTECODE_ADDRESS_RESIDENT_STATUS_INVALID 0u
#define BYTECODE_ADDRESS_RESIDENT_STATUS_FIRST_PC 1u

inline uint bytecode_address_resident_pc(
    device const ulong* rows,
    constant BytecodeAddressSparseParams& params,
    uint row)
{
    ulong metadata = booleanity_source_word(rows, params.physical_rows, 3u, row);
    ulong plus_one =
        (metadata >> BOOLEANITY_SOURCE_PC_SHIFT) & BOOLEANITY_SOURCE_PC_MASK;
    return plus_one == 0ul ? 0u : (uint)(plus_one - 1ul);
}

kernel void solinas_bytecode_address_resident_count(
    device const ulong* rows [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant BytecodeAddressSparseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.physical_rows) {
        return;
    }
    uint address = bytecode_address_resident_pc(rows, params, gid);
    if (gid == 0u) {
        atomic_store_explicit(
            &status[BYTECODE_ADDRESS_RESIDENT_STATUS_FIRST_PC],
            address,
            memory_order_relaxed);
    }
    if (address >= params.addresses) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_RESIDENT_STATUS_INVALID],
            1u,
            memory_order_relaxed);
        return;
    }
    uint outer = gid / params.inner_length;
    uint key = outer * params.addresses + address;
    atomic_fetch_add_explicit(&counts[key], 1u, memory_order_relaxed);
}

kernel void solinas_bytecode_address_resident_summarize(
    device const atomic_uint* counts [[buffer(0)]],
    device uint* summary [[buffer(1)]],
    constant BytecodeAddressSparseParams& params [[buffer(2)]],
    uint address [[thread_position_in_grid]])
{
    if (address >= params.addresses) {
        return;
    }
    uint items = 0u;
    uint population = 0u;
    for (uint outer = 0u; outer < params.outer_length; outer++) {
        uint count = atomic_load_explicit(
            &counts[outer * params.addresses + address],
            memory_order_relaxed);
        population += count;
        items += (count + BYTECODE_ADDRESS_RESIDENT_WORK_ITEM_ROWS - 1u)
            / BYTECODE_ADDRESS_RESIDENT_WORK_ITEM_ROWS;
    }
    summary[address] = items;
    summary[params.addresses + address] = population;
}

kernel void solinas_bytecode_address_resident_layout(
    device atomic_uint* counts [[buffer(0)]],
    device BytecodeAddressSparseWorkItem* work_items [[buffer(1)]],
    device atomic_uint* item_cursors [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant BytecodeAddressSparseParams& params [[buffer(4)]],
    uint outer [[thread_position_in_grid]])
{
    if (outer >= params.outer_length) {
        return;
    }
    uint start = 0u;
    for (uint address = 0u; address < params.addresses; address++) {
        uint key = outer * params.addresses + address;
        uint count = atomic_load_explicit(&counts[key], memory_order_relaxed);
        atomic_store_explicit(&counts[key], start, memory_order_relaxed);
        for (uint local = 0u; local < count;
             local += BYTECODE_ADDRESS_RESIDENT_WORK_ITEM_ROWS) {
            uint item = atomic_fetch_add_explicit(
                &item_cursors[address],
                1u,
                memory_order_relaxed);
            if (item >= params.work_items
                || outer > 0xffffu
                || address > 0xffffu
                || start + local > 0xffffu) {
                atomic_fetch_add_explicit(
                    &status[BYTECODE_ADDRESS_RESIDENT_STATUS_INVALID],
                    1u,
                    memory_order_relaxed);
                continue;
            }
            BytecodeAddressSparseWorkItem value;
            value.address = (ushort)address;
            value.outer = (ushort)outer;
            value.start = (ushort)(start + local);
            value.count = (ushort)min(
                BYTECODE_ADDRESS_RESIDENT_WORK_ITEM_ROWS,
                count - local);
            work_items[item] = value;
        }
        start += count;
    }
    if (start != params.inner_length) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_RESIDENT_STATUS_INVALID],
            1u,
            memory_order_relaxed);
    }
}

kernel void solinas_bytecode_address_resident_scatter(
    device const ulong* rows [[buffer(0)]],
    device atomic_uint* cursors [[buffer(1)]],
    device ushort* occurrences [[buffer(2)]],
    device ulong* magnitudes [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant BytecodeAddressSparseParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.physical_rows) {
        return;
    }
    uint address = bytecode_address_resident_pc(rows, params, gid);
    if (address >= params.addresses) {
        return;
    }
    uint outer = gid / params.inner_length;
    uint inner = gid - outer * params.inner_length;
    uint key = outer * params.addresses + address;
    uint offset = atomic_fetch_add_explicit(
        &cursors[key],
        1u,
        memory_order_relaxed);
    if (offset >= params.inner_length) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_RESIDENT_STATUS_INVALID],
            1u,
            memory_order_relaxed);
        return;
    }
    uint output = outer * params.inner_length + offset;
    ulong metadata = booleanity_source_word(rows, params.physical_rows, 3u, gid);
    ushort negative = (ushort)((metadata >> BOOLEANITY_SOURCE_FUSED_SIGN_SHIFT) & 1ul);
    occurrences[output] = (ushort)inner | (negative << 15u);
    magnitudes[output] = booleanity_source_word(rows, params.physical_rows, 2u, gid);
}

inline SolinasFp128 bytecode_address_major_signed_product(
    SolinasFp128 coefficient,
    ulong magnitude,
    bool negative)
{
    return solinas_half_width_mul_signed_u64(
        coefficient,
        magnitude,
        negative);
}

kernel void solinas_bytecode_address_sparse_worker_packed_4_5_4(
    device const ushort* occurrences [[buffer(0)]],
    device const ulong* magnitudes [[buffer(1)]],
    device const BytecodeAddressSparseWorkItem* work_items [[buffer(2)]],
    device const SolinasFp128* e_lo [[buffer(3)]],
    device const SolinasFp128* e_hi [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant BytecodeAddressSparseParams& params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint item_index = group * BYTECODE_ADDRESS_PACKED_ITEMS_PER_GROUP + simdgroup;
    if (item_index >= params.work_items) {
        return;
    }
    BytecodeAddressSparseWorkItem item = work_items[item_index];
    uint outer = item.outer;
    uint stream_base = outer * params.inner_length + item.start;

    SolinasFp128 base[BYTECODE_ADDRESS_WORKER_BASE_STAGES];
    for (uint stage = 0u; stage < BYTECODE_ADDRESS_WORKER_BASE_STAGES; stage++) {
        base[stage] = solinas_zero();
    }
    for (uint offset = lane; offset < item.count; offset += BYTECODE_ADDRESS_WORKER_SIMD_WIDTH) {
        uint inner = occurrences[stream_base + offset] & 0x7fffu;
        for (uint stage = 0u; stage < BYTECODE_ADDRESS_WORKER_BASE_STAGES; stage++) {
            base[stage] = solinas_add(
                base[stage],
                e_lo[stage * params.inner_length + inner]);
        }
    }
    for (uint stage = 0u; stage < BYTECODE_ADDRESS_WORKER_BASE_STAGES; stage++) {
        SolinasFp128 sum = solinas_simd_sum_32(base[stage]);
        if (lane == 0u) {
            partials[stage * params.work_items + item_index] = solinas_mul_wide(
                sum,
                e_hi[stage * params.outer_length + outer]);
        }
    }

    SolinasFp128 fused[
        BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES];
    for (uint local_stage = 0u;
         local_stage < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES;
         local_stage++) {
        fused[local_stage] = solinas_zero();
    }
    for (uint offset = lane; offset < item.count; offset += BYTECODE_ADDRESS_WORKER_SIMD_WIDTH) {
        ushort occurrence = occurrences[stream_base + offset];
        uint inner = occurrence & 0x7fffu;
        bool negative = (occurrence >> 15u) != 0u;
        ulong magnitude = magnitudes[stream_base + offset];
        for (uint local_stage = 0u;
             local_stage < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES;
             local_stage++) {
            uint stage = BYTECODE_ADDRESS_WORKER_BASE_STAGES + local_stage;
            fused[local_stage] = solinas_add(
                fused[local_stage],
                bytecode_address_major_signed_product(
                    e_lo[stage * params.inner_length + inner],
                    magnitude,
                    negative));
        }
    }
    for (uint local_stage = 0u;
         local_stage < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES;
         local_stage++) {
        SolinasFp128 sum = solinas_simd_sum_32(fused[local_stage]);
        if (lane == 0u) {
            uint stage = BYTECODE_ADDRESS_WORKER_BASE_STAGES + local_stage;
            partials[stage * params.work_items + item_index] = solinas_mul_wide(
                sum,
                e_hi[stage * params.outer_length + outer]);
        }
    }
}

kernel void solinas_bytecode_address_sparse_reduce(
    device const SolinasFp128* partials [[buffer(0)]],
    device const uint* address_offsets [[buffer(1)]],
    device const SolinasFp128* padding [[buffer(2)]],
    device SolinasFp128* output [[buffer(3)]],
    constant BytecodeAddressSparseParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint output_fields = params.stages * params.addresses;
    if (gid >= output_fields) {
        return;
    }
    uint stage = gid / params.addresses;
    uint address = gid - stage * params.addresses;
    uint begin = address_offsets[address];
    uint end = address_offsets[address + 1u];
    SolinasFp128 total = solinas_zero();
    for (uint item = begin; item < end; item++) {
        total = solinas_add(total, partials[stage * params.work_items + item]);
    }
    if (address == 0u && stage < params.base_stages) {
        total = solinas_add(total, padding[stage]);
    }
    output[gid] = total;
}
