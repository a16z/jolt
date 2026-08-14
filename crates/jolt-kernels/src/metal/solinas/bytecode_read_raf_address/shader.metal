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
