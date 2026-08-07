#if SOLINAS_OFFSET != 0xffffa7f7u
#error "bytecode address worker requires the Akita Solinas offset"
#endif

#define BYTECODE_ADDRESS_WORKER_STAGES 9u
#define BYTECODE_ADDRESS_WORKER_BASE_STAGES 5u
#define BYTECODE_ADDRESS_WORKER_SIMD_WIDTH 32u
#define BYTECODE_ADDRESS_WORKER_SIMDGROUPS 8u
#define BYTECODE_ADDRESS_PRODUCER_THREADS 1024u
#define BYTECODE_ADDRESS_PRODUCER_BINS_PER_THREAD 8u
#define BYTECODE_ADDRESS_PRODUCER_INNER_LENGTH 32768u
#define BYTECODE_ADDRESS_PRODUCER_INVALID_ROWS 0u
#define BYTECODE_ADDRESS_PRODUCER_COMPLETED_OUTERS 1u
#define BYTECODE_ADDRESS_PRODUCER_EMITTED_ROWS 2u

struct BytecodeAddressMajorRowWords {
    ulong lookup_lo;
    ulong lookup_hi;
    ulong ram_address_plus_one;
    ulong fused_inc_magnitude;
    ulong packed_pc_and_flags;
};

struct BytecodeAddressMajorParams {
    uint rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint outer_tiles;
    uint stages;
    uint base_stages;
    uint reserved;
};

inline uint bytecode_address_major_push_pc(
    BytecodeAddressMajorRowWords row,
    uint addresses,
    thread bool& valid)
{
    ulong plus_one = row.packed_pc_and_flags & 0x00fffffffffffffful;
    ulong pc = plus_one == 0ul ? 0ul : plus_one - 1ul;
    valid = pc < (ulong)addresses;
    return valid ? (uint)pc : 0u;
}

inline uint bytecode_address_major_simd_inclusive_sum(uint value, uint lane) {
    for (ushort offset = 1; offset < BYTECODE_ADDRESS_WORKER_SIMD_WIDTH; offset <<= 1) {
        uint previous = simd_shuffle_up(value, offset);
        if (lane >= (uint)offset) {
            value += previous;
        }
    }
    return value;
}

inline uint bytecode_address_major_threadgroup_exclusive_sum(
    uint local_total,
    uint lane,
    uint simdgroup,
    threadgroup atomic_uint* scratch,
    thread uint& total)
{
    uint simd_inclusive = bytecode_address_major_simd_inclusive_sum(local_total, lane);
    if (lane == BYTECODE_ADDRESS_WORKER_SIMD_WIDTH - 1u) {
        atomic_store_explicit(
            &scratch[simdgroup],
            simd_inclusive,
            memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        uint group_total = atomic_load_explicit(&scratch[lane], memory_order_relaxed);
        uint group_inclusive = bytecode_address_major_simd_inclusive_sum(group_total, lane);
        atomic_store_explicit(
            &scratch[lane],
            group_inclusive - group_total,
            memory_order_relaxed);
        if (lane == BYTECODE_ADDRESS_WORKER_SIMD_WIDTH - 1u) {
            atomic_store_explicit(&scratch[32], group_inclusive, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint simd_base = atomic_load_explicit(&scratch[simdgroup], memory_order_relaxed);
    total = atomic_load_explicit(&scratch[32], memory_order_relaxed);
    return simd_base + simd_inclusive - local_total;
}

kernel void solinas_bytecode_address_major_build_resident(
    device const BytecodeAddressMajorRowWords* rows [[buffer(0)]],
    device uint* cells [[buffer(1)]],
    device uint* inner_sign [[buffer(2)]],
    device ulong* magnitude [[buffer(3)]],
    constant BytecodeAddressMajorParams& params [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    threadgroup atomic_uint* bins [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint outer [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool supported = threads == BYTECODE_ADDRESS_PRODUCER_THREADS
        && params.addresses == BYTECODE_ADDRESS_PRODUCER_THREADS
            * BYTECODE_ADDRESS_PRODUCER_BINS_PER_THREAD
        && params.inner_length == BYTECODE_ADDRESS_PRODUCER_INNER_LENGTH
        && params.rows == params.inner_length * params.outer_length
        && outer < params.outer_length;
    if (!supported) {
        if (tid == 0u && outer < params.outer_length) {
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_PRODUCER_INVALID_ROWS],
                1u,
                memory_order_relaxed);
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_PRODUCER_COMPLETED_OUTERS],
                1u,
                memory_order_relaxed);
        }
        return;
    }

    for (uint address = tid; address < params.addresses; address += threads) {
        atomic_store_explicit(&bins[address], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint outer_base = outer * params.inner_length;
    for (uint inner = tid; inner < params.inner_length; inner += threads) {
        uint row_index = outer_base + inner;
        bool valid;
        uint address = bytecode_address_major_push_pc(rows[row_index], params.addresses, valid);
        if (!valid) {
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_PRODUCER_INVALID_ROWS],
                1u,
                memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&bins[address], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_counts[BYTECODE_ADDRESS_PRODUCER_BINS_PER_THREAD];
    uint local_total = 0u;
    uint first_address = tid * BYTECODE_ADDRESS_PRODUCER_BINS_PER_THREAD;
    for (uint i = 0u; i < BYTECODE_ADDRESS_PRODUCER_BINS_PER_THREAD; i++) {
        uint count = atomic_load_explicit(&bins[first_address + i], memory_order_relaxed);
        local_counts[i] = count;
        local_total += count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint occurrence_total;
    uint running = bytecode_address_major_threadgroup_exclusive_sum(
        local_total, lane, simdgroup, bins, occurrence_total);
    if (tid == 0u && occurrence_total != params.inner_length) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_PRODUCER_INVALID_ROWS],
            1u,
            memory_order_relaxed);
    }
    for (uint i = 0u; i < BYTECODE_ADDRESS_PRODUCER_BINS_PER_THREAD; i++) {
        uint address = first_address + i;
        uint count = local_counts[i];
        cells[address * params.outer_length + outer] = running | (count << 16u);
        atomic_store_explicit(&bins[address], running, memory_order_relaxed);
        running += count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint inner = tid; inner < params.inner_length; inner += threads) {
        uint row_index = outer_base + inner;
        BytecodeAddressMajorRowWords row = rows[row_index];
        bool valid;
        uint address = bytecode_address_major_push_pc(row, params.addresses, valid);
        (void)valid;
        uint destination = atomic_fetch_add_explicit(
            &bins[address],
            1u,
            memory_order_relaxed);
        if (destination < params.inner_length) {
            uint negative = (uint)(row.packed_pc_and_flags >> 63u);
            inner_sign[outer_base + destination] = inner | (negative << 31u);
            magnitude[outer_base + destination] = row.fused_inc_magnitude;
        } else {
            atomic_fetch_add_explicit(
                &status[BYTECODE_ADDRESS_PRODUCER_INVALID_ROWS],
                1u,
                memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_PRODUCER_COMPLETED_OUTERS],
            1u,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &status[BYTECODE_ADDRESS_PRODUCER_EMITTED_ROWS],
            occurrence_total,
            memory_order_relaxed);
    }
}

inline SolinasFp128 bytecode_address_major_from_u64(ulong value) {
    SolinasFp128 result = solinas_zero();
    result.limb.x = (uint)value;
    result.limb.y = (uint)(value >> 32u);
    return result;
}

inline SolinasFp128 bytecode_address_major_signed_product(
    SolinasFp128 coefficient,
    ulong magnitude,
    bool negative)
{
    SolinasFp128 product = solinas_mul_wide(
        coefficient,
        bytecode_address_major_from_u64(magnitude));
    return negative ? solinas_sub(solinas_zero(), product) : product;
}

inline SolinasFp128 bytecode_address_major_broadcast_zero(
    SolinasFp128 value)
{
    value.limb = simd_broadcast(value.limb, 0u);
    return value;
}

inline void bytecode_address_major_store_scratch(
    threadgroup uint* scratch,
    uint simdgroup,
    uint stage,
    SolinasFp128 value)
{
    uint base = (simdgroup * BYTECODE_ADDRESS_WORKER_BASE_STAGES + stage) * 4u;
    scratch[base + 0u] = value.limb.x;
    scratch[base + 1u] = value.limb.y;
    scratch[base + 2u] = value.limb.z;
    scratch[base + 3u] = value.limb.w;
}

inline SolinasFp128 bytecode_address_major_load_scratch(
    threadgroup uint* scratch,
    uint simdgroup,
    uint stage)
{
    uint base = (simdgroup * BYTECODE_ADDRESS_WORKER_BASE_STAGES + stage) * 4u;
    SolinasFp128 value;
    value.limb = uint4(
        scratch[base + 0u],
        scratch[base + 1u],
        scratch[base + 2u],
        scratch[base + 3u]);
    return value;
}

kernel void solinas_bytecode_address_major_worker_5_4(
    device const uint* cells [[buffer(0)]],
    device const uint* inner_sign [[buffer(1)]],
    device const ulong* magnitude [[buffer(2)]],
    device const SolinasFp128* e_lo [[buffer(3)]],
    device const SolinasFp128* e_hi [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant BytecodeAddressMajorParams& params [[buffer(6)]],
    threadgroup uint* scratch [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    uint address = group / params.outer_tiles;
    uint tile = group - address * params.outer_tiles;
    uint outers_per_tile = (params.outer_length + params.outer_tiles - 1u)
        / params.outer_tiles;
    uint outer_begin = tile * outers_per_tile;
    uint outer_end = min(params.outer_length, outer_begin + outers_per_tile);
    uint inner_mask = params.inner_length - 1u;
    uint output_fields = params.stages * params.addresses;

    SolinasFp128 owned = solinas_zero();
    for (uint outer = outer_begin + simdgroup;
         outer < outer_end;
         outer += BYTECODE_ADDRESS_WORKER_SIMDGROUPS) {
        uint packed_cell = cells[address * params.outer_length + outer];
        uint start = packed_cell & 0xffffu;
        uint count = packed_cell >> 16u;
        if (count == 0u) {
            continue;
        }
        SolinasFp128 sums[BYTECODE_ADDRESS_WORKER_BASE_STAGES];
        for (uint stage = 0u; stage < BYTECODE_ADDRESS_WORKER_BASE_STAGES; stage++) {
            sums[stage] = solinas_zero();
        }
        for (uint offset = lane; offset < count; offset += BYTECODE_ADDRESS_WORKER_SIMD_WIDTH) {
            uint stream = outer * params.inner_length + start + offset;
            uint inner = inner_sign[stream] & inner_mask;
            for (uint stage = 0u; stage < BYTECODE_ADDRESS_WORKER_BASE_STAGES; stage++) {
                sums[stage] = solinas_add(
                    sums[stage],
                    e_lo[stage * params.inner_length + inner]);
            }
        }
        for (uint stage = 0u; stage < BYTECODE_ADDRESS_WORKER_BASE_STAGES; stage++) {
            SolinasFp128 sum = bytecode_address_major_broadcast_zero(
                solinas_simd_sum_32(sums[stage]));
            if (lane == stage) {
                owned = solinas_add(
                    owned,
                    solinas_mul_wide(
                        sum,
                        e_hi[stage * params.outer_length + outer]));
            }
        }
    }
    if (lane < BYTECODE_ADDRESS_WORKER_BASE_STAGES) {
        bytecode_address_major_store_scratch(scratch, simdgroup, lane, owned);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u && lane < BYTECODE_ADDRESS_WORKER_BASE_STAGES) {
        SolinasFp128 total = solinas_zero();
        for (uint source = 0u; source < BYTECODE_ADDRESS_WORKER_SIMDGROUPS; source++) {
            total = solinas_add(
                total,
                bytecode_address_major_load_scratch(scratch, source, lane));
        }
        partials[tile * output_fields + lane * params.addresses + address] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    owned = solinas_zero();
    for (uint outer = outer_begin + simdgroup;
         outer < outer_end;
         outer += BYTECODE_ADDRESS_WORKER_SIMDGROUPS) {
        uint packed_cell = cells[address * params.outer_length + outer];
        uint start = packed_cell & 0xffffu;
        uint count = packed_cell >> 16u;
        if (count == 0u) {
            continue;
        }
        SolinasFp128 sums[BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES];
        for (uint local_stage = 0u;
             local_stage < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES;
             local_stage++) {
            sums[local_stage] = solinas_zero();
        }
        for (uint offset = lane; offset < count; offset += BYTECODE_ADDRESS_WORKER_SIMD_WIDTH) {
            uint stream = outer * params.inner_length + start + offset;
            uint packed_inner = inner_sign[stream];
            uint inner = packed_inner & inner_mask;
            bool negative = (packed_inner >> 31u) != 0u;
            ulong row_magnitude = magnitude[stream];
            for (uint local_stage = 0u;
                 local_stage < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES;
                 local_stage++) {
                uint stage = BYTECODE_ADDRESS_WORKER_BASE_STAGES + local_stage;
                sums[local_stage] = solinas_add(
                    sums[local_stage],
                    bytecode_address_major_signed_product(
                        e_lo[stage * params.inner_length + inner],
                        row_magnitude,
                        negative));
            }
        }
        for (uint local_stage = 0u;
             local_stage < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES;
             local_stage++) {
            uint stage = BYTECODE_ADDRESS_WORKER_BASE_STAGES + local_stage;
            SolinasFp128 sum = bytecode_address_major_broadcast_zero(
                solinas_simd_sum_32(sums[local_stage]));
            if (lane == local_stage) {
                owned = solinas_add(
                    owned,
                    solinas_mul_wide(
                        sum,
                        e_hi[stage * params.outer_length + outer]));
            }
        }
    }
    if (lane < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES) {
        bytecode_address_major_store_scratch(scratch, simdgroup, lane, owned);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u
        && lane < BYTECODE_ADDRESS_WORKER_STAGES - BYTECODE_ADDRESS_WORKER_BASE_STAGES) {
        SolinasFp128 total = solinas_zero();
        for (uint source = 0u; source < BYTECODE_ADDRESS_WORKER_SIMDGROUPS; source++) {
            total = solinas_add(
                total,
                bytecode_address_major_load_scratch(scratch, source, lane));
        }
        uint stage = BYTECODE_ADDRESS_WORKER_BASE_STAGES + lane;
        partials[tile * output_fields + stage * params.addresses + address] = total;
    }
}

kernel void solinas_bytecode_address_major_reduce_tiles(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant BytecodeAddressMajorParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint output_fields = params.stages * params.addresses;
    if (gid >= output_fields) {
        return;
    }
    SolinasFp128 total = solinas_zero();
    for (uint tile = 0u; tile < params.outer_tiles; tile++) {
        total = solinas_add(total, partials[tile * output_fields + gid]);
    }
    output[gid] = total;
}
