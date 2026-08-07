#if SOLINAS_OFFSET != 0xffffa7f7u
#error "RAM RAF successor requires the Akita Solinas offset"
#endif

#define RRS_ADDRESS_DOMAIN 8192u
#define RRS_INNER_LENGTH 32768u
#define RRS_TILE_ADDRESSES 1376u
#define RRS_TILE_COUNT 6u
#define RRS_ACCUMULATOR_WORDS 5u
#define RRS_DIRECT_THREADS 256u
#define RRS_BUCKET_THREADS 1024u
#define RRS_FINALIZE_THREADS 256u
#define RRS_STATUS_FLAGS 0u
#define RRS_STATUS_INVALID 1u
#define RRS_FLAG_UNSUPPORTED 1u
#define RRS_FLAG_INVALID_RECORD 2u

struct RamRafSuccessorAccessRecord {
    uint cycle;
    uint address;
};

struct RamRafSuccessorBucketDescriptor {
    uint first_record;
    uint record_count;
    uint outer;
    uint tile;
};

struct RamRafSuccessorDirectParams {
    uint record_count;
    uint rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint accumulator_words;
    uint threads;
    uint reserved;
};

struct RamRafSuccessorBucketedParams {
    uint descriptor_count;
    uint record_count;
    uint rows;
    uint addresses;
    uint inner_length;
    uint outer_length;
    uint tile_addresses;
    uint tiles;
    uint accumulator_words;
    uint threads;
    uint reserved[2];
};

struct RamRafSuccessorFinalizeParams {
    uint addresses;
    uint accumulator_words;
    uint threads;
    uint reserved;
};

inline bool rrs_is_zero(SolinasFp128 value) {
    return all(value.limb == uint4(0u));
}

inline void rrs_threadgroup_add(
    threadgroup atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * RRS_ACCUMULATOR_WORDS;
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

inline SolinasFp128 rrs_threadgroup_reduce(
    threadgroup atomic_uint* sums,
    uint field)
{
    uint base = field * RRS_ACCUMULATOR_WORDS;
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

inline void rrs_device_add(
    device atomic_uint* sums,
    uint field,
    SolinasFp128 value)
{
    uint base = field * RRS_ACCUMULATOR_WORDS;
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

inline SolinasFp128 rrs_device_reduce(
    device atomic_uint* sums,
    uint field)
{
    uint base = field * RRS_ACCUMULATOR_WORDS;
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

kernel void solinas_ram_raf_successor_direct(
    device const RamRafSuccessorAccessRecord* records [[buffer(0)]],
    device const SolinasFp128* e_lo [[buffer(1)]],
    device const SolinasFp128* e_hi [[buffer(2)]],
    device atomic_uint* output [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant RamRafSuccessorDirectParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint3 group_threads [[threads_per_threadgroup]])
{
    bool supported = params.inner_length == RRS_INNER_LENGTH
        && params.outer_length != 0u
        && params.rows / params.inner_length == params.outer_length
        && params.rows % params.inner_length == 0u
        && params.record_count <= params.rows
        && params.addresses == RRS_ADDRESS_DOMAIN
        && params.accumulator_words == RRS_ACCUMULATOR_WORDS
        && params.threads == RRS_DIRECT_THREADS
        && group_threads.x == RRS_DIRECT_THREADS
        && group_threads.y == 1u
        && group_threads.z == 1u
        && params.reserved == 0u;
    if (!supported) {
        if (gid == 0u) {
            atomic_fetch_or_explicit(
                &status[RRS_STATUS_FLAGS],
                RRS_FLAG_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    if (gid >= params.record_count) {
        return;
    }

    RamRafSuccessorAccessRecord record = records[gid];
    if (record.cycle >= params.rows || record.address >= params.addresses) {
        atomic_fetch_or_explicit(
            &status[RRS_STATUS_FLAGS],
            RRS_FLAG_INVALID_RECORD,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &status[RRS_STATUS_INVALID],
            1u,
            memory_order_relaxed);
        return;
    }
    uint inner = record.cycle & (RRS_INNER_LENGTH - 1u);
    uint outer = record.cycle / RRS_INNER_LENGTH;
    SolinasFp128 weight = solinas_mul_wide(e_lo[inner], e_hi[outer]);
    rrs_device_add(output, record.address, weight);
}

kernel void solinas_ram_raf_successor_bucketed(
    device const uint* records [[buffer(0)]],
    device const RamRafSuccessorBucketDescriptor* descriptors [[buffer(1)]],
    device const SolinasFp128* e_lo [[buffer(2)]],
    device const SolinasFp128* e_hi [[buffer(3)]],
    device atomic_uint* output [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant RamRafSuccessorBucketedParams& params [[buffer(6)]],
    threadgroup atomic_uint* local_sums [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_threads [[threads_per_threadgroup]])
{
    bool supported = params.inner_length == RRS_INNER_LENGTH
        && params.tiles == RRS_TILE_COUNT
        && params.outer_length != 0u
        && params.rows / params.inner_length == params.outer_length
        && params.rows % params.inner_length == 0u
        && params.record_count <= params.rows
        && params.descriptor_count <= params.outer_length * params.tiles
        && params.addresses == RRS_ADDRESS_DOMAIN
        && params.tile_addresses == RRS_TILE_ADDRESSES
        && params.accumulator_words == RRS_ACCUMULATOR_WORDS
        && params.threads == RRS_BUCKET_THREADS
        && group_threads.x == RRS_BUCKET_THREADS
        && group_threads.y == 1u
        && group_threads.z == 1u
        && params.reserved[0] == 0u
        && params.reserved[1] == 0u
        && group.y == 0u
        && group.z == 0u;
    if (!supported) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                &status[RRS_STATUS_FLAGS],
                RRS_FLAG_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    uint descriptor_index = group.x;
    if (descriptor_index >= params.descriptor_count) {
        return;
    }
    RamRafSuccessorBucketDescriptor descriptor = descriptors[descriptor_index];
    uint record_end = descriptor.first_record + descriptor.record_count;
    bool descriptor_valid = descriptor.record_count != 0u
        && record_end >= descriptor.first_record
        && record_end <= params.record_count
        && descriptor.outer < params.outer_length
        && descriptor.tile < params.tiles;
    if (!descriptor_valid) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                &status[RRS_STATUS_FLAGS],
                RRS_FLAG_INVALID_RECORD,
                memory_order_relaxed);
            atomic_fetch_add_explicit(
                &status[RRS_STATUS_INVALID],
                1u,
                memory_order_relaxed);
        }
        return;
    }

    uint tile_start = descriptor.tile * params.tile_addresses;
    uint active = min(params.tile_addresses, params.addresses - tile_start);
    uint local_words = active * RRS_ACCUMULATOR_WORDS;
    for (uint word = tid; word < local_words; word += RRS_BUCKET_THREADS) {
        atomic_store_explicit(&local_sums[word], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = descriptor.first_record + tid;
         index < record_end;
         index += RRS_BUCKET_THREADS)
    {
        uint packed = records[index];
        uint inner = packed & (RRS_INNER_LENGTH - 1u);
        uint local_address = (packed >> 15u) & 0x7ffu;
        uint reserved = packed >> 26u;
        if (reserved != 0u || local_address >= active) {
            atomic_fetch_or_explicit(
                &status[RRS_STATUS_FLAGS],
                RRS_FLAG_INVALID_RECORD,
                memory_order_relaxed);
            atomic_fetch_add_explicit(
                &status[RRS_STATUS_INVALID],
                1u,
                memory_order_relaxed);
            continue;
        }
        rrs_threadgroup_add(local_sums, local_address, e_lo[inner]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_address = tid;
         local_address < active;
         local_address += RRS_BUCKET_THREADS)
    {
        SolinasFp128 subtotal = rrs_threadgroup_reduce(local_sums, local_address);
        if (!rrs_is_zero(subtotal)) {
            SolinasFp128 weighted = solinas_mul_wide(subtotal, e_hi[descriptor.outer]);
            rrs_device_add(output, tile_start + local_address, weighted);
        }
    }
}

kernel void solinas_ram_raf_successor_finalize(
    device atomic_uint* sums [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant RamRafSuccessorFinalizeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint3 group_threads [[threads_per_threadgroup]])
{
    bool supported = params.addresses == RRS_ADDRESS_DOMAIN
        && params.accumulator_words == RRS_ACCUMULATOR_WORDS
        && params.threads == RRS_FINALIZE_THREADS
        && group_threads.x == RRS_FINALIZE_THREADS
        && group_threads.y == 1u
        && group_threads.z == 1u
        && params.reserved == 0u;
    if (!supported) {
        if (gid == 0u) {
            atomic_fetch_or_explicit(
                &status[RRS_STATUS_FLAGS],
                RRS_FLAG_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    if (gid < params.addresses) {
        output[gid] = rrs_device_reduce(sums, gid);
    }
}
