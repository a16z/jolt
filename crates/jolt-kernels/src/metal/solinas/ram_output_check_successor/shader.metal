// Concatenate after fp128.metal, half_width_probe/shader.metal, and
// simd_reduce.metal.

#define RAM_OUTPUT_SUCCESSOR_ADDRESSES 8192u
#define RAM_OUTPUT_SUCCESSOR_BLOCK_ELEMENTS 1024u
#define RAM_OUTPUT_SUCCESSOR_BLOCKS 8u
#define RAM_OUTPUT_SUCCESSOR_CHUNKS 8u
#define RAM_OUTPUT_SUCCESSOR_THREADS 128u
#define RAM_OUTPUT_SUCCESSOR_PARTIALS 64u
#define RAM_OUTPUT_SUCCESSOR_CHALLENGES 10u
#define RAM_OUTPUT_SUCCESSOR_HOST_WEIGHTS 0u
#define RAM_OUTPUT_SUCCESSOR_DEVICE_WEIGHTS 1u
#define RAM_OUTPUT_SUCCESSOR_STATUS_UNSUPPORTED 1u
#define RAM_OUTPUT_SUCCESSOR_STATUS_NONCANONICAL 4u

struct RamOutputSuccessorParams {
    uint addresses;
    uint block_elements;
    uint blocks;
    uint chunks_per_block;
    uint threads;
    uint weight_mode;
    uint reserved_0;
    uint reserved_1;
};

struct RamOutputReductionParams {
    uint input_count;
    uint blocks;
    uint chunks_per_block;
    uint reserved;
};

inline SolinasFp128 ram_output_successor_one() {
    SolinasFp128 value = solinas_zero();
    value.limb[0] = 1u;
    return value;
}

inline bool ram_output_successor_canonical(SolinasFp128 value) {
    return solinas_add_offset(value).carry == 0u;
}

inline bool ram_output_successor_params_valid(
    constant RamOutputSuccessorParams& params,
    uint threads,
    uint required_mode)
{
    return params.addresses == RAM_OUTPUT_SUCCESSOR_ADDRESSES
        && params.block_elements == RAM_OUTPUT_SUCCESSOR_BLOCK_ELEMENTS
        && params.blocks == RAM_OUTPUT_SUCCESSOR_BLOCKS
        && params.chunks_per_block == RAM_OUTPUT_SUCCESSOR_CHUNKS
        && params.threads == RAM_OUTPUT_SUCCESSOR_THREADS
        && params.weight_mode == required_mode
        && params.reserved_0 == 0u
        && params.reserved_1 == 0u
        && threads == RAM_OUTPUT_SUCCESSOR_THREADS
        && params.blocks * params.block_elements == params.addresses
        && params.chunks_per_block * params.threads
            == params.block_elements;
}

inline SolinasFp128 ram_output_successor_factor(
    SolinasFp128 challenge,
    uint low,
    uint bit)
{
    return ((low >> bit) & 1u) != 0u
        ? challenge
        : solinas_sub(ram_output_successor_one(), challenge);
}

inline SolinasFp128 ram_output_successor_direct_weight(
    device const SolinasFp128* challenges,
    uint low,
    thread bool& canonical)
{
    SolinasFp128 c0 = challenges[0];
    SolinasFp128 c1 = challenges[1];
    SolinasFp128 c2 = challenges[2];
    SolinasFp128 c3 = challenges[3];
    SolinasFp128 c4 = challenges[4];
    SolinasFp128 c5 = challenges[5];
    SolinasFp128 c6 = challenges[6];
    SolinasFp128 c7 = challenges[7];
    SolinasFp128 c8 = challenges[8];
    SolinasFp128 c9 = challenges[9];
    canonical = ram_output_successor_canonical(c0)
        && ram_output_successor_canonical(c1)
        && ram_output_successor_canonical(c2)
        && ram_output_successor_canonical(c3)
        && ram_output_successor_canonical(c4)
        && ram_output_successor_canonical(c5)
        && ram_output_successor_canonical(c6)
        && ram_output_successor_canonical(c7)
        && ram_output_successor_canonical(c8)
        && ram_output_successor_canonical(c9);

    SolinasFp128 weight = ram_output_successor_factor(c0, low, 0u);
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c1, low, 1u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c2, low, 2u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c3, low, 3u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c4, low, 4u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c5, low, 5u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c6, low, 6u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c7, low, 7u));
    weight = solinas_mul_wide(
        weight, ram_output_successor_factor(c8, low, 8u));
    return solinas_mul_wide(
        weight, ram_output_successor_factor(c9, low, 9u));
}

inline void ram_output_successor_write_partial(
    SolinasFp128 value,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint partial,
    uint lane,
    uint simdgroup)
{
    value = solinas_simd_sum_32(value);
    if (lane == 0u) {
        shared[simdgroup] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        value = lane < 4u ? shared[lane] : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            partials[partial] = value;
        }
    }
}

kernel void solinas_ram_output_successor_partials_host_weights(
    device const ulong* val_final [[buffer(0)]],
    device const SolinasFp128* weights [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant RamOutputSuccessorParams& params [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint partial [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (!ram_output_successor_params_valid(
            params, threads, RAM_OUTPUT_SUCCESSOR_HOST_WEIGHTS)) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                status,
                RAM_OUTPUT_SUCCESSOR_STATUS_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    if (partial >= RAM_OUTPUT_SUCCESSOR_PARTIALS) {
        return;
    }

    uint block = partial / RAM_OUTPUT_SUCCESSOR_CHUNKS;
    uint chunk = partial - block * RAM_OUTPUT_SUCCESSOR_CHUNKS;
    uint low = chunk * RAM_OUTPUT_SUCCESSOR_THREADS + tid;
    uint index = block * RAM_OUTPUT_SUCCESSOR_BLOCK_ELEMENTS + low;
    SolinasFp128 weight = weights[low];
    if (!ram_output_successor_canonical(weight)) {
        atomic_fetch_or_explicit(
            status,
            RAM_OUTPUT_SUCCESSOR_STATUS_NONCANONICAL,
            memory_order_relaxed);
    }
    SolinasFp128 value = solinas_half_width_mul_u64(
        weight, val_final[index]);
    ram_output_successor_write_partial(
        value, partials, shared, partial, lane, simdgroup);
}

kernel void solinas_ram_output_successor_partials_device_weights(
    device const ulong* val_final [[buffer(0)]],
    device const SolinasFp128* challenges [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant RamOutputSuccessorParams& params [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint partial [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (!ram_output_successor_params_valid(
            params, threads, RAM_OUTPUT_SUCCESSOR_DEVICE_WEIGHTS)) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                status,
                RAM_OUTPUT_SUCCESSOR_STATUS_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    if (partial >= RAM_OUTPUT_SUCCESSOR_PARTIALS) {
        return;
    }

    uint block = partial / RAM_OUTPUT_SUCCESSOR_CHUNKS;
    uint chunk = partial - block * RAM_OUTPUT_SUCCESSOR_CHUNKS;
    uint low = chunk * RAM_OUTPUT_SUCCESSOR_THREADS + tid;
    uint index = block * RAM_OUTPUT_SUCCESSOR_BLOCK_ELEMENTS + low;
    bool canonical = false;
    SolinasFp128 weight = ram_output_successor_direct_weight(
        challenges, low, canonical);
    if (!canonical) {
        atomic_fetch_or_explicit(
            status,
            RAM_OUTPUT_SUCCESSOR_STATUS_NONCANONICAL,
            memory_order_relaxed);
    }
    SolinasFp128 value = solinas_half_width_mul_u64(
        weight, val_final[index]);
    ram_output_successor_write_partial(
        value, partials, shared, partial, lane, simdgroup);
}

kernel void solinas_ram_output_successor_reduce8(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamOutputReductionParams& params [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    bool supported = params.input_count == RAM_OUTPUT_SUCCESSOR_PARTIALS
        && params.blocks == RAM_OUTPUT_SUCCESSOR_BLOCKS
        && params.chunks_per_block == RAM_OUTPUT_SUCCESSOR_CHUNKS
        && params.reserved == 0u
        && threads == 32u;
    if (!supported) {
        if (tid == 0u) {
            atomic_fetch_or_explicit(
                status,
                RAM_OUTPUT_SUCCESSOR_STATUS_UNSUPPORTED,
                memory_order_relaxed);
        }
        return;
    }
    if (tid >= RAM_OUTPUT_SUCCESSOR_BLOCKS) {
        return;
    }
    uint start = tid * RAM_OUTPUT_SUCCESSOR_CHUNKS;
    SolinasFp128 sum = partials[start];
    sum = solinas_add(sum, partials[start + 1u]);
    sum = solinas_add(sum, partials[start + 2u]);
    sum = solinas_add(sum, partials[start + 3u]);
    sum = solinas_add(sum, partials[start + 4u]);
    sum = solinas_add(sum, partials[start + 5u]);
    sum = solinas_add(sum, partials[start + 6u]);
    output[tid] = solinas_add(sum, partials[start + 7u]);
}
