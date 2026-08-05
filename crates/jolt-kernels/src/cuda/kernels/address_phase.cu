#define AP_CHUNK_LEN 8
#define AP_CHUNK_SIZE 256
#define AP_RAF_LANES 6
#define AP_MAX_SUFFIXES 4
#define AP_NO_TABLE 0xFFFFFFFFu
#define AP_SKIP 0xFFFFFFFFu

__device__ __forceinline__ u128 ap_index(const unsigned long long *__restrict__ bits,
                                         unsigned int j) {
    return ((u128)bits[2 * j + 1] << 64) | (u128)bits[2 * j];
}

__device__ __forceinline__ unsigned int ap_chunk(u128 index, unsigned int suffix_len) {
    return (unsigned int)((index >> suffix_len) & (u128)(AP_CHUNK_SIZE - 1));
}

extern "C" __global__ void ap_raf_keys_kernel(const unsigned long long *__restrict__ lookup_index,
                                              unsigned int suffix_len,
                                              unsigned int *__restrict__ keys,
                                              unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    keys[j] = ap_chunk(ap_index(lookup_index, j), suffix_len);
}

extern "C" __global__ void ap_table_keys_kernel(const unsigned long long *__restrict__ lookup_index,
                                                const unsigned int *__restrict__ table_index,
                                                const unsigned int *__restrict__ table_slots,
                                                unsigned int table_count,
                                                unsigned int suffix_len,
                                                unsigned int *__restrict__ keys,
                                                unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int table = table_index[j];
    if (table == AP_NO_TABLE || table >= table_count) {
        keys[j] = AP_SKIP;
        return;
    }
    unsigned int slot = table_slots[table];
    if (slot == AP_SKIP) {
        keys[j] = AP_SKIP;
        return;
    }
    keys[j] = slot * AP_CHUNK_SIZE + ap_chunk(ap_index(lookup_index, j), suffix_len);
}

extern "C" __global__ void ap_histogram_kernel(const unsigned int *__restrict__ keys,
                                               unsigned int *__restrict__ counts,
                                               unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int key = keys[j];
    if (key == AP_SKIP) return;
    atomicAdd(&counts[key], 1u);
}

extern "C" __global__ void ap_scatter_kernel(const unsigned int *__restrict__ keys,
                                             const unsigned int *__restrict__ offsets,
                                             unsigned int *__restrict__ cursors,
                                             unsigned int *__restrict__ order,
                                             unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int key = keys[j];
    if (key == AP_SKIP) return;
    unsigned int slot = atomicAdd(&cursors[key], 1u);
    order[offsets[key] + slot] = j;
}

__device__ __forceinline__ void ap_block_reduce(u64 *scratch, u64 *acc) {
    unsigned int tid = threadIdx.x;
    store4(scratch + tid * LIMBS, acc);
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            u64 a[LIMBS], b[LIMBS], sum[LIMBS];
            load4(scratch + tid * LIMBS, a);
            load4(scratch + (tid + stride) * LIMBS, b);
            fr_add(a, b, sum);
            store4(scratch + tid * LIMBS, sum);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ap_raf_reduce_kernel(const unsigned int *__restrict__ order,
                                                const unsigned int *__restrict__ offsets,
                                                const unsigned int *__restrict__ counts,
                                                const unsigned long long *__restrict__ lookup_index,
                                                const unsigned char *__restrict__ raf_flags,
                                                const u64 *__restrict__ u_evals,
                                                unsigned int suffix_len,
                                                unsigned int upper_suffix_bits,
                                                unsigned int canonical,
                                                u64 *__restrict__ buckets) {
    extern __shared__ u64 scratch[];
    unsigned int bucket = blockIdx.x;
    unsigned int start = offsets[bucket];
    unsigned int count = counts[bucket];

    u64 acc[AP_RAF_LANES][LIMBS];
    for (int lane = 0; lane < AP_RAF_LANES; lane++) {
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;
    }

    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        unsigned int j = order[start + i];
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)j * LIMBS, u);
        u128 index = ap_index(lookup_index, j);
        u128 suffix_bits = index & sfx_mask(suffix_len);

        if (canonical != 0u && raf_flags[j] != 0u) {
            u128 ones = ((u128)1 << upper_suffix_bits) - 1;
            if (upper_suffix_bits == 0u ||
                (suffix_bits >> (suffix_len - upper_suffix_bits)) == ones) {
                fr_add(acc[5], u, acc[5]);
            }
        }

        if (raf_flags[j] == 0u) {
            fr_add(acc[0], u, acc[0]);
            sfx_bits whole = sfx_new(suffix_bits, suffix_len);
            sfx_bits left, right;
            sfx_uninterleave(whole, &left, &right);
            unsigned long long left_value = sfx_u64(left);
            if (left_value != 0ULL) {
                u64 mont[LIMBS], term[LIMBS];
                u64 raw[LIMBS] = {left_value, 0, 0, 0};
                fr_to_mont(raw, mont);
                fr_mul(u, mont, term);
                fr_add(acc[2], term, acc[2]);
            }
            unsigned long long right_value = sfx_u64(right);
            if (right_value != 0ULL) {
                u64 mont[LIMBS], term[LIMBS];
                u64 raw[LIMBS] = {right_value, 0, 0, 0};
                fr_to_mont(raw, mont);
                fr_mul(u, mont, term);
                fr_add(acc[3], term, acc[3]);
            }
        } else {
            fr_add(acc[1], u, acc[1]);
            if (suffix_bits != 0) {
                u64 mont[LIMBS], term[LIMBS];
                u64 raw[LIMBS] = {(unsigned long long)suffix_bits,
                                  (unsigned long long)(suffix_bits >> 64), 0, 0};
                fr_to_mont(raw, mont);
                fr_mul(u, mont, term);
                fr_add(acc[4], term, acc[4]);
            }
        }
    }

    for (int lane = 0; lane < AP_RAF_LANES; lane++) {
        ap_block_reduce(scratch, acc[lane]);
        if (threadIdx.x == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(buckets + ((unsigned long long)lane * AP_CHUNK_SIZE + bucket) * LIMBS, total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ap_suffix_reduce_kernel(
    const unsigned int *__restrict__ order,
    const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ counts,
    const unsigned long long *__restrict__ lookup_index,
    const u64 *__restrict__ u_evals,
    const unsigned int *__restrict__ suffix_ids,
    const unsigned int *__restrict__ suffix_offsets,
    const unsigned int *__restrict__ suffix_counts,
    unsigned int suffix_len,
    u64 *__restrict__ buckets) {
    extern __shared__ u64 scratch[];
    unsigned int slot = blockIdx.x / AP_CHUNK_SIZE;
    unsigned int bucket = blockIdx.x % AP_CHUNK_SIZE;
    unsigned int start = offsets[blockIdx.x];
    unsigned int count = counts[blockIdx.x];
    unsigned int families = suffix_counts[slot];
    unsigned int family_base = suffix_offsets[slot];

    u64 acc[AP_MAX_SUFFIXES][LIMBS];
    for (int s = 0; s < AP_MAX_SUFFIXES; s++) {
        for (int l = 0; l < LIMBS; l++) acc[s][l] = 0;
    }

    for (unsigned int i = threadIdx.x; i < count; i += blockDim.x) {
        unsigned int j = order[start + i];
        u64 u[LIMBS];
        load4(u_evals + (unsigned long long)j * LIMBS, u);
        u128 index = ap_index(lookup_index, j);
        sfx_bits suffix = sfx_new(index & sfx_mask(suffix_len), suffix_len);

        for (unsigned int s = 0; s < families; s++) {
            unsigned long long value = sfx_eval(suffix_ids[family_base + s], suffix);
            if (value == 0ULL) continue;
            u64 term[LIMBS];
            if (value == 1ULL) {
                fr_add(acc[s], u, acc[s]);
                continue;
            }
            u64 mont[LIMBS];
            u64 raw[LIMBS] = {value, 0, 0, 0};
            fr_to_mont(raw, mont);
            fr_mul(u, mont, term);
            fr_add(acc[s], term, acc[s]);
        }
    }

    for (unsigned int s = 0; s < families; s++) {
        ap_block_reduce(scratch, acc[s]);
        if (threadIdx.x == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(buckets + ((unsigned long long)(family_base + s) * AP_CHUNK_SIZE + bucket) * LIMBS,
                   total);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ap_scale_shift_kernel(u64 *__restrict__ buckets,
                                                 const u64 *__restrict__ half_scale,
                                                 const u64 *__restrict__ full_scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= AP_CHUNK_SIZE) return;
    u64 scale[LIMBS], value[LIMBS], scaled[LIMBS];
    load4(half_scale, scale);
    load4(buckets + (unsigned long long)i * LIMBS, value);
    fr_mul(value, scale, scaled);
    store4(buckets + (unsigned long long)i * LIMBS, scaled);
    load4(full_scale, scale);
    load4(buckets + ((unsigned long long)AP_CHUNK_SIZE + i) * LIMBS, value);
    fr_mul(value, scale, scaled);
    store4(buckets + ((unsigned long long)AP_CHUNK_SIZE + i) * LIMBS, scaled);
}

extern "C" __global__ void ap_condense_kernel(const unsigned long long *__restrict__ lookup_index,
                                              u64 *__restrict__ u_evals,
                                              const u64 *__restrict__ v_prev,
                                              unsigned int suffix_len,
                                              unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int chunk = ap_chunk(ap_index(lookup_index, j), suffix_len);
    u64 u[LIMBS], v[LIMBS], product[LIMBS];
    load4(u_evals + (unsigned long long)j * LIMBS, u);
    load4(v_prev + (unsigned long long)chunk * LIMBS, v);
    fr_mul(u, v, product);
    store4(u_evals + (unsigned long long)j * LIMBS, product);
}
