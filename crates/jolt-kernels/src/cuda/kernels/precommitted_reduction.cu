extern "C" __global__ void pcr_round_kernel(const u64 *__restrict__ packed, unsigned int table_len,
                                            unsigned int half, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    u64 acc[2][LIMBS] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    if (j < half) {
        const u64 *value = packed;
        const u64 *eq = packed + (unsigned long long)table_len * LIMBS;
        u64 v0[LIMBS], v1[LIMBS], e0[LIMBS], e1[LIMBS];
        load4(value + (unsigned long long)(2 * j) * LIMBS, v0);
        load4(value + (unsigned long long)(2 * j + 1) * LIMBS, v1);
        load4(eq + (unsigned long long)(2 * j) * LIMBS, e0);
        load4(eq + (unsigned long long)(2 * j + 1) * LIMBS, e1);
        fr_mul(v0, e0, acc[0]);
        u64 doubled[LIMBS], v2[LIMBS], e2[LIMBS];
        fr_add(v1, v1, doubled);
        fr_sub(doubled, v0, v2);
        fr_add(e1, e1, doubled);
        fr_sub(doubled, e0, e2);
        fr_mul(v2, e2, acc[1]);
    }
    lane_block_reduce(scratch, 2, acc, partials);
}

extern "C" __global__ void pcr_scatter_kernel(const unsigned int *__restrict__ indices,
                                              const u64 *__restrict__ values, unsigned int count,
                                              unsigned int table_len, unsigned int row,
                                              u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 value[LIMBS];
    load4(values + (unsigned long long)i * LIMBS, value);
    store4(out + ((unsigned long long)row * table_len + indices[i]) * LIMBS, value);
}

extern "C" __global__ void pcr_value_fold_kernel(const u64 *__restrict__ chunks,
                                                 const u64 *__restrict__ weights,
                                                 unsigned int chunk_count, unsigned int table_len,
                                                 u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_len) return;
    u64 acc[LIMBS] = {0, 0, 0, 0};
    for (unsigned int c = 0; c < chunk_count; c++) {
        u64 coeff[LIMBS], weight[LIMBS], term[LIMBS], sum[LIMBS];
        load4(chunks + ((unsigned long long)c * table_len + i) * LIMBS, coeff);
        load4(weights + (unsigned long long)c * LIMBS, weight);
        fr_mul(coeff, weight, term);
        fr_add(acc, term, sum);
        for (int l = 0; l < LIMBS; l++) acc[l] = sum[l];
    }
    store4(out + (unsigned long long)i * LIMBS, acc);
}

extern "C" __global__ void pcr_lane_eq_kernel(const u64 *__restrict__ lane_weights,
                                              const u64 *__restrict__ eq_cycle,
                                              unsigned int chunk_cycle_len,
                                              unsigned int lane_capacity, unsigned int lane_outer,
                                              unsigned int table_len, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_len) return;
    unsigned int lane;
    unsigned int cycle;
    if (lane_outer) {
        lane = i / chunk_cycle_len;
        cycle = i % chunk_cycle_len;
    } else {
        cycle = i / lane_capacity;
        lane = i % lane_capacity;
    }
    u64 weight[LIMBS], cycle_eval[LIMBS], product[LIMBS];
    load4(lane_weights + (unsigned long long)lane * LIMBS, weight);
    load4(eq_cycle + (unsigned long long)cycle * LIMBS, cycle_eval);
    fr_mul(weight, cycle_eval, product);
    store4(out + (unsigned long long)i * LIMBS, product);
}

extern "C" __global__ void pcr_shift_eq_kernel(const u64 *__restrict__ challenges_be,
                                               unsigned int num_vars, unsigned int start_index,
                                               unsigned int domain_mask, unsigned int table_len,
                                               u64 *__restrict__ out) {
    extern __shared__ u64 values[];
    const unsigned int lane = threadIdx.x;
    const unsigned int suffix_vars = __ffs(blockDim.x) - 1;
    const unsigned int base =
        ((start_index & ~(blockDim.x - 1)) + blockIdx.x * blockDim.x) & domain_mask;
    if (lane == 0) {
        u64 acc[LIMBS];
        load4(FR_ONE, acc);
        for (unsigned int bit = 0; bit < num_vars - suffix_vars; bit++) {
            u64 challenge[LIMBS], factor[LIMBS], product[LIMBS];
            load4(challenges_be + (unsigned long long)bit * LIMBS, challenge);
            if ((base >> (num_vars - 1 - bit)) & 1u) {
                load4(challenge, factor);
            } else {
                fr_sub(FR_ONE, challenge, factor);
            }
            fr_mul(acc, factor, product);
            load4(product, acc);
        }
        store4(values, acc);
    }
    __syncthreads();
    for (unsigned int level = 0; level < suffix_vars; level++) {
        const unsigned int half = 1u << level;
        if (lane < half) {
            u64 parent[LIMBS], challenge[LIMBS], left[LIMBS], right[LIMBS];
            load4(values + lane * LIMBS, parent);
            load4(challenges_be + (unsigned long long)(num_vars - 1 - level) * LIMBS,
                  challenge);
            fr_mul(parent, challenge, right);
            fr_sub(parent, right, left);
            store4(values + lane * LIMBS, left);
            store4(values + (lane + half) * LIMBS, right);
        }
        __syncthreads();
    }
    const unsigned int offset = (base + lane - start_index) & domain_mask;
    if (offset < table_len) {
        u64 value[LIMBS];
        load4(values + lane * LIMBS, value);
        store4(out + (unsigned long long)offset * LIMBS, value);
    }
}

template <bool NativeWords>
__device__ __forceinline__ void pcr_place_row(const u64 *__restrict__ src,
                                             const unsigned int *__restrict__ new_lsb_to_old_lsb,
                                             unsigned int num_vars, unsigned int permute,
                                             unsigned int src_row, unsigned int table_len,
                                             unsigned int dst_row, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_len) return;
    unsigned int source = i;
    if (permute) {
        source = 0;
        for (unsigned int new_lsb = 0; new_lsb < num_vars; new_lsb++) {
            source |= ((i >> new_lsb) & 1u) << new_lsb_to_old_lsb[new_lsb];
        }
    }
    u64 value[LIMBS];
    const unsigned long long index = (unsigned long long)src_row * table_len + source;
    if (NativeWords) {
        u64 raw[LIMBS] = {src[index], 0, 0, 0};
        fr_to_mont(raw, value);
    } else {
        load4(src + index * LIMBS, value);
    }
    store4(out + ((unsigned long long)dst_row * table_len + i) * LIMBS, value);
}

extern "C" __global__ void pcr_place_row_kernel(const u64 *__restrict__ src,
                                                const unsigned int *__restrict__ new_lsb_to_old_lsb,
                                                unsigned int num_vars, unsigned int permute,
                                                unsigned int src_row, unsigned int table_len,
                                                unsigned int dst_row, u64 *__restrict__ out) {
    pcr_place_row<false>(src, new_lsb_to_old_lsb, num_vars, permute, src_row, table_len, dst_row, out);
}

extern "C" __global__ void pcr_place_u64_row_kernel(const u64 *__restrict__ src,
                                                    const unsigned int *__restrict__ new_lsb_to_old_lsb,
                                                    unsigned int num_vars, unsigned int permute,
                                                    unsigned int src_row, unsigned int table_len,
                                                    unsigned int dst_row, u64 *__restrict__ out) {
    pcr_place_row<true>(src, new_lsb_to_old_lsb, num_vars, permute, src_row, table_len, dst_row, out);
}
