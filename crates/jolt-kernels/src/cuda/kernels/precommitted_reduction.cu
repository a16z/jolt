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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_len) return;
    unsigned int index = (start_index + i) & domain_mask;
    u64 acc[LIMBS];
    load4(FR_ONE, acc);
    for (unsigned int bit = 0; bit < num_vars; bit++) {
        u64 challenge[LIMBS], factor[LIMBS], product[LIMBS];
        load4(challenges_be + (unsigned long long)bit * LIMBS, challenge);
        unsigned int shift = num_vars - 1 - bit;
        if ((index >> shift) & 1u) {
            for (int l = 0; l < LIMBS; l++) factor[l] = challenge[l];
        } else {
            u64 one[LIMBS];
            load4(FR_ONE, one);
            fr_sub(one, challenge, factor);
        }
        fr_mul(acc, factor, product);
        for (int l = 0; l < LIMBS; l++) acc[l] = product[l];
    }
    store4(out + (unsigned long long)i * LIMBS, acc);
}

extern "C" __global__ void pcr_place_row_kernel(const u64 *__restrict__ src,
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
    load4(src + ((unsigned long long)src_row * table_len + source) * LIMBS, value);
    store4(out + ((unsigned long long)dst_row * table_len + i) * LIMBS, value);
}
