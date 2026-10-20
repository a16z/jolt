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
