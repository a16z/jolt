__device__ __forceinline__ void eq_split_weight(const u64 *__restrict__ e_in,
                                                unsigned int e_in_len,
                                                const u64 *__restrict__ e_out,
                                                unsigned int num_x_in_bits, unsigned long long g,
                                                u64 *combined) {
    u64 weight[LIMBS];
    if (e_in_len <= 1) {
        load4(FR_ONE, weight);
    } else {
        unsigned long long x_in = g & ((1ull << num_x_in_bits) - 1ull);
        load4(e_in + x_in * LIMBS, weight);
    }
    u64 e_out_eval[LIMBS];
    load4(e_out + (g >> num_x_in_bits) * LIMBS, e_out_eval);
    fr_mul(weight, e_out_eval, combined);
}

__device__ __forceinline__ void lane_block_reduce(u64 *scratch, unsigned int lanes,
                                                  u64 (*acc)[LIMBS],
                                                  u64 *__restrict__ partials) {
    unsigned int tid = threadIdx.x;
    for (unsigned int lane = 0; lane < lanes; lane++) {
        store4(scratch + tid * LIMBS, acc[lane]);
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                u64 a[LIMBS], b[LIMBS], s[LIMBS];
                load4(scratch + tid * LIMBS, a);
                load4(scratch + (tid + stride) * LIMBS, b);
                fr_add(a, b, s);
                store4(scratch + tid * LIMBS, s);
            }
            __syncthreads();
        }
        if (tid == 0) {
            u64 total[LIMBS];
            load4(scratch, total);
            store4(partials + ((unsigned long long)lane * gridDim.x + blockIdx.x) * LIMBS, total);
        }
        __syncthreads();
    }
}
