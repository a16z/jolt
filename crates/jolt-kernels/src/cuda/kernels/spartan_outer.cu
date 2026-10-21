__device__ __forceinline__ void so_split_eq(const u64 *__restrict__ e_in,
                                            const u64 *__restrict__ e_out, unsigned int in_bits,
                                            unsigned long long index, u64 *out) {
    u64 inner[LIMBS], outer[LIMBS];
    load4(e_in + (index & ((1ull << in_bits) - 1ull)) * LIMBS, inner);
    load4(e_out + (index >> in_bits) * LIMBS, outer);
    fr_mul(inner, outer, out);
}

__device__ __forceinline__ void so_block_reduce(u64 *scratch, unsigned int lanes,
                                                const u64 acc[][LIMBS], u64 *partials) {
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
