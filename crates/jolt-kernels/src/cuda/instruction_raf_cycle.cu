extern "C" __global__ void instruction_raf_cycle_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ factors,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long pair_stride,
    unsigned long half,
    unsigned int in_bits
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (9 * 4);
    for (int k = 0; k < 9 * 4; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        // STUB: real body builds the 9 degree-product evals into acc[e*4..].
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (9 * 4);
            for (int e = 0; e < 9; e++) {
                u64 s[4];
                fr_add(acc + e * 4, other + e * 4, s);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = s[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 9 * 4; k++) out[blockIdx.x * (9 * 4) + k] = acc[k];
    }
}
