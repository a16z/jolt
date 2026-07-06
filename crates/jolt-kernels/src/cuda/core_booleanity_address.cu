extern "C" __global__ void core_booleanity_address_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ g_polys,
    const u64 *__restrict__ f_values,
    const u64 *__restrict__ gamma_squares,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long group_stride,
    unsigned int num_polys,
    unsigned long groups,
    unsigned int in_bits,
    unsigned int m
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long group = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (group < groups) {
        // STUB: real body accumulates acc[0]=eval_0, acc[4]=eval_infty.
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * 8;
            u64 s[4];
            fr_add(acc + 0, other + 0, s);
            for (int k = 0; k < 4; k++) acc[k] = s[k];
            fr_add(acc + 4, other + 4, s);
            for (int k = 0; k < 4; k++) acc[4 + k] = s[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 8; k++) out[blockIdx.x * 8 + k] = acc[k];
    }
}
