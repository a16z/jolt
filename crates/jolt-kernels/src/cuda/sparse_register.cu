extern "C" __global__ void sparse_register_round_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ val,
    const u64 *__restrict__ read_ra,
    const u64 *__restrict__ rd_wa,
    const u64 *__restrict__ prev_val,
    const u64 *__restrict__ next_val,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const unsigned int *__restrict__ pair,
    const u64 *__restrict__ rd_inc,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned int in_pairs,
    unsigned long items
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long item = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < items) {
        // STUB: real body accumulates acc[0]=weight*body0, acc[4]=weight*body2.
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
