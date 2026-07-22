extern "C" __global__ void hamming_booleanity_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ hamming_weight,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long half,
    unsigned int in_bits
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        unsigned long in_mask = ((unsigned long)1 << in_bits) - 1;
        u64 ei[4], eo[4], weight[4];
        load4(e_in + (row & in_mask) * 4, ei);
        load4(e_out + (row >> in_bits) * 4, eo);
        fr_mul(eo, ei, weight);

        u64 h0[4], h1[4];
        load4(hamming_weight + (row * 2) * 4, h0);
        load4(hamming_weight + (row * 2 + 1) * 4, h1);

        // q_constant += weight * (h0^2 - h0)
        u64 h0_sq[4], bool0[4], term0[4];
        fr_mul(h0, h0, h0_sq);
        fr_sub(h0_sq, h0, bool0);
        fr_mul(weight, bool0, term0);
        for (int k = 0; k < 4; k++) acc[k] = term0[k];

        // q_top += weight * (h1 - h0)^2
        u64 delta[4], delta_sq[4], term1[4];
        fr_sub(h1, h0, delta);
        fr_mul(delta, delta, delta_sq);
        fr_mul(weight, delta_sq, term1);
        for (int k = 0; k < 4; k++) acc[4 + k] = term1[k];
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * 8;
            u64 sum[4];
            fr_add(acc + 0, other + 0, sum);
            for (int k = 0; k < 4; k++) acc[k] = sum[k];
            fr_add(acc + 4, other + 4, sum);
            for (int k = 0; k < 4; k++) acc[4 + k] = sum[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 8; k++) out[blockIdx.x * 8 + k] = acc[k];
    }
}
