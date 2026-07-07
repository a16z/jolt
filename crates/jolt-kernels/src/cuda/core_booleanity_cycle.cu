extern "C" __global__ void core_booleanity_cycle_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ h_ptrs,
    const u64 *__restrict__ rho,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long pair_stride,
    unsigned int num_polys,
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

        u64 c_sum[4], q_sum[4];
        for (int k = 0; k < 4; k++) { c_sum[k] = 0; q_sum[k] = 0; }
        for (unsigned int i = 0; i < num_polys; i++) {
            const u64 *h = h_ptrs[i];
            u64 h0[4], h1[4];
            load4(h + (row * 2) * 4, h0);
            load4(h + (row * 2 + 1) * 4, h1);

            u64 r[4], diff[4], term[4], t[4];
            load4(rho + i * 4, r);
            fr_sub(h0, r, diff);
            fr_mul(h0, diff, term);
            fr_add(c_sum, term, t);
            for (int k = 0; k < 4; k++) c_sum[k] = t[k];

            u64 delta[4], dsq[4];
            fr_sub(h1, h0, delta);
            fr_mul(delta, delta, dsq);
            fr_add(q_sum, dsq, t);
            for (int k = 0; k < 4; k++) q_sum[k] = t[k];
        }
        fr_mul(weight, c_sum, acc + 0);
        fr_mul(weight, q_sum, acc + 4);
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
