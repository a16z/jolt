extern "C" __global__ void hamming_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ g_ptrs,
    const u64 *const *__restrict__ eq_virt_ptrs,
    const u64 *__restrict__ eq_bool,
    const u64 *__restrict__ gamma_powers,
    const u64 *__restrict__ two,
    unsigned int num_ra,
    unsigned long pair_stride,
    unsigned long half
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        u64 t[4];
        load4(two, t);

        u64 eb_lo[4], eb_hi[4], eb1[4];
        load4(eq_bool + (row * 2) * 4, eb_lo);
        load4(eq_bool + (row * 2 + 1) * 4, eb_hi);
        fr_sub(eb_hi, eb_lo, eb1);
        fr_mul(eb1, t, eb1);
        fr_add(eb_lo, eb1, eb1);

        u64 eval0[4], eval1[4];
        for (int k = 0; k < 4; k++) { eval0[k] = 0; eval1[k] = 0; }

        for (unsigned int i = 0; i < num_ra; i++) {
            const u64 *gi = g_ptrs[i];
            const u64 *evi = eq_virt_ptrs[i];

            u64 g_lo[4], g_hi[4], g1[4];
            load4(gi + (row * 2) * 4, g_lo);
            load4(gi + (row * 2 + 1) * 4, g_hi);
            fr_sub(g_hi, g_lo, g1);
            fr_mul(g1, t, g1);
            fr_add(g_lo, g1, g1);

            u64 ev_lo[4], ev_hi[4], ev1[4];
            load4(evi + (row * 2) * 4, ev_lo);
            load4(evi + (row * 2 + 1) * 4, ev_hi);
            fr_sub(ev_hi, ev_lo, ev1);
            fr_mul(ev1, t, ev1);
            fr_add(ev_lo, ev1, ev1);

            u64 gamma_hw[4], gamma_bool[4], gamma_virt[4];
            load4(gamma_powers + (3 * i) * 4, gamma_hw);
            load4(gamma_powers + (3 * i + 1) * 4, gamma_bool);
            load4(gamma_powers + (3 * i + 2) * 4, gamma_virt);

            u64 w0[4], w1[4], tmp[4];
            fr_mul(gamma_bool, eb_lo, w0);
            fr_add(gamma_hw, w0, w0);
            fr_mul(gamma_virt, ev_lo, tmp);
            fr_add(w0, tmp, w0);

            fr_mul(gamma_bool, eb1, w1);
            fr_add(gamma_hw, w1, w1);
            fr_mul(gamma_virt, ev1, tmp);
            fr_add(w1, tmp, w1);

            fr_mul(g_lo, w0, tmp);
            fr_add(eval0, tmp, eval0);
            fr_mul(g1, w1, tmp);
            fr_add(eval1, tmp, eval1);
        }
        for (int k = 0; k < 4; k++) { acc[k] = eval0[k]; acc[4 + k] = eval1[k]; }
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
