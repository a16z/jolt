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
        unsigned long in_mask = ((unsigned long)1 << in_bits) - 1;
        u64 ei[4], eo[4], weight[4];
        load4(e_in + (group & in_mask) * 4, ei);
        load4(e_out + (group >> in_bits) * 4, eo);
        fr_mul(eo, ei, weight);

        unsigned int block_len = 1u << m;
        unsigned int f_mask = (1u << (m - 1)) - 1;
        unsigned long block_start = group << m;
        for (unsigned int i = 0; i < num_polys; i++) {
            const u64 *g = g_polys + (unsigned long)i * group_stride * 4;
            u64 eval_0[4], eval_infty[4];
            for (int k = 0; k < 4; k++) { eval_0[k] = 0; eval_infty[k] = 0; }
            for (unsigned int k = 0; k < block_len; k++) {
                u64 g_k[4], f_k[4];
                load4(g + (block_start + k) * 4, g_k);
                load4(f_values + (k & f_mask) * 4, f_k);

                u64 g_times_f[4], eval_inf[4], t[4];
                fr_mul(g_k, f_k, g_times_f);
                fr_mul(g_times_f, f_k, eval_inf);
                if ((k >> (m - 1)) == 0) {
                    u64 d[4];
                    fr_sub(eval_inf, g_times_f, d);
                    fr_add(eval_0, d, t);
                    for (int j = 0; j < 4; j++) eval_0[j] = t[j];
                }
                fr_add(eval_infty, eval_inf, t);
                for (int j = 0; j < 4; j++) eval_infty[j] = t[j];
            }

            u64 gsq[4], w[4], scaled[4], t[4];
            load4(gamma_squares + i * 4, gsq);
            fr_mul(weight, gsq, w);
            fr_mul(w, eval_0, scaled);
            fr_add(acc + 0, scaled, t);
            for (int j = 0; j < 4; j++) acc[j] = t[j];
            fr_mul(w, eval_infty, scaled);
            fr_add(acc + 4, scaled, t);
            for (int j = 0; j < 4; j++) acc[4 + j] = t[j];
        }
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
