extern "C" __global__ void ram_rw_cycle_round_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ ra_coeff,
    const u64 *__restrict__ prev_val,
    const u64 *__restrict__ next_val,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const unsigned int *__restrict__ pair,
    const u64 *__restrict__ inc,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    const u64 *__restrict__ gamma,
    const u64 *__restrict__ one_plus_gamma_in,
    unsigned int in_pairs,
    unsigned long items
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long item = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < items) {
        u64 zero4[4] = {0, 0, 0, 0};
        int ei = even_idx[item];
        int oi = odd_idx[item];
        unsigned int p = pair[item];

        u64 ra0[4], ra1[4], val0[4], val1[4];
        for (int k = 0; k < 4; k++) {
            ra0[k] = 0; ra1[k] = 0; val0[k] = 0; val1[k] = 0;
        }

        if (ei >= 0 && oi >= 0) {
            load4(ra_coeff + (unsigned long)ei * 4, ra0);
            load4(ra_coeff + (unsigned long)oi * 4, ra1);
            load4(val_coeff + (unsigned long)ei * 4, val0);
            load4(val_coeff + (unsigned long)oi * 4, val1);
        } else if (ei >= 0) {
            load4(ra_coeff + (unsigned long)ei * 4, ra0);
            load4(val_coeff + (unsigned long)ei * 4, val0);
            load4(next_val + (unsigned long)ei * 4, val1);
        } else {
            load4(ra_coeff + (unsigned long)oi * 4, ra1);
            load4(prev_val + (unsigned long)oi * 4, val0);
            load4(val_coeff + (unsigned long)oi * 4, val1);
        }

        u64 g[4], one_plus_gamma[4];
        load4(gamma, g);
        load4(one_plus_gamma_in, one_plus_gamma);

        u64 inc0[4], inc1[4], inc_delta[4];
        load4(inc + (unsigned long)(2 * p) * 4, inc0);
        load4(inc + (unsigned long)(2 * p + 1) * 4, inc1);
        fr_sub(inc1, inc0, inc_delta);

        u64 val_delta[4];
        fr_sub(val1, val0, val_delta);

        u64 ra_delta[4];
        fr_sub(ra1, ra0, ra_delta);

        u64 t1[4], t2[4], body0[4], body_delta[4];
        fr_mul(one_plus_gamma, val0, t1);
        fr_mul(g, inc0, t2);
        fr_add(t1, t2, body0);
        fr_mul(one_plus_gamma, val_delta, t1);
        fr_mul(g, inc_delta, t2);
        fr_add(t1, t2, body_delta);

        u64 weight[4], esum[4];
        if (in_pairs == 0) {
            u64 eo0[4], eo1[4], ein[4];
            load4(e_out + (unsigned long)(2 * p) * 4, eo0);
            load4(e_out + (unsigned long)(2 * p + 1) * 4, eo1);
            fr_add(eo0, eo1, esum);
            load4(e_in, ein);
            fr_mul(ein, esum, weight);
        } else {
            unsigned int x_out = p / in_pairs;
            unsigned int x_in = p % in_pairs;
            u64 ei0[4], ei1[4], eout[4];
            load4(e_in + (unsigned long)(2 * x_in) * 4, ei0);
            load4(e_in + (unsigned long)(2 * x_in + 1) * 4, ei1);
            fr_add(ei0, ei1, esum);
            load4(e_out + (unsigned long)x_out * 4, eout);
            fr_mul(eout, esum, weight);
        }

        u64 wr0[4], wrd[4];
        fr_mul(weight, ra0, wr0);
        fr_mul(weight, ra_delta, wrd);
        fr_mul(wr0, body0, acc + 0);
        fr_mul(wrd, body_delta, acc + 4);
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
