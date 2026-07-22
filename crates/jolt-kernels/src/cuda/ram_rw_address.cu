extern "C" __global__ void ram_rw_address_round_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ ra_coeff,
    const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ val_init,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const unsigned int *__restrict__ pair,
    const u64 *__restrict__ one_plus_gamma_in,
    const u64 *__restrict__ gc_in,
    unsigned long num_groups
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 8;
    for (int k = 0; k < 8; k++) acc[k] = 0;

    unsigned long item = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < num_groups) {
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
            load4(val_init + (unsigned long)(2 * p + 1) * 4, val1);
        } else {
            load4(ra_coeff + (unsigned long)oi * 4, ra1);
            load4(val_init + (unsigned long)(2 * p) * 4, val0);
            load4(val_coeff + (unsigned long)oi * 4, val1);
        }

        u64 one_plus_gamma[4], gc[4];
        load4(one_plus_gamma_in, one_plus_gamma);
        load4(gc_in, gc);

        u64 ra_delta[4], val_delta[4], ra2[4], val2[4];
        fr_sub(ra1, ra0, ra_delta);
        fr_add(ra1, ra_delta, ra2);
        fr_sub(val1, val0, val_delta);
        fr_add(val1, val_delta, val2);

        u64 body0[4], body2[4], t[4];
        fr_mul(one_plus_gamma, val0, t);
        fr_add(t, gc, body0);
        fr_mul(one_plus_gamma, val2, t);
        fr_add(t, gc, body2);

        fr_mul(ra0, body0, acc + 0);
        fr_mul(ra2, body2, acc + 4);
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

extern "C" __global__ void ram_rw_address_bind(
    u64 *__restrict__ ra_out,
    u64 *__restrict__ val_out,
    u64 *__restrict__ prev_out,
    u64 *__restrict__ next_out,
    const u64 *__restrict__ ra_coeff,
    const u64 *__restrict__ val_coeff,
    const u64 *__restrict__ prev_val,
    const u64 *__restrict__ next_val,
    const u64 *__restrict__ val_init,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const unsigned int *__restrict__ pair,
    const u64 *__restrict__ challenge,
    unsigned long num_groups
) {
    unsigned long item = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= num_groups) return;

    u64 x[4];
    load4(challenge, x);
    int ei = even_idx[item];
    int oi = odd_idx[item];
    unsigned int p = pair[item];

    u64 ra[4], v[4], pv[4], nv[4];
    u64 zero4[4] = {0, 0, 0, 0};

    if (ei >= 0 && oi >= 0) {
        u64 rae[4], rao[4], ve[4], vo[4], pe[4], po[4], ne[4], no[4];
        load4(ra_coeff + (unsigned long)ei * 4, rae);
        load4(ra_coeff + (unsigned long)oi * 4, rao);
        load4(val_coeff + (unsigned long)ei * 4, ve);
        load4(val_coeff + (unsigned long)oi * 4, vo);
        load4(prev_val + (unsigned long)ei * 4, pe);
        load4(prev_val + (unsigned long)oi * 4, po);
        load4(next_val + (unsigned long)ei * 4, ne);
        load4(next_val + (unsigned long)oi * 4, no);
        sparse_register_linear_eval(rae, rao, x, ra);
        sparse_register_linear_eval(ve, vo, x, v);
        sparse_register_linear_eval(pe, po, x, pv);
        sparse_register_linear_eval(ne, no, x, nv);
    } else if (ei >= 0) {
        u64 rae[4], ve[4], pe[4], ne[4], ck[4];
        load4(ra_coeff + (unsigned long)ei * 4, rae);
        load4(val_coeff + (unsigned long)ei * 4, ve);
        load4(prev_val + (unsigned long)ei * 4, pe);
        load4(next_val + (unsigned long)ei * 4, ne);
        load4(val_init + (unsigned long)(2 * p + 1) * 4, ck);
        sparse_register_linear_eval(rae, zero4, x, ra);
        sparse_register_linear_eval(ve, ck, x, v);
        sparse_register_linear_eval(pe, ck, x, pv);
        sparse_register_linear_eval(ne, ck, x, nv);
    } else {
        u64 rao[4], vo[4], po[4], no[4], ck[4];
        load4(ra_coeff + (unsigned long)oi * 4, rao);
        load4(val_coeff + (unsigned long)oi * 4, vo);
        load4(prev_val + (unsigned long)oi * 4, po);
        load4(next_val + (unsigned long)oi * 4, no);
        load4(val_init + (unsigned long)(2 * p) * 4, ck);
        sparse_register_linear_eval(zero4, rao, x, ra);
        sparse_register_linear_eval(ck, vo, x, v);
        sparse_register_linear_eval(ck, po, x, pv);
        sparse_register_linear_eval(ck, no, x, nv);
    }

    for (int k = 0; k < 4; k++) {
        ra_out[item * 4 + k] = ra[k];
        val_out[item * 4 + k] = v[k];
        prev_out[item * 4 + k] = pv[k];
        next_out[item * 4 + k] = nv[k];
    }
}
