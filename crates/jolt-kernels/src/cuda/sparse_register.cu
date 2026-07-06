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
        u64 zero4[4] = {0, 0, 0, 0};
        int ei = even_idx[item];
        int oi = odd_idx[item];
        unsigned int p = pair[item];

        u64 v0[4], vd[4], r0[4], rd[4], w0[4], wd[4];
        for (int k = 0; k < 4; k++) {
            v0[k] = 0; vd[k] = 0; r0[k] = 0; rd[k] = 0; w0[k] = 0; wd[k] = 0;
        }

        if (ei >= 0 && oi >= 0) {
            u64 ve[4], vo[4], re[4], ro[4], we[4], wo[4];
            load4(val + (unsigned long)ei * 4, ve);
            load4(val + (unsigned long)oi * 4, vo);
            load4(read_ra + (unsigned long)ei * 4, re);
            load4(read_ra + (unsigned long)oi * 4, ro);
            load4(rd_wa + (unsigned long)ei * 4, we);
            load4(rd_wa + (unsigned long)oi * 4, wo);
            for (int k = 0; k < 4; k++) { v0[k] = ve[k]; r0[k] = re[k]; w0[k] = we[k]; }
            fr_sub(vo, ve, vd);
            fr_sub(ro, re, rd);
            fr_sub(wo, we, wd);
        } else if (ei >= 0) {
            u64 ve[4], nv[4], re[4], we[4];
            load4(val + (unsigned long)ei * 4, ve);
            load4(next_val + (unsigned long)ei * 4, nv);
            load4(read_ra + (unsigned long)ei * 4, re);
            load4(rd_wa + (unsigned long)ei * 4, we);
            for (int k = 0; k < 4; k++) { v0[k] = ve[k]; r0[k] = re[k]; w0[k] = we[k]; }
            fr_sub(nv, ve, vd);
            fr_sub(zero4, re, rd);
            fr_sub(zero4, we, wd);
        } else if (oi >= 0) {
            u64 pv[4], vo[4], ro[4], wo[4];
            load4(prev_val + (unsigned long)oi * 4, pv);
            load4(val + (unsigned long)oi * 4, vo);
            load4(read_ra + (unsigned long)oi * 4, ro);
            load4(rd_wa + (unsigned long)oi * 4, wo);
            for (int k = 0; k < 4; k++) { v0[k] = pv[k]; rd[k] = ro[k]; wd[k] = wo[k]; }
            fr_sub(vo, pv, vd);
        }

        u64 inc0[4], inc1[4], inc_delta[4];
        load4(rd_inc + (unsigned long)(2 * p) * 4, inc0);
        load4(rd_inc + (unsigned long)(2 * p + 1) * 4, inc1);
        fr_sub(inc1, inc0, inc_delta);

        u64 vi0[4], vid[4];
        fr_add(v0, inc0, vi0);
        fr_add(vd, inc_delta, vid);

        u64 t1[4], t2[4], body0[4], body2[4];
        fr_mul(w0, vi0, t1);
        fr_mul(r0, v0, t2);
        fr_add(t1, t2, body0);
        fr_mul(wd, vid, t1);
        fr_mul(rd, vd, t2);
        fr_add(t1, t2, body2);

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

        fr_mul(weight, body0, acc + 0);
        fr_mul(weight, body2, acc + 4);
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

__device__ __forceinline__ void sparse_register_linear_eval(
    const u64 *low, const u64 *high, const u64 *x, u64 *out
) {
    u64 diff[4], scaled[4];
    fr_sub(high, low, diff);
    fr_mul(diff, x, scaled);
    fr_add(low, scaled, out);
}

extern "C" __global__ void sparse_register_bind(
    u64 *__restrict__ val_out,
    u64 *__restrict__ read_ra_out,
    u64 *__restrict__ rd_wa_out,
    u64 *__restrict__ prev_out,
    u64 *__restrict__ next_out,
    const u64 *__restrict__ val,
    const u64 *__restrict__ read_ra,
    const u64 *__restrict__ rd_wa,
    const u64 *__restrict__ prev_val,
    const u64 *__restrict__ next_val,
    const int *__restrict__ even_idx,
    const int *__restrict__ odd_idx,
    const u64 *__restrict__ challenge,
    unsigned long items
) {
    unsigned long item = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= items) return;

    u64 x[4];
    load4(challenge, x);
    int ei = even_idx[item];
    int oi = odd_idx[item];

    u64 v[4], r[4], w[4], pv[4], nv[4];
    u64 zero4[4] = {0, 0, 0, 0};

    if (ei >= 0 && oi >= 0) {
        u64 ve[4], vo[4], re[4], ro[4], we[4], wo[4];
        load4(val + (unsigned long)ei * 4, ve);
        load4(val + (unsigned long)oi * 4, vo);
        load4(read_ra + (unsigned long)ei * 4, re);
        load4(read_ra + (unsigned long)oi * 4, ro);
        load4(rd_wa + (unsigned long)ei * 4, we);
        load4(rd_wa + (unsigned long)oi * 4, wo);
        sparse_register_linear_eval(ve, vo, x, v);
        sparse_register_linear_eval(re, ro, x, r);
        sparse_register_linear_eval(we, wo, x, w);
        load4(prev_val + (unsigned long)ei * 4, pv);
        load4(next_val + (unsigned long)oi * 4, nv);
    } else if (ei >= 0) {
        u64 ve[4], ne[4], re[4], we[4];
        load4(val + (unsigned long)ei * 4, ve);
        load4(next_val + (unsigned long)ei * 4, ne);
        load4(read_ra + (unsigned long)ei * 4, re);
        load4(rd_wa + (unsigned long)ei * 4, we);
        sparse_register_linear_eval(ve, ne, x, v);
        sparse_register_linear_eval(re, zero4, x, r);
        sparse_register_linear_eval(we, zero4, x, w);
        load4(prev_val + (unsigned long)ei * 4, pv);
        for (int k = 0; k < 4; k++) nv[k] = ne[k];
    } else {
        u64 po[4], vo[4], ro[4], wo[4];
        load4(prev_val + (unsigned long)oi * 4, po);
        load4(val + (unsigned long)oi * 4, vo);
        load4(read_ra + (unsigned long)oi * 4, ro);
        load4(rd_wa + (unsigned long)oi * 4, wo);
        sparse_register_linear_eval(po, vo, x, v);
        sparse_register_linear_eval(zero4, ro, x, r);
        sparse_register_linear_eval(zero4, wo, x, w);
        for (int k = 0; k < 4; k++) pv[k] = po[k];
        load4(next_val + (unsigned long)oi * 4, nv);
    }

    for (int k = 0; k < 4; k++) {
        val_out[item * 4 + k] = v[k];
        read_ra_out[item * 4 + k] = r[k];
        rd_wa_out[item * 4 + k] = w[k];
        prev_out[item * 4 + k] = pv[k];
        next_out[item * 4 + k] = nv[k];
    }
}
