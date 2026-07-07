extern "C" __global__ void core_booleanity_sparse_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const u64 *__restrict__ present_mask,
    const unsigned char *__restrict__ values,
    const u64 *__restrict__ rho,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long num_polys,
    unsigned long chunk_domain,
    unsigned long poly_stride,
    unsigned long half,
    unsigned int in_bits,
    unsigned int round
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

        unsigned int num_sets = 1u << (round - 1);
        unsigned long base = row << round;
        unsigned long set_stride = num_polys * chunk_domain;

        u64 c_sum[4], q_sum[4];
        for (int k = 0; k < 4; k++) { c_sum[k] = 0; q_sum[k] = 0; }
        for (unsigned int i = 0; i < num_polys; i++) {
            u64 h0[4], h1[4];
            for (int k = 0; k < 4; k++) { h0[k] = 0; h1[k] = 0; }
            for (unsigned int set = 0; set < num_sets; set++) {
                const u64 *table = tables + ((unsigned long)set * set_stride + i * chunk_domain) * 4;
                unsigned long s0 = base + set;
                unsigned long s1 = base + num_sets + set;
                if ((present_mask[s0] >> i) & 1ull) {
                    u64 e[4], t[4];
                    load4(table + (unsigned long)values[s0 * poly_stride + i] * 4, e);
                    fr_add(h0, e, t);
                    for (int k = 0; k < 4; k++) h0[k] = t[k];
                }
                if ((present_mask[s1] >> i) & 1ull) {
                    u64 e[4], t[4];
                    load4(table + (unsigned long)values[s1 * poly_stride + i] * 4, e);
                    fr_add(h1, e, t);
                    for (int k = 0; k < 4; k++) h1[k] = t[k];
                }
            }

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

extern "C" __global__ void core_booleanity_sparse_collapse8(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const u64 *__restrict__ present_mask,
    const unsigned char *__restrict__ values,
    unsigned long num_polys,
    unsigned long chunk_domain,
    unsigned long poly_stride,
    unsigned long out_len
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_polys * out_len;
    if (i < total) {
        unsigned long poly = i / out_len;
        unsigned long j = i % out_len;
        unsigned long base = 8 * j;
        unsigned long set_stride = num_polys * chunk_domain;
        u64 acc[4];
        for (int k = 0; k < 4; k++) acc[k] = 0;
        for (unsigned int set = 0; set < 8; set++) {
            unsigned long s = base + set;
            if ((present_mask[s] >> poly) & 1ull) {
                const u64 *table =
                    tables + ((unsigned long)set * set_stride + poly * chunk_domain) * 4;
                u64 e[4], t[4];
                load4(table + (unsigned long)values[s * poly_stride + poly] * 4, e);
                fr_add(acc, e, t);
                for (int k = 0; k < 4; k++) acc[k] = t[k];
            }
        }
        store4(out + i * 4, acc);
    }
}

extern "C" __global__ void core_booleanity_sparse_bind(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    const u64 *__restrict__ challenge,
    const u64 *__restrict__ one_minus,
    unsigned long set_elems,
    unsigned long num_sets
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_sets * set_elems;
    if (i < total) {
        u64 c[4], om[4], v[4], lo[4], hi[4];
        load4(challenge, c);
        load4(one_minus, om);
        load4(in + i * 4, v);
        fr_mul(om, v, lo);
        fr_mul(c, v, hi);
        store4(out + i * 4, lo);
        store4(out + (total + i) * 4, hi);
    }
}
