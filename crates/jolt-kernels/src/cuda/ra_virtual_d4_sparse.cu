extern "C" __global__ void ra_virtual_d4_sparse_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const short *__restrict__ values,
    const u64 *__restrict__ gamma_powers,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long num_chunks,
    unsigned long chunk_domain,
    unsigned long source_rows,
    unsigned int virtual_count,
    unsigned long half,
    unsigned int in_bits,
    unsigned int round
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * 16;
    for (int k = 0; k < 16; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        unsigned long in_mask = ((unsigned long)1 << in_bits) - 1;
        u64 ei[4], eo[4], weight[4];
        load4(e_in + (row & in_mask) * 4, ei);
        load4(e_out + (row >> in_bits) * 4, eo);
        fr_mul(eo, ei, weight);

        unsigned int num_sets = 1u << (round - 1);
        unsigned long base = row << round;
        unsigned long set_stride = num_chunks * chunk_domain;

        for (unsigned int v = 0; v < virtual_count; v++) {
            u64 g[4], vw[4];
            load4(gamma_powers + v * 4, g);
            fr_mul(weight, g, vw);

            // Gather the 4 chunk (lo,hi) pairs for this virtual by summing the table-sets.
            u64 lo[4][4], hi[4][4];
            for (int c = 0; c < 4; c++) {
                unsigned long ch = (unsigned long)v * 4 + c;
                for (int k = 0; k < 4; k++) { lo[c][k] = 0; hi[c][k] = 0; }
                for (unsigned int set = 0; set < num_sets; set++) {
                    const u64 *table =
                        tables + ((unsigned long)set * set_stride + ch * chunk_domain) * 4;
                    short v_lo = values[ch * source_rows + (base + set)];
                    short v_hi = values[ch * source_rows + (base + num_sets + set)];
                    if (v_lo >= 0) {
                        u64 e[4], t[4];
                        load4(table + (unsigned long)v_lo * 4, e);
                        fr_add(lo[c], e, t);
                        for (int k = 0; k < 4; k++) lo[c][k] = t[k];
                    }
                    if (v_hi >= 0) {
                        u64 e[4], t[4];
                        load4(table + (unsigned long)v_hi * 4, e);
                        fr_add(hi[c], e, t);
                        for (int k = 0; k < 4; k++) hi[c][k] = t[k];
                    }
                }
            }

            // d4 product over the 4 gathered pairs (mirrors ra_d4_virtual_product).
            u64 a1e[4], a2e[4], ainf[4];
            {
                u64 pinf[4], p2[4], qinf[4], q2[4];
                fr_sub(hi[0], lo[0], pinf); fr_add(hi[0], pinf, p2);
                fr_sub(hi[1], lo[1], qinf); fr_add(hi[1], qinf, q2);
                fr_mul(hi[0], hi[1], a1e);
                fr_mul(p2, q2, a2e);
                fr_mul(pinf, qinf, ainf);
            }
            u64 b1e[4], b2e[4], binf[4];
            {
                u64 pinf[4], p2[4], qinf[4], q2[4];
                fr_sub(hi[2], lo[2], pinf); fr_add(hi[2], pinf, p2);
                fr_sub(hi[3], lo[3], qinf); fr_add(hi[3], qinf, q2);
                fr_mul(hi[2], hi[3], b1e);
                fr_mul(p2, q2, b2e);
                fr_mul(pinf, qinf, binf);
            }
            u64 a3e[4], b3e[4];
            {
                u64 s[4], d[4];
                fr_add(a2e, ainf, s); fr_add(s, s, d); fr_sub(d, a1e, a3e);
                fr_add(b2e, binf, s); fr_add(s, s, d); fr_sub(d, b1e, b3e);
            }
            u64 prod[16];
            fr_mul(a1e, b1e, prod + 0);
            fr_mul(a2e, b2e, prod + 4);
            fr_mul(a3e, b3e, prod + 8);
            fr_mul(ainf, binf, prod + 12);

            for (int e = 0; e < 4; e++) {
                u64 scaled[4], t[4];
                fr_mul(prod + e * 4, vw, scaled);
                fr_add(acc + e * 4, scaled, t);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = t[k];
            }
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * 16;
            for (int e = 0; e < 4; e++) {
                u64 s[4];
                fr_add(acc + e * 4, other + e * 4, s);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = s[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 16; k++) out[blockIdx.x * 16 + k] = acc[k];
    }
}

extern "C" __global__ void ra_virtual_d4_sparse_bind(
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

extern "C" __global__ void ra_virtual_d4_sparse_collapse(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const short *__restrict__ values,
    unsigned long num_chunks,
    unsigned long chunk_domain,
    unsigned long source_rows,
    unsigned long out_len
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_chunks * out_len;
    if (i < total) {
        unsigned long chunk = i / out_len;
        unsigned long j = i % out_len;
        unsigned long base = 8 * j;
        unsigned long set_stride = num_chunks * chunk_domain;
        u64 acc[4];
        for (int k = 0; k < 4; k++) acc[k] = 0;
        for (unsigned int set = 0; set < 8; set++) {
            short v = values[chunk * source_rows + (base + set)];
            if (v >= 0) {
                const u64 *table =
                    tables + ((unsigned long)set * set_stride + chunk * chunk_domain) * 4;
                u64 e[4], t[4];
                load4(table + (unsigned long)v * 4, e);
                fr_add(acc, e, t);
                for (int k = 0; k < 4; k++) acc[k] = t[k];
            }
        }
        store4(out + i * 4, acc);
    }
}
