extern "C" __global__ void instruction_raf_cycle_sparse_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const unsigned short *__restrict__ values,
    const u64 *__restrict__ combined,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long num_chunks,
    unsigned long chunk_domain,
    unsigned long source_rows,
    unsigned long half,
    unsigned int in_bits,
    unsigned int round
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (9 * 4);
    for (int k = 0; k < 9 * 4; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        unsigned long in_mask = ((unsigned long)1 << in_bits) - 1;
        u64 ein[4], eout[4];
        load4(e_in + (row & in_mask) * 4, ein);
        load4(e_out + (row >> in_bits) * 4, eout);

        unsigned int num_sets = 1u << (round - 1);
        unsigned long base = row << round;
        unsigned long set_stride = num_chunks * chunk_domain;

        u64 lo[9 * 4], hi[9 * 4];
        {
            u64 c0[4], c1[4];
            load4(combined + (row * 2) * 4, c0);
            load4(combined + (row * 2 + 1) * 4, c1);
            fr_mul(c0, ein, lo + 0);
            fr_mul(c1, ein, hi + 0);
        }
        for (unsigned long ch = 0; ch < num_chunks; ch++) {
            u64 l[4], h[4];
            for (int k = 0; k < 4; k++) { l[k] = 0; h[k] = 0; }
            for (unsigned int set = 0; set < num_sets; set++) {
                const u64 *table =
                    tables + ((unsigned long)set * set_stride + ch * chunk_domain) * 4;
                unsigned short v_lo = values[ch * source_rows + (base + set)];
                unsigned short v_hi = values[ch * source_rows + (base + num_sets + set)];
                u64 e[4], t[4];
                load4(table + (unsigned long)v_lo * 4, e);
                fr_add(l, e, t);
                for (int k = 0; k < 4; k++) l[k] = t[k];
                load4(table + (unsigned long)v_hi * 4, e);
                fr_add(h, e, t);
                for (int k = 0; k < 4; k++) h[k] = t[k];
            }
            for (int k = 0; k < 4; k++) {
                lo[(ch + 1) * 4 + k] = l[k];
                hi[(ch + 1) * 4 + k] = h[k];
            }
        }

        u64 evals[9 * 4];
        irc_ep9(lo, hi, evals);
        for (int e = 0; e < 9; e++) {
            fr_mul(evals + e * 4, eout, acc + e * 4);
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (9 * 4);
            for (int e = 0; e < 9; e++) {
                u64 s[4];
                fr_add(acc + e * 4, other + e * 4, s);
                for (int k = 0; k < 4; k++) acc[e * 4 + k] = s[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int k = 0; k < 9 * 4; k++) out[blockIdx.x * (9 * 4) + k] = acc[k];
    }
}

extern "C" __global__ void instruction_raf_cycle_sparse_collapse(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const unsigned short *__restrict__ values,
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
            unsigned short v = values[chunk * source_rows + (base + set)];
            const u64 *table =
                tables + ((unsigned long)set * set_stride + chunk * chunk_domain) * 4;
            u64 e[4], t[4];
            load4(table + (unsigned long)v * 4, e);
            fr_add(acc, e, t);
            for (int k = 0; k < 4; k++) acc[k] = t[k];
        }
        store4(out + i * 4, acc);
    }
}
