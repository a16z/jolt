extern "C" __global__ void bytecode_cycle_sparse_pairs(
    u64 *__restrict__ out,
    const u64 *__restrict__ tables,
    const short *__restrict__ values,
    const u64 *__restrict__ combined_eq,
    const u64 *__restrict__ points,
    unsigned long num_chunks,
    unsigned long chunk_domain,
    unsigned long source_rows,
    unsigned long half,
    unsigned int degree,
    unsigned int round
) {
    extern __shared__ u64 sdata[];
    u64 *acc = sdata + threadIdx.x * (degree * 4);
    for (unsigned int k = 0; k < degree * 4; k++) acc[k] = 0;

    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < half) {
        unsigned int num_sets = 1u << (round - 1);
        unsigned long base = row << round;
        unsigned long set_stride = num_chunks * chunk_domain;

        u64 prod[8][4];
        bool prod_init = false;

        for (unsigned long ch = 0; ch < num_chunks; ch++) {
            u64 lo[4], hi[4];
            for (int k = 0; k < 4; k++) { lo[k] = 0; hi[k] = 0; }
            for (unsigned int set = 0; set < num_sets; set++) {
                const u64 *table =
                    tables + ((unsigned long)set * set_stride + ch * chunk_domain) * 4;
                short v_lo = values[ch * source_rows + (base + set)];
                short v_hi = values[ch * source_rows + (base + num_sets + set)];
                if (v_lo >= 0) {
                    u64 e[4], t[4];
                    load4(table + (unsigned long)v_lo * 4, e);
                    fr_add(lo, e, t);
                    for (int k = 0; k < 4; k++) lo[k] = t[k];
                }
                if (v_hi >= 0) {
                    u64 e[4], t[4];
                    load4(table + (unsigned long)v_hi * 4, e);
                    fr_add(hi, e, t);
                    for (int k = 0; k < 4; k++) hi[k] = t[k];
                }
            }
            u64 slope[4];
            fr_sub(hi, lo, slope);
            for (unsigned int p = 0; p < degree; p++) {
                u64 x[4], lin[4];
                load4(points + p * 4, x);
                fr_mul(slope, x, lin);
                fr_add(lo, lin, lin);
                if (!prod_init) {
                    for (int k = 0; k < 4; k++) prod[p][k] = lin[k];
                } else {
                    u64 t[4];
                    fr_mul(prod[p], lin, t);
                    for (int k = 0; k < 4; k++) prod[p][k] = t[k];
                }
            }
            prod_init = true;
        }
        if (!prod_init) {
            for (unsigned int p = 0; p < degree; p++) {
                for (int k = 0; k < 4; k++) prod[p][k] = 0;
            }
        }

        u64 clo[4], chi[4], cslope[4];
        load4(combined_eq + (row * 2) * 4, clo);
        load4(combined_eq + (row * 2 + 1) * 4, chi);
        fr_sub(chi, clo, cslope);
        for (unsigned int p = 0; p < degree; p++) {
            u64 x[4];
            load4(points + p * 4, x);
            u64 w[4], t[4], term[4];
            fr_mul(cslope, x, w);
            fr_add(clo, w, w);
            fr_mul(prod[p], w, term);
            fr_add(acc + p * 4, term, t);
            for (int k = 0; k < 4; k++) acc[p * 4 + k] = t[k];
        }
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            u64 *other = sdata + (threadIdx.x + stride) * (degree * 4);
            for (unsigned int p = 0; p < degree; p++) {
                u64 s[4];
                fr_add(acc + p * 4, other + p * 4, s);
                for (int k = 0; k < 4; k++) acc[p * 4 + k] = s[k];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (unsigned int k = 0; k < degree * 4; k++) out[blockIdx.x * (degree * 4) + k] = acc[k];
    }
}
