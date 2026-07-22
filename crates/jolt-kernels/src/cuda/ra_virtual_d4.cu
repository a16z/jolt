__device__ __forceinline__ void ra_d4_virtual_product(
    const u64 *const *__restrict__ factor_ptrs,
    unsigned long row,
    unsigned int base,
    u64 *prod
) {
    u64 lo[4][4], hi[4][4];
    for (int i = 0; i < 4; i++) {
        const u64 *c = factor_ptrs[base + i];
        load4(c + (row * 2) * 4, lo[i]);
        load4(c + (row * 2 + 1) * 4, hi[i]);
    }

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

    fr_mul(a1e, b1e, prod + 0);
    fr_mul(a2e, b2e, prod + 4);
    fr_mul(a3e, b3e, prod + 8);
    fr_mul(ainf, binf, prod + 12);
}

extern "C" __global__ void ra_virtual_d4_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ factor_ptrs,
    const u64 *__restrict__ gamma_powers,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long pair_stride,
    unsigned int virtual_count,
    unsigned long half,
    unsigned int in_bits
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

        u64 sum[16];
        for (int k = 0; k < 16; k++) sum[k] = 0;
        for (unsigned int v = 0; v < virtual_count; v++) {
            u64 prod[16];
            ra_d4_virtual_product(factor_ptrs, row, v * 4, prod);
            u64 g[4];
            load4(gamma_powers + v * 4, g);
            for (int e = 0; e < 4; e++) {
                u64 scaled[4], t[4];
                fr_mul(prod + e * 4, g, scaled);
                fr_add(sum + e * 4, scaled, t);
                for (int k = 0; k < 4; k++) sum[e * 4 + k] = t[k];
            }
        }
        for (int e = 0; e < 4; e++) {
            fr_mul(sum + e * 4, weight, acc + e * 4);
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
