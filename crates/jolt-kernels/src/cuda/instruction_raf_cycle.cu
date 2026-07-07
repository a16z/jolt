__device__ __forceinline__ void irc_ep2(
    const u64 *p0, const u64 *p1, const u64 *q0, const u64 *q1,
    u64 *e1, u64 *e2, u64 *einf
) {
    u64 pinf[4], p2[4], qinf[4], q2[4];
    fr_sub(p1, p0, pinf);
    fr_add(pinf, p1, p2);
    fr_sub(q1, q0, qinf);
    fr_add(qinf, q1, q2);
    fr_mul(p1, q1, e1);
    fr_mul(p2, q2, e2);
    fr_mul(pinf, qinf, einf);
}

__device__ __forceinline__ void irc_extrap2(
    const u64 *e1, const u64 *e2, const u64 *einf, u64 *out
) {
    u64 v[4];
    fr_add(e2, einf, v);
    fr_add(v, v, v);
    fr_sub(v, e1, out);
}

__device__ __forceinline__ void irc_ep4(
    const u64 *lo, const u64 *hi,
    u64 *o1, u64 *o2, u64 *o3, u64 *o4, u64 *oinf
) {
    u64 a1[4], a2[4], ainf[4], a3[4], a4[4];
    irc_ep2(lo + 0, hi + 0, lo + 4, hi + 4, a1, a2, ainf);
    irc_extrap2(a1, a2, ainf, a3);
    irc_extrap2(a2, a3, ainf, a4);
    u64 b1[4], b2[4], binf[4], b3[4], b4[4];
    irc_ep2(lo + 8, hi + 8, lo + 12, hi + 12, b1, b2, binf);
    irc_extrap2(b1, b2, binf, b3);
    irc_extrap2(b2, b3, binf, b4);
    fr_mul(a1, b1, o1);
    fr_mul(a2, b2, o2);
    fr_mul(a3, b3, o3);
    fr_mul(a4, b4, o4);
    fr_mul(ainf, binf, oinf);
}

__device__ __forceinline__ void irc_extrap4_next2(
    const u64 *v0, const u64 *v1, const u64 *v2, const u64 *v3,
    const u64 *einf6, u64 *e5, u64 *e6
) {
    u64 e4m3[4];
    fr_sub(v3, v2, e4m3);
    fr_add(einf6, e4m3, e5);
    fr_add(e5, v1, e5);
    fr_add(e5, e5, e5);
    fr_sub(e5, v2, e5);
    fr_add(e5, e5, e5);
    fr_sub(e5, v0, e5);
    fr_sub(e5, e4m3, e6);
    fr_add(e6, einf6, e6);
    fr_add(e6, e6, e6);
    fr_sub(e6, v3, e6);
    fr_add(e6, e6, e6);
    fr_sub(e6, v1, e6);
}

__device__ __forceinline__ void irc_extrap4_2p2(
    const u64 *e1, const u64 *e2, const u64 *e3, const u64 *e4, const u64 *einf,
    u64 *e5, u64 *e6, u64 *e7, u64 *e8
) {
    u64 t2[4], einf6[4];
    fr_add(einf, einf, t2);
    fr_add(t2, t2, einf6);
    fr_add(einf6, t2, einf6);
    irc_extrap4_next2(e1, e2, e3, e4, einf6, e5, e6);
    irc_extrap4_next2(e3, e4, e5, e6, einf6, e7, e8);
}

// out holds 9 field elements: [a1*b1 .. a8*b8, ainf*binf] over the first 8 pairs.
__device__ __forceinline__ void irc_ep8(const u64 *lo, const u64 *hi, u64 *out) {
    u64 a1[4], a2[4], a3[4], a4[4], ainf[4], a5[4], a6[4], a7[4], a8[4];
    irc_ep4(lo + 0, hi + 0, a1, a2, a3, a4, ainf);
    irc_extrap4_2p2(a1, a2, a3, a4, ainf, a5, a6, a7, a8);
    u64 b1[4], b2[4], b3[4], b4[4], binf[4], b5[4], b6[4], b7[4], b8[4];
    irc_ep4(lo + 16, hi + 16, b1, b2, b3, b4, binf);
    irc_extrap4_2p2(b1, b2, b3, b4, binf, b5, b6, b7, b8);
    fr_mul(a1, b1, out + 0 * 4);
    fr_mul(a2, b2, out + 1 * 4);
    fr_mul(a3, b3, out + 2 * 4);
    fr_mul(a4, b4, out + 3 * 4);
    fr_mul(a5, b5, out + 4 * 4);
    fr_mul(a6, b6, out + 5 * 4);
    fr_mul(a7, b7, out + 6 * 4);
    fr_mul(a8, b8, out + 7 * 4);
    fr_mul(ainf, binf, out + 8 * 4);
}

// lo/hi hold 9 pairs (flat, 4 limbs each). out holds 9 field elements.
__device__ __forceinline__ void irc_ep9(const u64 *lo, const u64 *hi, u64 *out) {
    irc_ep8(lo, hi, out);
    const u64 *lin0 = lo + 32;
    const u64 *lin1 = hi + 32;
    u64 delta[4];
    fr_sub(lin1, lin0, delta);
    u64 ls[8][4];
    for (int k = 0; k < 4; k++) ls[0][k] = lin1[k];
    for (int i = 1; i < 8; i++) fr_add(ls[i - 1], delta, ls[i]);
    for (int i = 0; i < 8; i++) fr_mul(out + i * 4, ls[i], out + i * 4);
    fr_mul(out + 8 * 4, delta, out + 8 * 4);
}

extern "C" __global__ void instruction_raf_cycle_pairs(
    u64 *__restrict__ out,
    const u64 *const *__restrict__ factor_ptrs,
    const u64 *__restrict__ e_in,
    const u64 *__restrict__ e_out,
    unsigned long pair_stride,
    unsigned long half,
    unsigned int in_bits
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

        u64 lo[9 * 4], hi[9 * 4];
        // Factor 0 = combined, with e_in folded into its linear pair only.
        {
            const u64 *combined = factor_ptrs[0];
            u64 c0[4], c1[4];
            load4(combined + (row * 2) * 4, c0);
            load4(combined + (row * 2 + 1) * 4, c1);
            fr_mul(c0, ein, lo + 0);
            fr_mul(c1, ein, hi + 0);
        }
        // Factors 1..8 = the 8 ra chunks.
        for (int c = 0; c < 8; c++) {
            const u64 *factor = factor_ptrs[c + 1];
            load4(factor + (row * 2) * 4, lo + (c + 1) * 4);
            load4(factor + (row * 2 + 1) * 4, hi + (c + 1) * 4);
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
