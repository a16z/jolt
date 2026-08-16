extern "C" __global__ void bap_bind_squared_kernel(const u64 *__restrict__ in,
                                                   const u64 *__restrict__ low,
                                                   const u64 *__restrict__ high,
                                                   u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    u64 a[LIMBS], b[LIMBS], w0[LIMBS], w1[LIMBS], pa[LIMBS], pb[LIMBS], s[LIMBS];
    load4(in + (2ull * (unsigned long long)i) * LIMBS, a);
    load4(in + (2ull * (unsigned long long)i + 1ull) * LIMBS, b);
    load4(low, w0);
    load4(high, w1);
    fr_mul(a, w0, pa);
    fr_mul(b, w1, pb);
    fr_add(pa, pb, s);
    store4(out + (unsigned long long)i * LIMBS, s);
}

extern "C" __global__ void bap_message_kernel(const u64 *__restrict__ linear,
                                              const u64 *__restrict__ squared,
                                              const u64 *__restrict__ rho, unsigned int polys,
                                              unsigned int half, const u64 *__restrict__ e_in,
                                              unsigned int e_in_len,
                                              const u64 *__restrict__ e_out,
                                              unsigned int num_x_in_bits,
                                              u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long y = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[OHF_MAX_LANES][LIMBS];
    for (int lane = 0; lane < OHF_MAX_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (y < half) {
        unsigned long long len = 2ull * (unsigned long long)half;
        u64 fold_zero[2 * PA_SLOTS], fold_lead[2 * PA_SLOTS];
        pa_zero(fold_zero);
        pa_zero(fold_lead);

        for (unsigned int p = 0; p < polys; p++) {
            const u64 *a = linear + (unsigned long long)p * len * LIMBS;
            const u64 *b = squared + (unsigned long long)p * len * LIMBS;
            u64 a0[LIMBS], b0[LIMBS], b1[LIMBS], w[LIMBS], diff[LIMBS], total[LIMBS];
            load4(a + (2 * y) * LIMBS, a0);
            load4(b + (2 * y) * LIMBS, b0);
            load4(b + (2 * y + 1) * LIMBS, b1);
            load4(rho + (unsigned long long)p * LIMBS, w);
            fr_sub(b0, a0, diff);
            fr_add(b0, b1, total);
            pa_fold_mul_accum(w, diff, fold_zero);
            pa_fold_mul_accum(w, total, fold_lead);
        }

        u64 q0[LIMBS], q2[LIMBS], weight[LIMBS];
        pa_finalize(fold_zero, q0);
        pa_finalize(fold_lead, q2);
        ohf_weight(e_in, e_in_len, e_out, num_x_in_bits, y, weight);
        fr_mul(q0, weight, acc[0]);
        fr_mul(q2, weight, acc[1]);
    }

    ohf_block_reduce(scratch, 2, acc, partials);
}
