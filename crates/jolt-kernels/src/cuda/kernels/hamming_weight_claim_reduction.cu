extern "C" __global__ void hwr_weights_kernel(const u64 *__restrict__ eq_booleanity,
                                              const u64 *__restrict__ eq_virtualization,
                                              const u64 *__restrict__ powers,
                                              u64 *__restrict__ out, unsigned int polys,
                                              unsigned int addresses) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= polys * addresses) return;
    unsigned int p = idx / addresses;
    unsigned int k = idx - p * addresses;

    u64 hamming[LIMBS], booleanity[LIMBS], virtualization[LIMBS];
    load4(powers + (3ull * p) * LIMBS, hamming);
    load4(powers + (3ull * p + 1) * LIMBS, booleanity);
    load4(powers + (3ull * p + 2) * LIMBS, virtualization);

    u64 eq_bool[LIMBS], eq_virt[LIMBS];
    load4(eq_booleanity + (unsigned long long)k * LIMBS, eq_bool);
    load4(eq_virtualization + (unsigned long long)idx * LIMBS, eq_virt);

    u64 weighted_bool[LIMBS], weighted_virt[LIMBS], partial[LIMBS], total[LIMBS];
    fr_mul(booleanity, eq_bool, weighted_bool);
    fr_mul(virtualization, eq_virt, weighted_virt);
    fr_add(hamming, weighted_bool, partial);
    fr_add(partial, weighted_virt, total);
    store4(out + (unsigned long long)idx * LIMBS, total);
}

extern "C" __global__ void hwr_message_kernel(const u64 *__restrict__ folded,
                                              const u64 *__restrict__ weights, unsigned int polys,
                                              unsigned int half, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long y = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[OHF_MAX_LANES][LIMBS];
    for (int lane = 0; lane < OHF_MAX_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (y < half) {
        unsigned long long len = 2ull * (unsigned long long)half;
        u64 fold_one[2 * PA_SLOTS], fold_infinity[2 * PA_SLOTS];
        pa_zero(fold_one);
        pa_zero(fold_infinity);

        for (unsigned int p = 0; p < polys; p++) {
            const u64 *folded_row = folded + (unsigned long long)p * len * LIMBS;
            const u64 *weight_row = weights + (unsigned long long)p * len * LIMBS;
            u64 g0[LIMBS], g1[LIMBS], w0[LIMBS], w1[LIMBS], gd[LIMBS], wd[LIMBS];
            load4(folded_row + (2 * y) * LIMBS, g0);
            load4(folded_row + (2 * y + 1) * LIMBS, g1);
            load4(weight_row + (2 * y) * LIMBS, w0);
            load4(weight_row + (2 * y + 1) * LIMBS, w1);
            pa_fold_mul_accum(g1, w1, fold_one);
            fr_sub(g1, g0, gd);
            fr_sub(w1, w0, wd);
            pa_fold_mul_accum(gd, wd, fold_infinity);
        }

        pa_finalize(fold_one, acc[0]);
        pa_finalize(fold_infinity, acc[1]);
    }

    lane_block_reduce(scratch, 2, acc, partials);
}
