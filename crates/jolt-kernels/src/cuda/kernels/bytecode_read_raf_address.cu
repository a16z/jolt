extern "C" __global__ void brap_one_hot_kernel(u64 *__restrict__ out, unsigned int index,
                                              unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 value[LIMBS];
    if (i == index) {
        load4(FR_ONE, value);
    } else {
        for (int l = 0; l < LIMBS; l++) value[l] = 0;
    }
    store4(out + (unsigned long long)i * LIMBS, value);
}

extern "C" __global__ void brap_term_kernel(const u64 *__restrict__ table,
                                            const u64 *__restrict__ addend,
                                            const u64 *__restrict__ scales,
                                            u64 *__restrict__ out, unsigned int offset,
                                            unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    u64 value[LIMBS], factor[LIMBS], scaled[LIMBS];
    load4(table + (unsigned long long)i * LIMBS, value);
    load4(scales, factor);
    fr_mul(value, factor, scaled);

    u64 extra[LIMBS], weight[LIMBS], weighted[LIMBS], total[LIMBS];
    load4(addend + (unsigned long long)i * LIMBS, extra);
    load4(scales + LIMBS, weight);
    fr_mul(extra, weight, weighted);
    fr_add(scaled, weighted, total);
    store4(out + ((unsigned long long)offset + (unsigned long long)i) * LIMBS, total);
}

extern "C" __global__ void brap_message_kernel(const u64 *__restrict__ left,
                                               const u64 *__restrict__ right, unsigned int terms,
                                               unsigned int half, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long y = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[OHF_MAX_LANES][LIMBS];
    for (int lane = 0; lane < OHF_MAX_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (y < half) {
        unsigned long long len = 2ull * (unsigned long long)half;
        u64 fold_zero[2 * PA_SLOTS], fold_two[2 * PA_SLOTS];
        pa_zero(fold_zero);
        pa_zero(fold_two);

        for (unsigned int t = 0; t < terms; t++) {
            const u64 *l = left + (unsigned long long)t * len * LIMBS;
            const u64 *r = right + (unsigned long long)t * len * LIMBS;
            u64 l0[LIMBS], l1[LIMBS], r0[LIMBS], r1[LIMBS], l2[LIMBS], r2[LIMBS], doubled[LIMBS];
            load4(l + (2 * y) * LIMBS, l0);
            load4(l + (2 * y + 1) * LIMBS, l1);
            load4(r + (2 * y) * LIMBS, r0);
            load4(r + (2 * y + 1) * LIMBS, r1);
            fr_add(l1, l1, doubled);
            fr_sub(doubled, l0, l2);
            fr_add(r1, r1, doubled);
            fr_sub(doubled, r0, r2);
            pa_fold_mul_accum(l0, r0, fold_zero);
            pa_fold_mul_accum(l2, r2, fold_two);
        }

        pa_finalize(fold_zero, acc[0]);
        pa_finalize(fold_two, acc[1]);
    }

    lane_block_reduce(scratch, 2, acc, partials);
}
