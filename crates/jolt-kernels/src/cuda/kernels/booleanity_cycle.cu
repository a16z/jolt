__device__ __forceinline__ void brc_accumulate(const u64 *h0, const u64 *h1,
                                               const u64 *__restrict__ rho, unsigned int p,
                                               u64 *fold_c, u64 *fold_e, u64 *fold_one) {
    u64 scale[LIMBS], diff[LIMBS], delta[LIMBS];
    load4(rho + (unsigned long long)p * LIMBS, scale);
    fr_sub(h0, scale, diff);
    pa_fold_mul_accum(h0, diff, fold_c);
    fr_sub(h1, scale, diff);
    pa_fold_mul_accum(h1, diff, fold_one);
    fr_sub(h1, h0, delta);
    pa_fold_mul_accum(delta, delta, fold_e);
}

__device__ __forceinline__ void brc_finish(const u64 *fold_c, const u64 *fold_e, const u64 *fold_one,
                                           const u64 *__restrict__ e_in, unsigned int e_in_len,
                                           const u64 *__restrict__ e_out,
                                           unsigned int num_x_in_bits, unsigned long long g,
                                           u64 acc[3][LIMBS]) {
    u64 combined[LIMBS], constant[LIMBS], quadratic[LIMBS];
    eq_split_weight(e_in, e_in_len, e_out, num_x_in_bits, g, combined);
    pa_finalize(fold_c, constant);
    pa_finalize(fold_e, quadratic);
    fr_mul(constant, combined, acc[0]);
    fr_mul(quadratic, combined, acc[1]);
    pa_finalize(fold_one, constant);
    fr_mul(constant, combined, acc[2]);
}

extern "C" __global__ void brc_message_dense_kernel(
    const u64 *__restrict__ dense, const u64 *__restrict__ rho, unsigned int polys,
    unsigned int half, const u64 *__restrict__ e_in, unsigned int e_in_len,
    const u64 *__restrict__ e_out, unsigned int num_x_in_bits, u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[3][LIMBS];
    for (int lane = 0; lane < 3; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        u64 fold_c[2 * PA_SLOTS], fold_e[2 * PA_SLOTS], fold_one[2 * PA_SLOTS];
        pa_zero(fold_c);
        pa_zero(fold_e);
        pa_zero(fold_one);

        for (unsigned int p = 0; p < polys; p++) {
            const u64 *row = dense + (unsigned long long)p * 2 * half * LIMBS;
            u64 h0[LIMBS], h1[LIMBS];
            load4(row + (unsigned long long)(2 * g) * LIMBS, h0);
            load4(row + (unsigned long long)(2 * g + 1) * LIMBS, h1);
            brc_accumulate(h0, h1, rho, p, fold_c, fold_e, fold_one);
        }

        brc_finish(fold_c, fold_e, fold_one, e_in, e_in_len, e_out, num_x_in_bits, g, acc);
    }

    lane_block_reduce(scratch, 3, acc, partials);
}
