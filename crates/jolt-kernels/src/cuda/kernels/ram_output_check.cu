extern "C" __global__ void roc_message_kernel(const u64 *__restrict__ tables, unsigned int half,
                                              const u64 *__restrict__ e_in, unsigned int e_in_len,
                                              const u64 *__restrict__ e_out,
                                              unsigned int num_x_in_bits,
                                              u64 *__restrict__ partials) {
    extern __shared__ u64 scratch[];
    unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    u64 acc[OHF_MAX_LANES][LIMBS];
    for (int lane = 0; lane < OHF_MAX_LANES; lane++)
        for (int l = 0; l < LIMBS; l++) acc[lane][l] = 0;

    if (g < half) {
        unsigned long long len = 2ull * (unsigned long long)half;
        const u64 *mask = tables;
        const u64 *val_final = tables + len * LIMBS;
        const u64 *val_io = tables + 2ull * len * LIMBS;

        u64 io0[LIMBS], io1[LIMBS], vf0[LIMBS], vf1[LIMBS], vio0[LIMBS], vio1[LIMBS];
        load4(mask + (2 * g) * LIMBS, io0);
        load4(mask + (2 * g + 1) * LIMBS, io1);
        load4(val_final + (2 * g) * LIMBS, vf0);
        load4(val_final + (2 * g + 1) * LIMBS, vf1);
        load4(val_io + (2 * g) * LIMBS, vio0);
        load4(val_io + (2 * g + 1) * LIMBS, vio1);

        u64 v0[LIMBS], v1[LIMBS], constant[LIMBS], mask_delta[LIMBS], value_delta[LIMBS],
            quadratic[LIMBS], weight[LIMBS];
        fr_sub(vf0, vio0, v0);
        fr_sub(vf1, vio1, v1);
        fr_mul(io0, v0, constant);
        fr_sub(io1, io0, mask_delta);
        fr_sub(v1, v0, value_delta);
        fr_mul(mask_delta, value_delta, quadratic);

        eq_split_weight(e_in, e_in_len, e_out, num_x_in_bits, g, weight);
        fr_mul(constant, weight, acc[0]);
        fr_mul(quadratic, weight, acc[1]);
    }

    lane_block_reduce(scratch, 2, acc, partials);
}
