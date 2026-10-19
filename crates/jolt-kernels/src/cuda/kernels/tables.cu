__device__ __constant__ u64 FR_ONE[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

extern "C" __global__ void eq_double_kernel(const u64 *__restrict__ in,
                                            u64 r0, u64 r1, u64 r2, u64 r3,
                                            u64 *__restrict__ out,
                                            unsigned int prev_len) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prev_len) return;
    u64 base[LIMBS], one[LIMBS], one_minus_r[LIMBS], lo[LIMBS], hi[LIMBS];
    u64 r[LIMBS] = {r0, r1, r2, r3};
    load4(in + j * LIMBS, base);
    load4(FR_ONE, one);
    fr_sub(one, r, one_minus_r);
    fr_mul(base, one_minus_r, lo);
    fr_mul(base, r, hi);
    store4(out + (2 * j) * LIMBS, lo);
    store4(out + (2 * j + 1) * LIMBS, hi);
}

extern "C" __global__ void lt_double_kernel(u64 *__restrict__ evals,
                                            u64 r0, u64 r1, u64 r2, u64 r3,
                                            unsigned int half) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    u64 x[LIMBS], y[LIMBS], diff[LIMBS], nx[LIMBS];
    u64 r[LIMBS] = {r0, r1, r2, r3};
    load4(evals + j * LIMBS, x);
    fr_mul(x, r, y);
    fr_sub(r, y, diff);
    fr_add(x, diff, nx);
    store4(evals + (j + half) * LIMBS, y);
    store4(evals + j * LIMBS, nx);
}
