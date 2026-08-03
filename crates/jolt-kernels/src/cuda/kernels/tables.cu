__device__ __constant__ u64 FR_R2[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
};
__device__ __constant__ u64 FR_ONE[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

__device__ __forceinline__ void fr_to_mont(const u64 *raw, u64 *out) {
    u64 r2[LIMBS];
    load4(FR_R2, r2);
    fr_mul(raw, r2, out);
}

extern "C" __global__ void u64_to_mont_kernel(const u64 *__restrict__ in,
                                             u64 *__restrict__ out,
                                             unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 raw[LIMBS] = {in[i], 0, 0, 0};
    u64 r[LIMBS];
    fr_to_mont(raw, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void i128_to_mont_kernel(const u64 *__restrict__ magnitude,
                                               const unsigned char *__restrict__ negative,
                                               u64 *__restrict__ out,
                                               unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 raw[LIMBS] = {magnitude[2 * i], magnitude[2 * i + 1], 0, 0};
    u64 r[LIMBS];
    fr_to_mont(raw, r);
    if (negative[i]) {
        u64 zero[LIMBS] = {0, 0, 0, 0};
        u64 neg[LIMBS];
        fr_sub(zero, r, neg);
        store4(out + i * LIMBS, neg);
    } else {
        store4(out + i * LIMBS, r);
    }
}

extern "C" __global__ void eq_double_kernel(const u64 *__restrict__ in,
                                            const u64 *__restrict__ r_i,
                                            u64 *__restrict__ out,
                                            unsigned int prev_len) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= prev_len) return;
    u64 base[LIMBS], r[LIMBS], one[LIMBS], one_minus_r[LIMBS], lo[LIMBS], hi[LIMBS];
    load4(in + j * LIMBS, base);
    load4(r_i, r);
    load4(FR_ONE, one);
    fr_sub(one, r, one_minus_r);
    fr_mul(base, one_minus_r, lo);
    fr_mul(base, r, hi);
    store4(out + (2 * j) * LIMBS, lo);
    store4(out + (2 * j + 1) * LIMBS, hi);
}

extern "C" __global__ void lt_double_kernel(u64 *__restrict__ evals,
                                            const u64 *__restrict__ r_i,
                                            unsigned int half) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    u64 x[LIMBS], r[LIMBS], y[LIMBS], diff[LIMBS], nx[LIMBS];
    load4(evals + j * LIMBS, x);
    load4(r_i, r);
    fr_mul(x, r, y);
    fr_sub(r, y, diff);
    fr_add(x, diff, nx);
    store4(evals + (j + half) * LIMBS, y);
    store4(evals + j * LIMBS, nx);
}
