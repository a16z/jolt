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

extern "C" __global__ void twos_i128_to_mont_kernel(const u64 *__restrict__ value,
                                                    u64 *__restrict__ out,
                                                    unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 lo = value[2 * i];
    u64 hi = value[2 * i + 1];
    bool negative = (hi >> 63) != 0ull;
    if (negative) {
        u64 next = ~lo + 1ull;
        hi = ~hi + (next == 0ull ? 1ull : 0ull);
        lo = next;
    }
    u64 raw[LIMBS] = {lo, hi, 0, 0};
    u64 r[LIMBS];
    fr_to_mont(raw, r);
    if (negative) {
        u64 zero[LIMBS] = {0, 0, 0, 0};
        u64 neg[LIMBS];
        fr_sub(zero, r, neg);
        store4(out + i * LIMBS, neg);
    } else {
        store4(out + i * LIMBS, r);
    }
}

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

extern "C" __global__ void fr_delta_u64_kernel(const u64 *__restrict__ lo,
                                              const u64 *__restrict__ hi, unsigned int n,
                                              u64 *__restrict__ out) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    u64 lo_raw[LIMBS] = {lo[j], 0, 0, 0};
    u64 hi_raw[LIMBS] = {hi[j], 0, 0, 0};
    u64 lo_mont[LIMBS], hi_mont[LIMBS], delta[LIMBS];
    fr_to_mont(lo_raw, lo_mont);
    fr_to_mont(hi_raw, hi_mont);
    fr_sub(hi_mont, lo_mont, delta);
    store4(out + (unsigned long long)j * LIMBS, delta);
}
