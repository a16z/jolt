extern "C" __global__ void add_kernel(const u64 *__restrict__ a,
                                     const u64 *__restrict__ b,
                                     u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x[LIMBS], y[LIMBS], r[LIMBS];
    load4(a + i * LIMBS, x);
    load4(b + i * LIMBS, y);
    fr_add(x, y, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void sub_kernel(const u64 *__restrict__ a,
                                     const u64 *__restrict__ b,
                                     u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x[LIMBS], y[LIMBS], r[LIMBS];
    load4(a + i * LIMBS, x);
    load4(b + i * LIMBS, y);
    fr_sub(x, y, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void mul_kernel(const u64 *__restrict__ a,
                                     const u64 *__restrict__ b,
                                     u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x[LIMBS], y[LIMBS], r[LIMBS];
    load4(a + i * LIMBS, x);
    load4(b + i * LIMBS, y);
    fr_mul(x, y, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void mul_scalar_kernel(const u64 *__restrict__ a,
                                            const u64 *__restrict__ scalar,
                                            u64 *__restrict__ out,
                                            unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x[LIMBS], s[LIMBS], r[LIMBS];
    load4(a + i * LIMBS, x);
    load4(scalar, s);
    fr_mul(x, s, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void add_scalar_kernel(const u64 *__restrict__ a,
                                            const u64 *__restrict__ scalar,
                                            u64 *__restrict__ out,
                                            unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x[LIMBS], s[LIMBS], r[LIMBS];
    load4(a + i * LIMBS, x);
    load4(scalar, s);
    fr_add(x, s, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void fma_kernel(const u64 *__restrict__ acc,
                                      const u64 *__restrict__ a,
                                      const u64 *__restrict__ b,
                                      u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 c[LIMBS], x[LIMBS], y[LIMBS], p[LIMBS], r[LIMBS];
    load4(acc + i * LIMBS, c);
    load4(a + i * LIMBS, x);
    load4(b + i * LIMBS, y);
    fr_mul(x, y, p);
    fr_add(c, p, r);
    store4(out + i * LIMBS, r);
}
