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

extern "C" __global__ void bind_low_to_high_kernel(const u64 *__restrict__ in,
                                                  const u64 *__restrict__ challenge,
                                                  u64 *__restrict__ out,
                                                  unsigned int half) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    u64 lo[LIMBS], hi[LIMBS], c[LIMBS], d[LIMBS], t[LIMBS], r[LIMBS];
    load4(in + (2 * i) * LIMBS, lo);
    load4(in + (2 * i + 1) * LIMBS, hi);
    load4(challenge, c);
    fr_sub(hi, lo, d);
    fr_mul(c, d, t);
    fr_add(lo, t, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void bind_high_to_low_kernel(const u64 *__restrict__ in,
                                                  const u64 *__restrict__ challenge,
                                                  u64 *__restrict__ out,
                                                  unsigned int half) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    u64 lo[LIMBS], hi[LIMBS], c[LIMBS], d[LIMBS], t[LIMBS], r[LIMBS];
    load4(in + i * LIMBS, lo);
    load4(in + (i + half) * LIMBS, hi);
    load4(challenge, c);
    fr_sub(hi, lo, d);
    fr_mul(c, d, t);
    fr_add(lo, t, r);
    store4(out + i * LIMBS, r);
}

extern "C" __global__ void sum_reduce_kernel(const u64 *__restrict__ in,
                                             u64 *__restrict__ partials,
                                             unsigned int n) {
    __shared__ u64 scratch[BLOCK * LIMBS];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    if (i < n) load4(in + i * LIMBS, acc);
    store4(scratch + tid * LIMBS, acc);
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            u64 x[LIMBS], y[LIMBS], r[LIMBS];
            load4(scratch + tid * LIMBS, x);
            load4(scratch + (tid + stride) * LIMBS, y);
            fr_add(x, y, r);
            store4(scratch + tid * LIMBS, r);
        }
        __syncthreads();
    }

    if (tid == 0) {
        u64 total[LIMBS];
        load4(scratch, total);
        store4(partials + blockIdx.x * LIMBS, total);
    }
}
