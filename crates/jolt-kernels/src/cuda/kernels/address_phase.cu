extern "C" __global__ void ap_bind_lanes_kernel(const u64 *const *__restrict__ in_ptrs,
                                               u64 *const *__restrict__ out_ptrs,
                                               const unsigned int *__restrict__ counts,
                                               u64 c0, u64 c1, u64 c2, u64 c3,
                                               unsigned int half,
                                               unsigned int stride,
                                               unsigned int max_count) {
    unsigned int lane = blockIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_count || i >= counts[lane]) return;
    const u64 *in = in_ptrs[lane];
    u64 *out = out_ptrs[lane];
    unsigned int column = i / half;
    unsigned int b = i % half;
    const u64 *base = in + (unsigned long long)column * stride * LIMBS;
    u64 lo[LIMBS], hi[LIMBS], d[LIMBS], t[LIMBS], r[LIMBS];
    u64 c[LIMBS] = {c0, c1, c2, c3};
    load4(base + (unsigned long long)b * LIMBS, lo);
    load4(base + ((unsigned long long)b + half) * LIMBS, hi);
    fr_sub(hi, lo, d);
    fr_mul(c, d, t);
    fr_add(lo, t, r);
    store4(out + (unsigned long long)i * LIMBS, r);
}
