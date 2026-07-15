extern "C" __global__ void batched_bind_high_to_low_kernel(
    u64 *__restrict__ out,
    const u64 *__restrict__ values,
    const u64 *__restrict__ challenge,
    unsigned long half,
    unsigned long num_polys
) {
    unsigned long t = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_polys * half;
    if (t < total) {
        unsigned long poly = t / half;
        unsigned long i = t % half;
        unsigned long len = half * 2;
        const u64 *in_base = values + poly * len * 4;
        u64 lo[4], hi[4], c[4];
        load4(in_base + i * 4, lo);
        load4(in_base + (i + half) * 4, hi);
        load4(challenge, c);
        u64 diff[4];
        fr_sub(hi, lo, diff);
        fr_mul(diff, c, diff);
        fr_add(lo, diff, lo);
        store4(out + (poly * half + i) * 4, lo);
    }
}
