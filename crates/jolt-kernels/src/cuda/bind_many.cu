extern "C" __global__ void bind_many_kernel(
    u64 *const *__restrict__ out_ptrs,
    const u64 *const *__restrict__ in_ptrs,
    const u64 *__restrict__ challenge,
    unsigned long half,
    unsigned long num_polys
) {
    unsigned long t = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = num_polys * half;
    if (t < total) {
        unsigned long poly = t / half;
        unsigned long i = t % half;
        const u64 *in = in_ptrs[poly];
        u64 *out = out_ptrs[poly];
        u64 lo[4], hi[4], c[4];
        load4(in + (i * 2) * 4, lo);
        load4(in + (i * 2 + 1) * 4, hi);
        load4(challenge, c);
        u64 diff[4];
        fr_sub(hi, lo, diff);
        fr_mul(diff, c, diff);
        fr_add(lo, diff, lo);
        store4(out + i * 4, lo);
    }
}
