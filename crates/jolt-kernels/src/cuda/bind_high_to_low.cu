extern "C" __global__ void bind_high_to_low(
    u64 *__restrict__ out,
    const u64 *__restrict__ values,
    const u64 *__restrict__ challenge,
    unsigned long half
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half) {
        u64 lo[4], hi[4], c[4];
        load4(values + i * 4, lo);
        load4(values + (i + half) * 4, hi);
        load4(challenge, c);
        u64 diff[4];
        fr_sub(hi, lo, diff);
        fr_mul(diff, c, diff);
        fr_add(lo, diff, lo);
        store4(out + i * 4, lo);
    }
}
