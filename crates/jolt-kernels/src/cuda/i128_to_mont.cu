extern "C" __global__ void i128_to_mont(
    u64 *__restrict__ out,
    const u64 *__restrict__ abs_lo,
    const u64 *__restrict__ abs_hi,
    const unsigned char *__restrict__ neg,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 raw[4] = {abs_lo[i], abs_hi[i], 0, 0};
    u64 mont[4];
    raf_to_mont(raw, mont);
    if (neg[i]) {
        u64 zero[4] = {0, 0, 0, 0};
        u64 res[4];
        fr_sub(zero, mont, res);
        for (int k = 0; k < 4; k++) out[i * 4 + k] = res[k];
    } else {
        for (int k = 0; k < 4; k++) out[i * 4 + k] = mont[k];
    }
}
