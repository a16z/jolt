extern "C" __global__ void mul_scalar(
    u64 *__restrict__ io,
    const u64 *__restrict__ scalar,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], s[4];
        load4(io + i * 4, x);
        load4(scalar, s);
        fr_mul(x, s, x);
        store4(io + i * 4, x);
    }
}
