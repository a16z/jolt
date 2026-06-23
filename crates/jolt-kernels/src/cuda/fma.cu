extern "C" __global__ void fma_kernel(u64 *__restrict__ io, const u64 *__restrict__ b, const u64 *__restrict__ c, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], y[4], z[4];
        load4(io + i * 4, x);
        load4(b + i * 4, y);
        load4(c + i * 4, z);
        fr_mul(x, y, x);
        fr_add(x, z, x);
        store4(io + i * 4, x);
    }
}
