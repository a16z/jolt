extern "C" __global__ void sub_kernel(u64 *__restrict__ io, const u64 *__restrict__ b, unsigned long n) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u64 x[4], y[4];
        load4(io + i * 4, x);
        load4(b + i * 4, y);
        fr_sub(x, y, x);
        store4(io + i * 4, x);
    }
}
