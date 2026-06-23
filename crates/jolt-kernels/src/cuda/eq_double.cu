extern "C" __global__ void eq_double(u64 *__restrict__ out, const u64 *__restrict__ in, const u64 *__restrict__ challenge, unsigned long size_in) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_in) {
        u64 scalar[4], c[4], hi[4], lo[4];
        load4(in + i * 4, scalar);
        load4(challenge, c);
        fr_mul(scalar, c, hi);
        fr_sub(scalar, hi, lo);
        store4(out + (i * 2 + 1) * 4, hi);
        store4(out + (i * 2) * 4, lo);
    }
}
