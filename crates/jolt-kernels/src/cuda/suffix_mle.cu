__device__ __forceinline__ u64 suffix_mle_eval(
    u128 bits,
    unsigned int len,
    unsigned int variant
) {
    (void)bits;
    (void)len;
    (void)variant;
    return 0;
}

extern "C" __global__ void suffix_mle_probe(
    u64 *__restrict__ out,
    const u64 *__restrict__ bits_lo,
    const u64 *__restrict__ bits_hi,
    const unsigned int *__restrict__ len,
    const unsigned int *__restrict__ variant,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u128 bits = ((u128)bits_hi[i] << 64) | (u128)bits_lo[i];
    out[i] = suffix_mle_eval(bits, len[i], variant[i]);
}
