__device__ __forceinline__ void combine_eval(
    unsigned int variant,
    const u64 *__restrict__ prefixes,
    const u64 *__restrict__ suffixes,
    unsigned int suffix_count,
    u64 *out
) {
    for (int k = 0; k < 4; k++) out[k] = 0;
}

extern "C" __global__ void prefix_combine_probe(
    u64 *__restrict__ out,
    const u64 *__restrict__ prefixes,
    const u64 *__restrict__ suffixes,
    const unsigned int *__restrict__ suffix_count,
    const unsigned int *__restrict__ variant,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    combine_eval(
        variant[i],
        prefixes + i * 46 * 4,
        suffixes + i * 4 * 4,
        suffix_count[i],
        out + i * 4
    );
}
