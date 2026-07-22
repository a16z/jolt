extern "C" __global__ void u64_to_mont(
    u64 *__restrict__ out,
    const u64 *__restrict__ in,
    unsigned long n
) {
    unsigned long i = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 raw[4] = {in[i], 0, 0, 0};
    u64 mont[4];
    raf_to_mont(raw, mont);
    for (int k = 0; k < 4; k++) out[i * 4 + k] = mont[k];
}
