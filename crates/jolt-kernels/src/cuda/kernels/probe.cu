extern "C" __global__ void fr_identity_probe(const u64 *__restrict__ in,
                                            u64 *__restrict__ out,
                                            unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 a[LIMBS];
    load4(in + i * LIMBS, a);
    store4(out + i * LIMBS, a);
}
