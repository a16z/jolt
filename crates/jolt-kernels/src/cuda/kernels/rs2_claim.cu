extern "C" __global__ void rs2_claim_kernel(const unsigned int *__restrict__ indices,
                                          const u64 *__restrict__ eq_cycle,
                                          const u64 *__restrict__ eq_address,
                                          unsigned int cycles, unsigned int addresses,
                                          u64 *__restrict__ out) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cycles) return;
    unsigned int index = indices[j];
    u64 acc[LIMBS] = {0, 0, 0, 0};
    if (index < addresses) {
        u64 c[LIMBS], a[LIMBS];
        load4(eq_cycle + (unsigned long long)j * LIMBS, c);
        load4(eq_address + (unsigned long long)index * LIMBS, a);
        fr_mul(c, a, acc);
    }
    store4(out + (unsigned long long)j * LIMBS, acc);
}
