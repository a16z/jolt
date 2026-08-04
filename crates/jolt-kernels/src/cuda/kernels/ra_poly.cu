#define RA_COLD 0xFFFFFFFFu

extern "C" __global__ void ra_split_tables_kernel(const u64 *__restrict__ in,
                                                  const u64 *__restrict__ eq_zero,
                                                  const u64 *__restrict__ eq_one,
                                                  u64 *__restrict__ out,
                                                  unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 value[LIMBS], z[LIMBS], o[LIMBS], lo[LIMBS], hi[LIMBS];
    load4(in + (unsigned long long)i * LIMBS, value);
    load4(eq_zero, z);
    load4(eq_one, o);
    fr_mul(value, z, lo);
    fr_mul(value, o, hi);
    store4(out + (unsigned long long)i * LIMBS, lo);
    store4(out + ((unsigned long long)n + (unsigned long long)i) * LIMBS, hi);
}

extern "C" __global__ void ra_gather_kernel(const unsigned int *__restrict__ indices,
                                            const u64 *__restrict__ tables,
                                            const unsigned int *__restrict__ bases,
                                            unsigned int slots,
                                            unsigned int addresses,
                                            unsigned int stride,
                                            u64 *__restrict__ out,
                                            unsigned int len) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= len) return;
    u64 total[LIMBS] = {0, 0, 0, 0};
    unsigned long long base = (unsigned long long)j * (unsigned long long)stride;
    for (unsigned int slot = 0; slot < slots; slot++) {
        unsigned int address = indices[base + (unsigned long long)bases[slot]];
        if (address == RA_COLD) continue;
        u64 value[LIMBS];
        load4(tables + ((unsigned long long)slot * (unsigned long long)addresses +
                        (unsigned long long)address) *
                           LIMBS,
              value);
        fr_add(total, value, total);
    }
    store4(out + (unsigned long long)j * LIMBS, total);
}
