#define OPN_NO_HOT 0xFFFFFFFFu

extern "C" __global__ void opening_one_hot_embed_kernel(const unsigned int *__restrict__ hot,
                                                        unsigned long long cycles,
                                                        unsigned long long domain,
                                                        u64 *__restrict__ out) {
    unsigned long long cycle = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cycle >= cycles) return;
    unsigned int address = hot[cycle];
    if (address == OPN_NO_HOT) return;
    unsigned long long index = (unsigned long long)address * cycles + cycle;
    if (index >= domain) return;
    u64 one[LIMBS];
    load4(FR_ONE, one);
    store4(out + index * LIMBS, one);
}

extern "C" __global__ void opening_one_hot_fold_kernel(const unsigned int *__restrict__ hot,
                                                       const u64 *__restrict__ left,
                                                       unsigned long long cycles,
                                                       unsigned long long base,
                                                       unsigned long long len,
                                                       unsigned long long columns,
                                                       unsigned int sigma, unsigned long long rows,
                                                       u64 *__restrict__ out) {
    unsigned long long column = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    u64 acc[LIMBS] = {0, 0, 0, 0};
    unsigned long long mask = columns - 1ull;
    unsigned long long first = (column + columns - (base & mask)) & mask;
    for (unsigned long long local = first; local < len; local += columns) {
        unsigned int address = hot[local];
        if (address == OPN_NO_HOT) continue;
        unsigned long long index = (unsigned long long)address * cycles + base + local;
        unsigned long long row = index >> sigma;
        if (row >= rows) continue;
        u64 weight[LIMBS], sum[LIMBS];
        load4(left + row * LIMBS, weight);
        fr_add(acc, weight, sum);
        for (int limb = 0; limb < LIMBS; limb++) acc[limb] = sum[limb];
    }
    store4(out + column * LIMBS, acc);
}
