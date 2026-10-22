extern "C" __global__ void irv_eq_double_kernel(const u64 *__restrict__ in,
                                                const u64 *__restrict__ point,
                                                u64 *__restrict__ out, unsigned int polys,
                                                unsigned int prev_len, unsigned int level,
                                                unsigned int chunk_bits) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= polys * prev_len) return;
    unsigned int p = idx / prev_len;
    unsigned int i = idx - p * prev_len;

    u64 base[LIMBS], r[LIMBS], one[LIMBS], one_minus_r[LIMBS], lo[LIMBS], hi[LIMBS];
    load4(in + (unsigned long long)idx * LIMBS, base);
    load4(point + (unsigned long long)(p * chunk_bits + level) * LIMBS, r);
    load4(FR_ONE, one);
    fr_sub(one, r, one_minus_r);
    fr_mul(base, one_minus_r, lo);
    fr_mul(base, r, hi);

    unsigned long long out_base = (unsigned long long)p * 2 * prev_len + 2 * i;
    store4(out + out_base * LIMBS, lo);
    store4(out + (out_base + 1) * LIMBS, hi);
}

extern "C" __global__ void irv_tables_split_kernel(const u64 *__restrict__ in,
                                                   const u64 *__restrict__ eq_zero,
                                                   const u64 *__restrict__ eq_one,
                                                   u64 *__restrict__ out, unsigned int polys,
                                                   unsigned int len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= polys * len) return;
    unsigned int p = idx / len;
    unsigned int i = idx - p * len;

    u64 base[LIMBS], zero[LIMBS], one[LIMBS], lo[LIMBS], hi[LIMBS];
    load4(in + (unsigned long long)idx * LIMBS, base);
    load4(eq_zero, zero);
    load4(eq_one, one);
    fr_mul(base, zero, lo);
    fr_mul(base, one, hi);

    unsigned long long out_base = (unsigned long long)p * 2 * len;
    store4(out + (out_base + i) * LIMBS, lo);
    store4(out + (out_base + len + i) * LIMBS, hi);
}
