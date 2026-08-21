#define UNR_SLOTS 7

__device__ __constant__ u64 UNR_MONT_2_64[4] = {
    13026360628005025158ULL, 8110012406944207806ULL,
    11528403486311440623ULL, 2403517766477359206ULL
};


__device__ __forceinline__ void unr_zero(u64 *folded) {
    for (int i = 0; i < 2 * UNR_SLOTS; i++) folded[i] = 0;
}

__device__ __forceinline__ void unr_add_folded(u64 *folded, const u64 *other) {
    for (int i = 0; i < 2 * UNR_SLOTS; i++) folded[i] += other[i];
}

__device__ __forceinline__ void unr_mul_words(const u64 *value,
                                              const unsigned long long *words,
                                              unsigned int word_count,
                                              u64 *folded) {
    for (unsigned int w = 0; w < word_count; w++) {
        unsigned long long m = words[w];
        if (m == 0ULL) continue;
        u64 carry = 0;
        for (int i = 0; i < LIMBS; i++) {
            u128 t = (u128)value[i] * (u128)m + (u128)carry;
            unsigned int slot = w + (unsigned int)i;
            unsigned long long piece = (unsigned long long)t;
            folded[2 * slot] += (piece & 0xFFFFFFFFULL);
            folded[2 * slot + 1] += (piece >> 32);
            carry = (u64)(t >> 64);
        }
        unsigned int slot = w + LIMBS;
        folded[2 * slot] += (carry & 0xFFFFFFFFULL);
        folded[2 * slot + 1] += (carry >> 32);
    }
}

__device__ __forceinline__ void unr_add_field(u64 *folded, const u64 *value) {
    for (int i = 0; i < LIMBS; i++) {
        unsigned long long piece = (unsigned long long)value[i];
        folded[2 * i] += (piece & 0xFFFFFFFFULL);
        folded[2 * i + 1] += (piece >> 32);
    }
}

__device__ __forceinline__ void unr_finalize(const u64 *folded, u64 *out) {
    u64 limbs[UNR_SLOTS + 1];
    u64 carry = 0;
    for (int i = 0; i < UNR_SLOTS; i++) {
        u128 t = (u128)folded[2 * i] + ((u128)folded[2 * i + 1] << 32) + (u128)carry;
        limbs[i] = (u64)t;
        carry = (u64)(t >> 64);
    }
    limbs[UNR_SLOTS] = carry;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    u64 scale[LIMBS];
    load4(UNR_MONT_2_64, scale);
    for (int i = UNR_SLOTS; i >= 0; i--) {
        u64 scaled[LIMBS];
        fr_mul(acc, scale, scaled);
        u64 addend[LIMBS] = {limbs[i], 0, 0, 0};
        fr_add(scaled, addend, acc);
    }
    store4(out, acc);
}

__device__ __forceinline__ void unr_scatter_add(u64 *slots, const u64 *product, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        unsigned long long lo = (unsigned long long)product[2 * i];
        unsigned long long hi = (unsigned long long)product[2 * i + 1];
        if (lo != 0ULL) atomicAdd((unsigned long long *)&slots[2 * i], lo);
        if (hi != 0ULL) atomicAdd((unsigned long long *)&slots[2 * i + 1], hi);
    }
}

extern "C" __global__ void unr_mul_scatter_kernel(const u64 *__restrict__ values,
                                                  const unsigned long long *__restrict__ mults,
                                                  unsigned int mult_words,
                                                  const unsigned int *__restrict__ buckets,
                                                  u64 *__restrict__ slots,
                                                  unsigned int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    u64 folded[2 * UNR_SLOTS];
    for (int i = 0; i < 2 * UNR_SLOTS; i++) folded[i] = 0;

    u64 v[LIMBS];
    load4(values + (unsigned long long)j * LIMBS, v);

    bool nonzero = false;
    for (unsigned int w = 0; w < mult_words; w++) {
        unsigned long long m = mults[(unsigned long long)j * mult_words + w];
        if (m == 0ULL) continue;
        nonzero = true;
        u64 carry = 0;
        for (int i = 0; i < LIMBS; i++) {
            u128 t = (u128)v[i] * (u128)m + (u128)carry;
            unsigned int slot = w + (unsigned int)i;
            unsigned long long piece = (unsigned long long)t;
            folded[2 * slot] += (piece & 0xFFFFFFFFULL);
            folded[2 * slot + 1] += (piece >> 32);
            carry = (u64)(t >> 64);
        }
        unsigned int slot = w + LIMBS;
        folded[2 * slot] += (carry & 0xFFFFFFFFULL);
        folded[2 * slot + 1] += (carry >> 32);
    }
    if (!nonzero) return;

    u64 *target = slots + (unsigned long long)buckets[j] * (2 * UNR_SLOTS);
    unr_scatter_add(target, folded, UNR_SLOTS);
}

extern "C" __global__ void unr_fold_chunks_kernel(const u64 *__restrict__ slots,
                                                  unsigned int chunks, unsigned int groups,
                                                  u64 *__restrict__ out) {
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;

    u64 acc[2 * UNR_SLOTS];
    unr_zero(acc);
    const u64 *base = slots + (unsigned long long)g * chunks * (2 * UNR_SLOTS);
    for (unsigned int c = 0; c < chunks; c++) {
        unr_add_folded(acc, base + (unsigned long long)c * (2 * UNR_SLOTS));
    }

    u64 *target = out + (unsigned long long)g * (2 * UNR_SLOTS);
    for (int i = 0; i < 2 * UNR_SLOTS; i++) target[i] = acc[i];
}

extern "C" __global__ void unr_reduce_kernel(const u64 *__restrict__ slots,
                                             u64 *__restrict__ out,
                                             unsigned int bucket_count) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= bucket_count) return;

    const u64 *half = slots + (unsigned long long)b * (2 * UNR_SLOTS);
    u64 limbs[UNR_SLOTS + 1];
    for (int i = 0; i < UNR_SLOTS + 1; i++) limbs[i] = 0;

    u64 carry = 0;
    for (int i = 0; i < UNR_SLOTS; i++) {
        u128 t = (u128)half[2 * i] + ((u128)half[2 * i + 1] << 32) + (u128)carry;
        limbs[i] = (u64)t;
        carry = (u64)(t >> 64);
    }
    limbs[UNR_SLOTS] = carry;

    u64 acc[LIMBS] = {0, 0, 0, 0};
    u64 scale[LIMBS];
    load4(UNR_MONT_2_64, scale);
    for (int i = UNR_SLOTS; i >= 0; i--) {
        u64 scaled[LIMBS];
        fr_mul(acc, scale, scaled);
        u64 addend[LIMBS] = {limbs[i], 0, 0, 0};
        fr_add(scaled, addend, acc);
    }
    store4(out + (unsigned long long)b * LIMBS, acc);
}
