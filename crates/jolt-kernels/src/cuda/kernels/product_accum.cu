#define PA_SLOTS 8

__device__ __forceinline__ void pa_add_piece(u64 *folded, unsigned int slot,
                                             unsigned long long piece) {
    folded[2 * slot] += (piece & 0xFFFFFFFFULL);
    folded[2 * slot + 1] += (piece >> 32);
}

__device__ __forceinline__ void pa_fold_mul(const u64 *a, const u64 *b, u64 *folded) {
    for (int i = 0; i < 2 * PA_SLOTS; i++) folded[i] = 0;
    for (int i = 0; i < LIMBS; i++) {
        for (int j = 0; j < LIMBS; j++) {
            u128 p = (u128)a[i] * (u128)b[j];
            pa_add_piece(folded, (unsigned int)(i + j), (unsigned long long)p);
            pa_add_piece(folded, (unsigned int)(i + j + 1),
                         (unsigned long long)(p >> 64));
        }
    }
}

__device__ __forceinline__ void pa_fold_mul_accum(const u64 *a, const u64 *b, u64 *folded) {
    for (int i = 0; i < LIMBS; i++) {
        for (int j = 0; j < LIMBS; j++) {
            u128 p = (u128)a[i] * (u128)b[j];
            pa_add_piece(folded, (unsigned int)(i + j), (unsigned long long)p);
            pa_add_piece(folded, (unsigned int)(i + j + 1),
                         (unsigned long long)(p >> 64));
        }
    }
}

__device__ __forceinline__ void pa_zero(u64 *folded) {
    for (int i = 0; i < 2 * PA_SLOTS; i++) folded[i] = 0;
}

__device__ __forceinline__ void pa_finalize(const u64 *folded, u64 *out) {
    u64 limbs[PA_SLOTS + 1];
    u128 carry = 0;
    for (int i = 0; i < PA_SLOTS; i++) {
        u128 t = (u128)folded[2 * i] + ((u128)folded[2 * i + 1] << 32) + carry;
        limbs[i] = (u64)t;
        carry = t >> 64;
    }
    limbs[PA_SLOTS] = (u64)carry;

    // Products carry two Montgomery factors: reduce x / R as
    // low / R + middle + high * R, where R = 2^256.
    u64 low[LIMBS], middle[LIMBS], high[LIMBS];
    u64 raw_one[LIMBS] = {1, 0, 0, 0};
    u64 top[LIMBS] = {limbs[PA_SLOTS], 0, 0, 0};
    fr_reduce_256(limbs, low);
    fr_reduce_256(limbs + LIMBS, middle);
    fr_mul(low, raw_one, low);
    fr_mul(top, FR_R2, high);
    fr_add(low, middle, out);
    fr_add(out, high, out);
}
