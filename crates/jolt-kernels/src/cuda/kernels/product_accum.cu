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

    u64 acc[LIMBS] = {0, 0, 0, 0};
    u64 scale[LIMBS];
    load4(UNR_MONT_2_64, scale);
    for (int i = PA_SLOTS; i >= 0; i--) {
        u64 scaled[LIMBS];
        fr_mul(acc, scale, scaled);
        u64 addend[LIMBS] = {limbs[i], 0, 0, 0};
        fr_add(scaled, addend, acc);
    }
    u64 raw_one[LIMBS] = {1, 0, 0, 0};
    u64 reduced[LIMBS];
    fr_mul(acc, raw_one, reduced);
    store4(out, reduced);
}
