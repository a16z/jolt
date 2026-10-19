__device__ __constant__ u64 FQ_MODULUS[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
__device__ __constant__ u64 FQ_INV = 0x87d20782e4866389ULL;
__device__ __constant__ u64 FQ_EXP[4] = {
    0x3c208c16d87cfd45ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
__device__ __constant__ u64 FQ_MONT_ONE[4] = {
    0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
};


__device__ __forceinline__ int fq_is_zero(const u64 *a) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

__device__ __forceinline__ void fq_copy(const u64 *a, u64 *out) {
    for (int i = 0; i < 4; i++) out[i] = a[i];
}

__device__ void fq_add(const u64 *a, const u64 *b, u64 *out) {
    field_add(a, b, FQ_MODULUS, out);
}

__device__ void fq_sub(const u64 *a, const u64 *b, u64 *out) {
    field_sub(a, b, FQ_MODULUS, out);
}

__device__ void fq_mul(const u64 *a, const u64 *b, u64 *out) {
    field_mont_mul(a, b, FQ_MODULUS, FQ_INV, out);
}

__device__ __forceinline__ void fq_sqr(const u64 *a, u64 *out) {
    fq_mul(a, a, out);
}

__device__ __noinline__ void fq_inverse(const u64 *a, u64 *out) {
    u64 acc[LIMBS], square[LIMBS], scratch[LIMBS];
    fq_copy(FQ_MONT_ONE, acc);
    fq_copy(a, square);
    if (fq_is_zero(a)) {
        for (int i = 0; i < LIMBS; i++) out[i] = 0;
        return;
    }
#pragma unroll 1
    for (int limb = 0; limb < 4; limb++) {
        u64 bits = FQ_EXP[limb];
#pragma unroll 1
        for (int bit = 0; bit < 64; bit++) {
            if ((bits >> bit) & 1ULL) {
                fq_mul(acc, square, scratch);
                fq_copy(scratch, acc);
            }
            fq_sqr(square, scratch);
            fq_copy(scratch, square);
        }
    }
    fq_copy(acc, out);
}

extern "C" __global__ void msm_fq_add_kernel(const u64 *__restrict__ left,
                                             const u64 *__restrict__ right, u64 *__restrict__ out,
                                             unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 a[LIMBS], b[LIMBS], c[LIMBS];
    load4(left + (unsigned long long)i * LIMBS, a);
    load4(right + (unsigned long long)i * LIMBS, b);
    fq_add(a, b, c);
    store4(out + (unsigned long long)i * LIMBS, c);
}

extern "C" __global__ void msm_fq_sub_kernel(const u64 *__restrict__ left,
                                             const u64 *__restrict__ right, u64 *__restrict__ out,
                                             unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 a[LIMBS], b[LIMBS], c[LIMBS];
    load4(left + (unsigned long long)i * LIMBS, a);
    load4(right + (unsigned long long)i * LIMBS, b);
    fq_sub(a, b, c);
    store4(out + (unsigned long long)i * LIMBS, c);
}

extern "C" __global__ void msm_fq_mul_kernel(const u64 *__restrict__ left,
                                             const u64 *__restrict__ right, u64 *__restrict__ out,
                                             unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 a[LIMBS], b[LIMBS], c[LIMBS];
    load4(left + (unsigned long long)i * LIMBS, a);
    load4(right + (unsigned long long)i * LIMBS, b);
    fq_mul(a, b, c);
    store4(out + (unsigned long long)i * LIMBS, c);
}

extern "C" __global__ void msm_fq_batch_inverse_kernel(const u64 *__restrict__ values,
                                                       u64 *__restrict__ out, unsigned int count,
                                                       unsigned int chunk) {
    unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long start = (unsigned long long)thread * chunk;
    if (start >= count) return;
    unsigned int len = chunk;
    if (start + len > count) len = (unsigned int)(count - start);

    u64 running[LIMBS], scratch[LIMBS], value[LIMBS];
    fq_copy(FQ_MONT_ONE, running);
    for (unsigned int i = 0; i < len; i++) {
        load4(values + (start + i) * LIMBS, value);
        store4(out + (start + i) * LIMBS, running);
        if (!fq_is_zero(value)) {
            fq_mul(running, value, scratch);
            fq_copy(scratch, running);
        }
    }

    u64 inverse[LIMBS];
    fq_inverse(running, inverse);

    for (int i = (int)len - 1; i >= 0; i--) {
        load4(values + (start + (unsigned int)i) * LIMBS, value);
        if (fq_is_zero(value)) {
            for (int limb = 0; limb < LIMBS; limb++) scratch[limb] = 0;
            store4(out + (start + (unsigned int)i) * LIMBS, scratch);
            continue;
        }
        load4(out + (start + (unsigned int)i) * LIMBS, scratch);
        u64 result[LIMBS];
        fq_mul(scratch, inverse, result);
        store4(out + (start + (unsigned int)i) * LIMBS, result);
        fq_mul(inverse, value, scratch);
        fq_copy(scratch, inverse);
    }
}
