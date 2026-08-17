#define POINT_BLOCK 128

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

__device__ __forceinline__ int fq_geq_modulus(const u64 *a) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != FQ_MODULUS[i]) return a[i] > FQ_MODULUS[i];
    }
    return 1;
}

__device__ __forceinline__ void fq_sub_modulus(u64 *a) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) a[i] = sbb(a[i], FQ_MODULUS[i], &borrow);
}

__device__ __forceinline__ int fq_is_zero(const u64 *a) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

__device__ __forceinline__ void fq_copy(const u64 *a, u64 *out) {
    for (int i = 0; i < 4; i++) out[i] = a[i];
}

__device__ void fq_add(const u64 *a, const u64 *b, u64 *out) {
    u64 carry = 0;
    for (int i = 0; i < 4; i++) out[i] = adc(a[i], b[i], &carry);
    if (carry != 0 || fq_geq_modulus(out)) fq_sub_modulus(out);
}

__device__ void fq_sub(const u64 *a, const u64 *b, u64 *out) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) out[i] = sbb(a[i], b[i], &borrow);
    if (borrow != 0) {
        u64 carry = 0;
        for (int i = 0; i < 4; i++) out[i] = adc(out[i], FQ_MODULUS[i], &carry);
    }
}

__device__ void fq_mul(const u64 *a, const u64 *b, u64 *out) {
    u64 t[6];
    for (int i = 0; i < 6; i++) t[i] = 0;

    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) t[j] = mac(t[j], a[j], b[i], &carry);
        u64 c = 0;
        t[4] = adc(t[4], carry, &c);
        t[5] = adc(t[5], 0, &c);

        u64 m = t[0] * FQ_INV;
        u64 c2 = 0;
        mac(t[0], m, FQ_MODULUS[0], &c2);
        for (int j = 1; j < 4; j++) t[j - 1] = mac(t[j], m, FQ_MODULUS[j], &c2);
        u64 c3 = 0;
        t[3] = adc(t[4], c2, &c3);
        t[4] = adc(t[5], 0, &c3);
        t[5] = 0;
    }

    for (int i = 0; i < 4; i++) out[i] = t[i];
    if (t[4] != 0 || fq_geq_modulus(out)) fq_sub_modulus(out);
}

__device__ __forceinline__ void fq_sqr(const u64 *a, u64 *out) {
    fq_mul(a, a, out);
}

__device__ __forceinline__ void fq_neg(const u64 *a, u64 *out) {
    if (fq_is_zero(a)) {
        for (int i = 0; i < 4; i++) out[i] = 0;
        return;
    }
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) out[i] = sbb(FQ_MODULUS[i], a[i], &borrow);
}

__device__ __forceinline__ void fq_double(const u64 *a, u64 *out) {
    fq_add(a, a, out);
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

__device__ __forceinline__ int jac_is_zero(const u64 *p) {
    return fq_is_zero(p + 2 * LIMBS);
}

__device__ __forceinline__ void jac_set_zero(u64 *p) {
    for (int i = 0; i < 3 * LIMBS; i++) p[i] = 0;
}

__device__ __forceinline__ void jac_copy(const u64 *p, u64 *out) {
    for (int i = 0; i < 3 * LIMBS; i++) out[i] = p[i];
}

__device__ __noinline__ void jac_double(const u64 *p, u64 *out) {
    if (jac_is_zero(p)) {
        jac_set_zero(out);
        return;
    }
    const u64 *x = p;
    const u64 *y = p + LIMBS;
    const u64 *z = p + 2 * LIMBS;
    u64 a[LIMBS], b[LIMBS], c[LIMBS], d[LIMBS], e[LIMBS], f[LIMBS], t0[LIMBS], t1[LIMBS];
    fq_sqr(x, a);
    fq_sqr(y, b);
    fq_sqr(b, c);
    fq_add(x, b, t0);
    fq_sqr(t0, t1);
    fq_sub(t1, a, t0);
    fq_sub(t0, c, t1);
    fq_double(t1, d);
    fq_add(a, a, t0);
    fq_add(t0, a, e);
    fq_sqr(e, f);
    fq_double(d, t0);
    fq_sub(f, t0, out);
    fq_sub(d, out, t0);
    fq_mul(e, t0, t1);
    fq_double(c, t0);
    fq_double(t0, c);
    fq_double(c, t0);
    fq_sub(t1, t0, out + LIMBS);
    fq_mul(y, z, t0);
    fq_double(t0, out + 2 * LIMBS);
}

__device__ __noinline__ void jac_add(const u64 *p, const u64 *q, u64 *out) {
    if (jac_is_zero(p)) {
        jac_copy(q, out);
        return;
    }
    if (jac_is_zero(q)) {
        jac_copy(p, out);
        return;
    }
    u64 z1z1[LIMBS], z2z2[LIMBS], u1[LIMBS], u2[LIMBS], s1[LIMBS], s2[LIMBS];
    u64 h[LIMBS], i[LIMBS], j[LIMBS], r[LIMBS], v[LIMBS], t0[LIMBS], t1[LIMBS];
    fq_sqr(p + 2 * LIMBS, z1z1);
    fq_sqr(q + 2 * LIMBS, z2z2);
    fq_mul(p, z2z2, u1);
    fq_mul(q, z1z1, u2);
    fq_mul(p + LIMBS, z2z2, t0);
    fq_mul(t0, q + 2 * LIMBS, s1);
    fq_mul(q + LIMBS, z1z1, t0);
    fq_mul(t0, p + 2 * LIMBS, s2);
    fq_sub(u2, u1, h);
    fq_sub(s2, s1, t0);
    if (fq_is_zero(h) && fq_is_zero(t0)) {
        jac_double(p, out);
        return;
    }
    if (fq_is_zero(h)) {
        jac_set_zero(out);
        return;
    }
    fq_double(t0, r);
    fq_double(h, t0);
    fq_sqr(t0, i);
    fq_mul(h, i, j);
    fq_mul(u1, i, v);
    fq_sqr(r, t0);
    fq_sub(t0, j, t1);
    fq_double(v, t0);
    fq_sub(t1, t0, out);
    fq_sub(v, out, t0);
    fq_mul(r, t0, t1);
    fq_mul(s1, j, t0);
    fq_double(t0, j);
    fq_sub(t1, j, out + LIMBS);
    fq_add(p + 2 * LIMBS, q + 2 * LIMBS, t0);
    fq_sqr(t0, t1);
    fq_sub(t1, z1z1, t0);
    fq_sub(t0, z2z2, t1);
    fq_mul(t1, h, out + 2 * LIMBS);
}

__device__ __noinline__ void jac_add_affine(const u64 *p, const u64 *ax, const u64 *ay, u64 *out) {
    if (fq_is_zero(ax) && fq_is_zero(ay)) {
        jac_copy(p, out);
        return;
    }
    if (jac_is_zero(p)) {
        fq_copy(ax, out);
        fq_copy(ay, out + LIMBS);
        fq_copy(FQ_MONT_ONE, out + 2 * LIMBS);
        return;
    }
    u64 z1z1[LIMBS], u2[LIMBS], s2[LIMBS], h[LIMBS], hh[LIMBS];
    u64 i[LIMBS], j[LIMBS], r[LIMBS], v[LIMBS], t0[LIMBS], t1[LIMBS];
    fq_sqr(p + 2 * LIMBS, z1z1);
    fq_mul(ax, z1z1, u2);
    fq_mul(ay, z1z1, t0);
    fq_mul(t0, p + 2 * LIMBS, s2);
    fq_sub(u2, p, h);
    fq_sub(s2, p + LIMBS, t0);
    if (fq_is_zero(h) && fq_is_zero(t0)) {
        jac_double(p, out);
        return;
    }
    if (fq_is_zero(h)) {
        jac_set_zero(out);
        return;
    }
    fq_double(t0, r);
    fq_sqr(h, hh);
    fq_double(hh, t0);
    fq_double(t0, i);
    fq_mul(h, i, j);
    fq_mul(p, i, v);
    fq_sqr(r, t0);
    fq_sub(t0, j, t1);
    fq_double(v, t0);
    fq_sub(t1, t0, out);
    fq_sub(v, out, t0);
    fq_mul(r, t0, t1);
    fq_mul(p + LIMBS, j, t0);
    fq_double(t0, j);
    fq_sub(t1, j, out + LIMBS);
    fq_add(p + 2 * LIMBS, h, t0);
    fq_sqr(t0, t1);
    fq_sub(t1, z1z1, t0);
    fq_sub(t0, hh, out + 2 * LIMBS);
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

extern "C" __global__ void msm_g1_double_kernel(const u64 *__restrict__ points,
                                                u64 *__restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 p[3 * LIMBS], q[3 * LIMBS];
    for (int limb = 0; limb < 3; limb++) {
        load4(points + ((unsigned long long)i * 3 + limb) * LIMBS, p + limb * LIMBS);
    }
    jac_double(p, q);
    for (int limb = 0; limb < 3; limb++) {
        store4(out + ((unsigned long long)i * 3 + limb) * LIMBS, q + limb * LIMBS);
    }
}

extern "C" __global__ void msm_g1_add_kernel(const u64 *__restrict__ left,
                                             const u64 *__restrict__ right, u64 *__restrict__ out,
                                             unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 a[3 * LIMBS], b[3 * LIMBS], c[3 * LIMBS];
    for (int limb = 0; limb < 3; limb++) {
        load4(left + ((unsigned long long)i * 3 + limb) * LIMBS, a + limb * LIMBS);
        load4(right + ((unsigned long long)i * 3 + limb) * LIMBS, b + limb * LIMBS);
    }
    jac_add(a, b, c);
    for (int limb = 0; limb < 3; limb++) {
        store4(out + ((unsigned long long)i * 3 + limb) * LIMBS, c + limb * LIMBS);
    }
}

extern "C" __global__ void msm_g1_add_affine_kernel(const u64 *__restrict__ left,
                                                    const u64 *__restrict__ right,
                                                    u64 *__restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 a[3 * LIMBS], c[3 * LIMBS], bx[LIMBS], by[LIMBS];
    for (int limb = 0; limb < 3; limb++) {
        load4(left + ((unsigned long long)i * 3 + limb) * LIMBS, a + limb * LIMBS);
    }
    load4(right + ((unsigned long long)i * 2) * LIMBS, bx);
    load4(right + ((unsigned long long)i * 2 + 1) * LIMBS, by);
    jac_add_affine(a, bx, by, c);
    for (int limb = 0; limb < 3; limb++) {
        store4(out + ((unsigned long long)i * 3 + limb) * LIMBS, c + limb * LIMBS);
    }
}

extern "C" __global__ void msm_affine_denominators_kernel(const u64 *__restrict__ left,
                                                          const u64 *__restrict__ right,
                                                          u64 *__restrict__ out,
                                                          unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 x1[LIMBS], x2[LIMBS], d[LIMBS];
    load4(left + ((unsigned long long)i * 2) * LIMBS, x1);
    load4(right + ((unsigned long long)i * 2) * LIMBS, x2);
    fq_sub(x2, x1, d);
    store4(out + (unsigned long long)i * LIMBS, d);
}

extern "C" __global__ void msm_affine_combine_kernel(const u64 *__restrict__ left,
                                                     const u64 *__restrict__ right,
                                                     const u64 *__restrict__ inverses,
                                                     u64 *__restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 x1[LIMBS], y1[LIMBS], x2[LIMBS], y2[LIMBS], inv[LIMBS];
    u64 lambda[LIMBS], x3[LIMBS], y3[LIMBS], t0[LIMBS], t1[LIMBS];
    load4(left + ((unsigned long long)i * 2) * LIMBS, x1);
    load4(left + ((unsigned long long)i * 2 + 1) * LIMBS, y1);
    load4(right + ((unsigned long long)i * 2) * LIMBS, x2);
    load4(right + ((unsigned long long)i * 2 + 1) * LIMBS, y2);
    load4(inverses + (unsigned long long)i * LIMBS, inv);
    fq_sub(y2, y1, t0);
    fq_mul(t0, inv, lambda);
    fq_sqr(lambda, t0);
    fq_sub(t0, x1, t1);
    fq_sub(t1, x2, x3);
    fq_sub(x1, x3, t0);
    fq_mul(lambda, t0, t1);
    fq_sub(t1, y1, y3);
    store4(out + ((unsigned long long)i * 2) * LIMBS, x3);
    store4(out + ((unsigned long long)i * 2 + 1) * LIMBS, y3);
}

extern "C" __global__ void msm_from_montgomery_kernel(const u64 *__restrict__ values,
                                                      u64 *__restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 value[LIMBS], one[LIMBS], result[LIMBS];
    load4(values + (unsigned long long)i * LIMBS, value);
    one[0] = 1;
    one[1] = 0;
    one[2] = 0;
    one[3] = 0;
    fr_mul(value, one, result);
    store4(out + (unsigned long long)i * LIMBS, result);
}

__device__ __forceinline__ unsigned int msm_digit(const u64 *scalar, unsigned int shift,
                                                  unsigned int mask) {
    unsigned int limb = shift >> 6;
    unsigned int bit = shift & 63u;
    if (limb >= LIMBS) return 0;
    u64 value = scalar[limb] >> bit;
    if (bit != 0 && limb + 1 < LIMBS) value |= scalar[limb + 1] << (64u - bit);
    return (unsigned int)(value & (u64)mask);
}

extern "C" __global__ void msm_digits_kernel(const u64 *__restrict__ scalars, unsigned int count,
                                             unsigned int shift, unsigned int mask,
                                             unsigned int *__restrict__ digits) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 scalar[LIMBS];
    load4(scalars + (unsigned long long)i * LIMBS, scalar);
    digits[i] = msm_digit(scalar, shift, mask);
}

extern "C" __global__ void msm_bucket_count_kernel(const unsigned int *__restrict__ digits,
                                                   unsigned int count, unsigned int row_len,
                                                   unsigned int buckets,
                                                   unsigned int *__restrict__ counts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int digit = digits[i];
    if (digit == 0) return;
    unsigned int row = i / row_len;
    atomicAdd(&counts[row * buckets + digit], 1u);
}

extern "C" __global__ void msm_bucket_scatter_kernel(const unsigned int *__restrict__ digits,
                                                     const unsigned char *__restrict__ signs,
                                                     unsigned int count, unsigned int row_len,
                                                     unsigned int buckets,
                                                     unsigned int *__restrict__ cursor,
                                                     unsigned int *__restrict__ indices) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int digit = digits[i];
    if (digit == 0) return;
    unsigned int row = i / row_len;
    unsigned int column = i % row_len;
    unsigned int position = atomicAdd(&cursor[row * buckets + digit], 1u);
    indices[position] = column | ((unsigned int)signs[i] << 31);
}

extern "C" __global__ void msm_segment_sum_kernel(const u64 *__restrict__ bases,
                                                  const unsigned int *__restrict__ indices,
                                                  const unsigned int *__restrict__ offsets,
                                                  const unsigned int *__restrict__ counts,
                                                  unsigned int segments, u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int segment = blockIdx.x;
    if (segment >= segments) return;
    unsigned int start = offsets[segment];
    unsigned int end = start + counts[segment];

    u64 acc[3 * LIMBS], tmp[3 * LIMBS];
    jac_set_zero(acc);
    for (unsigned int i = start + threadIdx.x; i < end; i += blockDim.x) {
        unsigned int index = indices[i];
        unsigned int negate = index >> 31;
        index &= 0x7fffffffu;
        u64 bx[LIMBS], by[LIMBS];
        load4(bases + (unsigned long long)index * 2 * LIMBS, bx);
        load4(bases + ((unsigned long long)index * 2 + 1) * LIMBS, by);
        if (negate != 0) {
            u64 negated[LIMBS];
            fq_neg(by, negated);
            jac_add_affine(acc, bx, negated, tmp);
        } else {
            jac_add_affine(acc, bx, by, tmp);
        }
        jac_copy(tmp, acc);
    }

    u64 *slot = scratch + (unsigned long long)threadIdx.x * 3 * LIMBS;
    jac_copy(acc, slot);
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            jac_add(slot, scratch + (unsigned long long)(threadIdx.x + stride) * 3 * LIMBS, tmp);
            jac_copy(tmp, slot);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int limb = 0; limb < 3; limb++) {
            store4(out + ((unsigned long long)segment * 3 + limb) * LIMBS,
                   scratch + limb * LIMBS);
        }
    }
}

extern "C" __global__ void msm_window_accumulate_kernel(u64 *__restrict__ accumulator,
                                                        const u64 *__restrict__ window,
                                                        unsigned int rows,
                                                        unsigned int doublings) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    u64 acc[3 * LIMBS], addend[3 * LIMBS], tmp[3 * LIMBS];
    for (int limb = 0; limb < 3; limb++) {
        load4(accumulator + ((unsigned long long)row * 3 + limb) * LIMBS, acc + limb * LIMBS);
        load4(window + ((unsigned long long)row * 3 + limb) * LIMBS, addend + limb * LIMBS);
    }
    for (unsigned int i = 0; i < doublings; i++) {
        jac_double(acc, tmp);
        jac_copy(tmp, acc);
    }
    jac_add(acc, addend, tmp);
    for (int limb = 0; limb < 3; limb++) {
        store4(accumulator + ((unsigned long long)row * 3 + limb) * LIMBS, tmp + limb * LIMBS);
    }
}

extern "C" __global__ void msm_block_embed_kernel(const u64 *__restrict__ src,
                                                  unsigned int sigma_block,
                                                  unsigned int sigma_main, unsigned int count,
                                                  u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int row = i >> sigma_block;
    unsigned int column = i & ((1u << sigma_block) - 1u);
    u64 value[LIMBS];
    load4(src + (unsigned long long)i * LIMBS, value);
    store4(out + (((unsigned long long)row << sigma_main) | column) * LIMBS, value);
}

extern "C" __global__ void msm_scatter_strided_kernel(const u64 *__restrict__ src,
                                                      unsigned int stride, unsigned int count,
                                                      u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 value[LIMBS];
    load4(src + (unsigned long long)i * LIMBS, value);
    store4(out + (unsigned long long)i * stride * LIMBS, value);
}

extern "C" __global__ void msm_scatter_one_hot_kernel(const u64 *__restrict__ src,
                                                      unsigned int cycles,
                                                      unsigned int cycle_stride,
                                                      unsigned int one_hot_stride,
                                                      unsigned int count, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int address = i / cycles;
    unsigned int cycle = i % cycles;
    u64 value[LIMBS];
    load4(src + (unsigned long long)i * LIMBS, value);
    store4(out + ((unsigned long long)cycle * cycle_stride + (unsigned long long)address *
                                                                one_hot_stride) *
                     LIMBS,
           value);
}

extern "C" __global__ void msm_fold_rows_kernel(const u64 *__restrict__ table,
                                                const u64 *__restrict__ left, unsigned int rows,
                                                unsigned int columns, u64 *__restrict__ out) {
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    u64 acc[LIMBS] = {0, 0, 0, 0};
    for (unsigned int row = 0; row < rows; row++) {
        u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
        load4(left + (unsigned long long)row * LIMBS, weight);
        load4(table + ((unsigned long long)row * columns + column) * LIMBS, value);
        fr_mul(weight, value, term);
        fr_add(acc, term, sum);
        for (int limb = 0; limb < LIMBS; limb++) acc[limb] = sum[limb];
    }
    store4(out + (unsigned long long)column * LIMBS, acc);
}

extern "C" __global__ void msm_segment_sum_small_kernel(const u64 *__restrict__ bases,
                                                        const unsigned int *__restrict__ indices,
                                                        const unsigned int *__restrict__ offsets,
                                                        const unsigned int *__restrict__ counts,
                                                        unsigned int segments,
                                                        u64 *__restrict__ out) {
    unsigned int segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    unsigned int start = offsets[segment];
    unsigned int end = start + counts[segment];

    u64 acc[3 * LIMBS], tmp[3 * LIMBS];
    jac_set_zero(acc);
    for (unsigned int i = start; i < end; i++) {
        unsigned int index = indices[i];
        unsigned int negate = index >> 31;
        index &= 0x7fffffffu;
        u64 bx[LIMBS], by[LIMBS];
        load4(bases + (unsigned long long)index * 2 * LIMBS, bx);
        load4(bases + ((unsigned long long)index * 2 + 1) * LIMBS, by);
        if (negate != 0) {
            u64 negated[LIMBS];
            fq_neg(by, negated);
            jac_add_affine(acc, bx, negated, tmp);
        } else {
            jac_add_affine(acc, bx, by, tmp);
        }
        jac_copy(tmp, acc);
    }
    for (int limb = 0; limb < 3; limb++) {
        store4(out + ((unsigned long long)segment * 3 + limb) * LIMBS, acc + limb * LIMBS);
    }
}

extern "C" __global__ void msm_bucket_reduce_parallel_kernel(
    const u64 *__restrict__ buckets_points, unsigned int rows, unsigned int buckets,
    u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int groups = blockDim.x;
    unsigned int span = (buckets - 1 + groups - 1) / groups;
    unsigned int lo = 1 + threadIdx.x * span;
    unsigned int hi = lo + span;
    if (hi > buckets) hi = buckets;

    u64 acc[3 * LIMBS], running[3 * LIMBS], tmp[3 * LIMBS], bucket[3 * LIMBS];
    jac_set_zero(acc);
    jac_set_zero(running);
    if (lo < hi) {
        for (int b = (int)hi - 1; b >= (int)lo; b--) {
            unsigned long long base = ((unsigned long long)row * buckets + (unsigned int)b) * 3;
            for (int limb = 0; limb < 3; limb++) {
                load4(buckets_points + (base + limb) * LIMBS, bucket + limb * LIMBS);
            }
            jac_add(acc, bucket, tmp);
            jac_copy(tmp, acc);
            jac_add(running, acc, tmp);
            jac_copy(tmp, running);
        }
        unsigned int weight = lo - 1;
        if (weight != 0) {
            u64 scaled[3 * LIMBS], addend[3 * LIMBS];
            jac_set_zero(scaled);
            jac_copy(acc, addend);
            while (weight != 0) {
                if ((weight & 1u) != 0) {
                    jac_add(scaled, addend, tmp);
                    jac_copy(tmp, scaled);
                }
                jac_double(addend, tmp);
                jac_copy(tmp, addend);
                weight >>= 1;
            }
            jac_add(running, scaled, tmp);
            jac_copy(tmp, running);
        }
    }

    u64 *slot = scratch + (unsigned long long)threadIdx.x * 3 * LIMBS;
    jac_copy(running, slot);
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            jac_add(slot, scratch + (unsigned long long)(threadIdx.x + stride) * 3 * LIMBS, tmp);
            jac_copy(tmp, slot);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int limb = 0; limb < 3; limb++) {
            store4(out + ((unsigned long long)row * 3 + limb) * LIMBS, scratch + limb * LIMBS);
        }
    }
}
