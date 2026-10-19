#define MSM_COLD 0xFFFFFFFFu

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

__device__ __forceinline__ void glv_mul_acc(const u64 *a, int alen, const u64 *b, int blen,
                                           u64 *acc, int acclen) {
    for (int j = 0; j < blen; j++) {
        u64 carry = 0;
        for (int l = 0; l < alen; l++) {
            acc[j + l] = mac(acc[j + l], a[l], b[j], &carry);
        }
        int index = j + alen;
        while (carry != 0 && index < acclen) {
            u64 next = 0;
            acc[index] = adc(acc[index], carry, &next);
            carry = next;
            index++;
        }
    }
}

__device__ __forceinline__ int glv_sub_abs(const u64 *x, const u64 *y, int len, u64 *out) {
    int order = 0;
    for (int i = len - 1; i >= 0; i--) {
        if (x[i] != y[i]) {
            order = (x[i] > y[i]) ? 1 : -1;
            break;
        }
    }
    u64 borrow = 0;
    if (order >= 0) {
        for (int i = 0; i < len; i++) out[i] = sbb(x[i], y[i], &borrow);
        return 0;
    }
    for (int i = 0; i < len; i++) out[i] = sbb(y[i], x[i], &borrow);
    return 1;
}

extern "C" __global__ void msm_glv_decompose_2d_kernel(const u64 *__restrict__ scalars,
                                                     const u64 *__restrict__ constants,
                                                     unsigned int count,
                                                     u64 *__restrict__ out_scalars,
                                                     unsigned char *__restrict__ out_signs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const u64 *recip_c = constants;
    const u64 *recip_b = constants + 5;
    const u64 *coeff_a = constants + 9;
    const u64 *coeff_b = constants + 11;
    const u64 *coeff_c = constants + 12;

    u64 k[LIMBS];
    load4(scalars + (unsigned long long)i * LIMBS, k);

    u64 wide[9];
    for (int j = 0; j < 9; j++) wide[j] = 0;
    glv_mul_acc(k, LIMBS, recip_c, 5, wide, 9);
    u64 p[3] = {wide[6], wide[7], wide[8]};

    for (int j = 0; j < 9; j++) wide[j] = 0;
    glv_mul_acc(k, LIMBS, recip_b, 4, wide, 9);
    u64 q[3] = {wide[6], wide[7], wide[8]};

    u64 left[6], right[6];
    for (int j = 0; j < 6; j++) {
        left[j] = 0;
        right[j] = 0;
    }
    glv_mul_acc(p, 3, coeff_a, 2, left, 6);
    glv_mul_acc(q, 3, coeff_b, 1, left, 6);
    u64 padded[6] = {k[0], k[1], k[2], k[3], 0, 0};
    u64 first[6];
    int first_negative = glv_sub_abs(padded, left, 6, first);

    for (int j = 0; j < 6; j++) {
        right[j] = 0;
        left[j] = 0;
    }
    glv_mul_acc(p, 3, coeff_b, 1, left, 6);
    glv_mul_acc(q, 3, coeff_c, 2, right, 6);
    u64 second[6];
    int second_negative = glv_sub_abs(left, right, 6, second);

    unsigned long long mapped = (unsigned long long)count + i;
    store4(out_scalars + (unsigned long long)i * LIMBS, first);
    store4(out_scalars + mapped * LIMBS, second);
    out_signs[i] = (unsigned char)first_negative;
    out_signs[mapped] = (unsigned char)second_negative;
}

extern "C" __global__ void msm_g1_endomorphism_kernel(const u64 *__restrict__ jacobian,
                                                     const u64 *__restrict__ beta,
                                                     unsigned int count, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 x[LIMBS], y[LIMBS], z[LIMBS], scaled[LIMBS], factor[LIMBS];
    load4(jacobian + ((unsigned long long)i * 3) * LIMBS, x);
    load4(jacobian + ((unsigned long long)i * 3 + 1) * LIMBS, y);
    load4(jacobian + ((unsigned long long)i * 3 + 2) * LIMBS, z);
    load4(beta, factor);

    store4(out + ((unsigned long long)i * 3) * LIMBS, x);
    store4(out + ((unsigned long long)i * 3 + 1) * LIMBS, y);
    store4(out + ((unsigned long long)i * 3 + 2) * LIMBS, z);

    unsigned long long mapped = (unsigned long long)count + i;
    fq_mul(x, factor, scaled);
    store4(out + (mapped * 3) * LIMBS, scaled);
    store4(out + (mapped * 3 + 1) * LIMBS, y);
    store4(out + (mapped * 3 + 2) * LIMBS, z);
}

extern "C" __global__ void msm_jacobian_z_kernel(const u64 *__restrict__ jacobian,
                                                unsigned int count, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 z[LIMBS];
    load4(jacobian + ((unsigned long long)i * 3 + 2) * LIMBS, z);
    if (fq_is_zero(z)) {
        u64 placeholder[LIMBS];
        placeholder[0] = 1ull;
        for (int limb = 1; limb < LIMBS; limb++) placeholder[limb] = 0ull;
        store4(out + (unsigned long long)i * LIMBS, placeholder);
        return;
    }
    store4(out + (unsigned long long)i * LIMBS, z);
}

extern "C" __global__ void msm_jacobian_to_affine_kernel(const u64 *__restrict__ jacobian,
                                                         const u64 *__restrict__ z_inverses,
                                                         unsigned int count,
                                                         u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 x[LIMBS], y[LIMBS], z[LIMBS];
    load4(jacobian + ((unsigned long long)i * 3) * LIMBS, x);
    load4(jacobian + ((unsigned long long)i * 3 + 1) * LIMBS, y);
    load4(jacobian + ((unsigned long long)i * 3 + 2) * LIMBS, z);

    u64 zero[LIMBS];
    for (int limb = 0; limb < LIMBS; limb++) zero[limb] = 0ull;
    if (fq_is_zero(z)) {
        store4(out + ((unsigned long long)i * 2) * LIMBS, zero);
        store4(out + ((unsigned long long)i * 2 + 1) * LIMBS, zero);
        return;
    }

    u64 inv[LIMBS], inv2[LIMBS], inv3[LIMBS], affine[LIMBS];
    load4(z_inverses + (unsigned long long)i * LIMBS, inv);
    fq_sqr(inv, inv2);
    fq_mul(inv2, inv, inv3);
    fq_mul(x, inv2, affine);
    store4(out + ((unsigned long long)i * 2) * LIMBS, affine);
    fq_mul(y, inv3, affine);
    store4(out + ((unsigned long long)i * 2 + 1) * LIMBS, affine);
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
                                             unsigned int limbs, unsigned int shift,
                                             unsigned int mask,
                                             unsigned int *__restrict__ digits) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    u64 scalar[LIMBS];
    const u64 *source = scalars + (unsigned long long)i * limbs;
    if (limbs >= LIMBS) {
        load4(source, scalar);
    } else {
        for (int l = 0; l < LIMBS; l++) scalar[l] = 0;
        for (unsigned int l = 0; l < limbs; l++) scalar[l] = source[l];
    }
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

extern "C" __global__ void msm_one_hot_count_kernel(const unsigned int *__restrict__ hot,
                                                   unsigned int cycles, unsigned int chunk_len,
                                                   unsigned int chunk_count,
                                                   unsigned int one_hot_k,
                                                   unsigned int *__restrict__ counts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cycles) return;
    unsigned int address = hot[i];
    if (address == MSM_COLD) return;
    if (address >= one_hot_k) {
        atomicAdd(&counts[one_hot_k * chunk_count], 1u);
        return;
    }
    atomicAdd(&counts[address * chunk_count + i / chunk_len], 1u);
}

extern "C" __global__ void msm_one_hot_scatter_kernel(const unsigned int *__restrict__ hot,
                                                     unsigned int cycles, unsigned int chunk_len,
                                                     unsigned int chunk_count,
                                                     unsigned int one_hot_k,
                                                     unsigned int *__restrict__ cursor,
                                                     unsigned int *__restrict__ indices) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cycles) return;
    unsigned int address = hot[i];
    if (address == MSM_COLD || address >= one_hot_k) return;
    unsigned int segment = address * chunk_count + i / chunk_len;
    unsigned int position = atomicAdd(&cursor[segment], 1u);
    indices[position] = i % chunk_len;
}

extern "C" __global__ void msm_one_hot_count_shared_kernel(const unsigned int *__restrict__ hot,
                                                           unsigned int cycles,
                                                           unsigned int chunk_len,
                                                           unsigned int chunk_count,
                                                           unsigned int one_hot_k,
                                                           unsigned int slots,
                                                           unsigned int *__restrict__ counts) {
    extern __shared__ unsigned int one_hot_tally[];
    unsigned int *tally = one_hot_tally;
    unsigned int origin = blockIdx.x * blockDim.x;
    unsigned int first = origin / chunk_len;
    for (unsigned int s = threadIdx.x; s < slots; s += blockDim.x) tally[s] = 0u;
    __syncthreads();

    unsigned int i = origin + threadIdx.x;
    if (i < cycles) {
        unsigned int address = hot[i];
        if (address == MSM_COLD) {
            // a cold cycle selects no address
        } else if (address >= one_hot_k) {
            atomicAdd(&counts[one_hot_k * chunk_count], 1u);
        } else {
            atomicAdd(&tally[(i / chunk_len - first) * one_hot_k + address], 1u);
        }
    }
    __syncthreads();

    for (unsigned int s = threadIdx.x; s < slots; s += blockDim.x) {
        unsigned int n = tally[s];
        unsigned int chunk = first + s / one_hot_k;
        if (n == 0u || chunk >= chunk_count) continue;
        atomicAdd(&counts[(s % one_hot_k) * chunk_count + chunk], n);
    }
}

extern "C" __global__ void msm_one_hot_scatter_shared_kernel(const unsigned int *__restrict__ hot,
                                                             unsigned int cycles,
                                                             unsigned int chunk_len,
                                                             unsigned int chunk_count,
                                                             unsigned int one_hot_k,
                                                             unsigned int slots,
                                                             unsigned int *__restrict__ cursor,
                                                             unsigned int *__restrict__ indices) {
    extern __shared__ unsigned int one_hot_slots[];
    unsigned int *tally = one_hot_slots;
    unsigned int *reserved = one_hot_slots + slots;
    unsigned int origin = blockIdx.x * blockDim.x;
    unsigned int first = origin / chunk_len;
    for (unsigned int s = threadIdx.x; s < slots; s += blockDim.x) tally[s] = 0u;
    __syncthreads();

    unsigned int i = origin + threadIdx.x;
    unsigned int slot = slots;
    if (i < cycles) {
        unsigned int address = hot[i];
        if (address != MSM_COLD && address < one_hot_k) {
            slot = (i / chunk_len - first) * one_hot_k + address;
            atomicAdd(&tally[slot], 1u);
        }
    }
    __syncthreads();

    for (unsigned int s = threadIdx.x; s < slots; s += blockDim.x) {
        unsigned int n = tally[s];
        unsigned int chunk = first + s / one_hot_k;
        reserved[s] = (n != 0u && chunk < chunk_count)
                          ? atomicAdd(&cursor[(s % one_hot_k) * chunk_count + chunk], n)
                          : 0u;
        tally[s] = 0u;
    }
    __syncthreads();

    if (slot < slots) {
        indices[reserved[slot] + atomicAdd(&tally[slot], 1u)] = i % chunk_len;
    }
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

#define MSM_SEGMENT_WARP 32u

extern "C" __global__ void msm_segment_sum_warp_kernel(const u64 *__restrict__ bases,
                                                       const unsigned int *__restrict__ indices,
                                                       const unsigned int *__restrict__ offsets,
                                                       const unsigned int *__restrict__ counts,
                                                       unsigned int segments,
                                                       u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int lane = threadIdx.x & (MSM_SEGMENT_WARP - 1u);
    unsigned int segment =
        blockIdx.x * (blockDim.x / MSM_SEGMENT_WARP) + threadIdx.x / MSM_SEGMENT_WARP;
    if (segment >= segments) return;
    unsigned int start = offsets[segment];
    unsigned int end = start + counts[segment];

    u64 acc[3 * LIMBS], tmp[3 * LIMBS];
    jac_set_zero(acc);
    for (unsigned int i = start + lane; i < end; i += MSM_SEGMENT_WARP) {
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
    __syncwarp();
    for (unsigned int stride = MSM_SEGMENT_WARP >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            jac_add(slot, scratch + (unsigned long long)(threadIdx.x + stride) * 3 * LIMBS, tmp);
            jac_copy(tmp, slot);
        }
        __syncwarp();
    }
    if (lane == 0) {
        for (int limb = 0; limb < 3; limb++) {
            store4(out + ((unsigned long long)segment * 3 + limb) * LIMBS, slot + limb * LIMBS);
        }
    }
}

__device__ __forceinline__ void msm_segment_accumulate(const u64 *__restrict__ bases,
                                                       const unsigned int *__restrict__ indices,
                                                       unsigned int start, unsigned int end,
                                                       unsigned int first, unsigned int stride,
                                                       u64 *acc) {
    u64 tmp[3 * LIMBS];
    jac_set_zero(acc);
    for (unsigned int i = start + first; i < end; i += stride) {
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
}

extern "C" __global__ void msm_segment_sum_classed_kernel(
    const u64 *__restrict__ bases, const unsigned int *__restrict__ indices,
    const unsigned int *__restrict__ offsets, const unsigned int *__restrict__ counts,
    const unsigned int *__restrict__ ids, unsigned int fat, unsigned int thin,
    u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    u64 acc[3 * LIMBS], tmp[3 * LIMBS];
    u64 *slot_points = scratch + (unsigned long long)threadIdx.x * 3 * LIMBS;

    if (blockIdx.x < fat) {
        unsigned int segment = ids[blockIdx.x];
        unsigned int start = offsets[segment];
        msm_segment_accumulate(bases, indices, start, start + counts[segment], threadIdx.x,
                               blockDim.x, acc);
        jac_copy(acc, slot_points);
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                jac_add(slot_points,
                        scratch + (unsigned long long)(threadIdx.x + stride) * 3 * LIMBS, tmp);
                jac_copy(tmp, slot_points);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            for (int limb = 0; limb < 3; limb++) {
                store4(out + ((unsigned long long)segment * 3 + limb) * LIMBS,
                       scratch + limb * LIMBS);
            }
        }
        return;
    }

    unsigned int warps = blockDim.x / MSM_SEGMENT_WARP;
    unsigned int lane = threadIdx.x & (MSM_SEGMENT_WARP - 1u);
    unsigned int slot = (blockIdx.x - fat) * warps + threadIdx.x / MSM_SEGMENT_WARP;
    if (slot >= thin) return;
    unsigned int segment = ids[fat + slot];
    unsigned int start = offsets[segment];
    msm_segment_accumulate(bases, indices, start, start + counts[segment], lane, MSM_SEGMENT_WARP,
                           acc);
    jac_copy(acc, slot_points);
    __syncwarp();
    for (unsigned int stride = MSM_SEGMENT_WARP >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            jac_add(slot_points, scratch + (unsigned long long)(threadIdx.x + stride) * 3 * LIMBS,
                    tmp);
            jac_copy(tmp, slot_points);
        }
        __syncwarp();
    }
    if (lane == 0) {
        for (int limb = 0; limb < 3; limb++) {
            store4(out + ((unsigned long long)segment * 3 + limb) * LIMBS,
                   slot_points + limb * LIMBS);
        }
    }
}

extern "C" __global__ void msm_one_hot_complement_kernel(
    const u64 *__restrict__ total, const unsigned int *__restrict__ targets,
    unsigned int target_count, unsigned int chunks, unsigned int addresses, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= target_count) return;
    unsigned int target = targets[i];
    unsigned int chunk = target % chunks;
    u64 acc[3 * LIMBS], point[3 * LIMBS], tmp[3 * LIMBS];
    jac_set_zero(acc);
    for (unsigned int address = 0; address < addresses; address++) {
        unsigned long long segment = (unsigned long long)address * chunks + chunk;
        if (segment == target) continue;
        for (int limb = 0; limb < 3; limb++) {
            load4(out + (segment * 3 + limb) * LIMBS, point + limb * LIMBS);
        }
        jac_add(acc, point, tmp);
        jac_copy(tmp, acc);
    }
    fq_neg(acc + LIMBS, tmp);
    fq_copy(tmp, acc + LIMBS);
    jac_add(total, acc, tmp);
    for (int limb = 0; limb < 3; limb++) {
        store4(out + ((unsigned long long)target * 3 + limb) * LIMBS, tmp + limb * LIMBS);
    }
}

extern "C" __global__ void msm_window_fold_kernel(const u64 *__restrict__ window_points,
                                                 unsigned int rows, unsigned int windows,
                                                 unsigned int window_bits,
                                                 u64 *__restrict__ out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    u64 acc[3 * LIMBS], addend[3 * LIMBS], tmp[3 * LIMBS];
    jac_set_zero(acc);
    for (int window = (int)windows - 1; window >= 0; window--) {
        if ((unsigned int)window + 1 != windows) {
            for (unsigned int i = 0; i < window_bits; i++) {
                jac_double(acc, tmp);
                jac_copy(tmp, acc);
            }
        }
        unsigned long long base = ((unsigned long long)(unsigned int)window * rows + row) * 3;
        for (int limb = 0; limb < 3; limb++) {
            load4(window_points + (base + limb) * LIMBS, addend + limb * LIMBS);
        }
        jac_add(acc, addend, tmp);
        jac_copy(tmp, acc);
    }
    for (int limb = 0; limb < 3; limb++) {
        store4(out + ((unsigned long long)row * 3 + limb) * LIMBS, acc + limb * LIMBS);
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

// Limit the warp fallback's register footprint to preserve small-bucket throughput.
extern "C" __global__ __launch_bounds__(256, 2)
void msm_segment_sum_small_kernel(const u64 *__restrict__ bases,
                                  const unsigned int *__restrict__ indices,
                                  const unsigned int *__restrict__ offsets,
                                  const unsigned int *__restrict__ counts,
                                  unsigned int segments, u64 *__restrict__ out) {
    constexpr unsigned int SERIAL_LIMIT = 64u;
    unsigned int segment = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadIdx.x & (MSM_SEGMENT_WARP - 1u);
    unsigned int start = segment < segments ? offsets[segment] : 0;
    unsigned int size = segment < segments ? counts[segment] : 0;
    unsigned int pending = __ballot_sync(0xffffffffu, size > SERIAL_LIMIT);

    u64 acc[3 * LIMBS], tmp[3 * LIMBS], peer[3 * LIMBS];
    if (segment < segments && size <= SERIAL_LIMIT) {
        msm_segment_accumulate(bases, indices, start, start + size, 0, 1, acc);
        for (int limb = 0; limb < 3; limb++) {
            store4(out + ((unsigned long long)segment * 3 + limb) * LIMBS, acc + limb * LIMBS);
        }
    }

    // The average bucket size can hide a few long serial chains. Share those
    // buckets across their warp without building a separate segment plan.
    while (pending != 0) {
        unsigned int leader = __ffs(pending) - 1;
        unsigned int begin = __shfl_sync(0xffffffffu, start, leader);
        unsigned int length = __shfl_sync(0xffffffffu, size, leader);
        msm_segment_accumulate(bases, indices, begin, begin + length, lane, MSM_SEGMENT_WARP, acc);
        for (unsigned int stride = MSM_SEGMENT_WARP >> 1; stride != 0; stride >>= 1) {
            for (int limb = 0; limb < 3 * LIMBS; limb++) {
                peer[limb] = __shfl_down_sync(0xffffffffu, acc[limb], stride);
            }
            if (lane < stride) {
                jac_add(acc, peer, tmp);
                jac_copy(tmp, acc);
            }
        }
        if (lane == 0) {
            unsigned int target = (segment & ~(MSM_SEGMENT_WARP - 1u)) + leader;
            for (int limb = 0; limb < 3; limb++) {
                store4(out + ((unsigned long long)target * 3 + limb) * LIMBS, acc + limb * LIMBS);
            }
        }
        pending &= pending - 1;
    }
}

extern "C" __global__ void msm_bucket_reduce_chunked_kernel(
    const u64 *__restrict__ buckets_points, unsigned int rows, unsigned int buckets,
    unsigned int chunks, u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int row = blockIdx.x / chunks;
    unsigned int chunk = blockIdx.x - row * chunks;
    if (row >= rows) return;

    unsigned int weighted = (buckets > 1) ? (buckets - 1) : 0;
    unsigned int chunk_span = (weighted + chunks - 1) / chunks;
    unsigned int chunk_lo = 1 + chunk * chunk_span;
    unsigned int chunk_hi = chunk_lo + chunk_span;
    if (chunk_hi > buckets) chunk_hi = buckets;
    unsigned int width = (chunk_hi > chunk_lo) ? (chunk_hi - chunk_lo) : 0;

    unsigned int groups = blockDim.x;
    unsigned int span = (width + groups - 1) / groups;
    unsigned int lo = chunk_lo + threadIdx.x * span;
    unsigned int hi = lo + span;
    if (hi > chunk_hi) hi = chunk_hi;

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
        unsigned long long dest = (unsigned long long)row * chunks + chunk;
        for (int limb = 0; limb < 3; limb++) {
            store4(out + (dest * 3 + limb) * LIMBS, scratch + limb * LIMBS);
        }
    }
}

extern "C" __global__ void msm_point_rows_sum_kernel(const u64 *__restrict__ partials,
                                                    unsigned int rows, unsigned int count,
                                                    u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    u64 acc[3 * LIMBS], tmp[3 * LIMBS], value[3 * LIMBS];
    jac_set_zero(acc);
    for (unsigned int index = threadIdx.x; index < count; index += blockDim.x) {
        unsigned long long base = ((unsigned long long)row * count + index) * 3;
        for (int limb = 0; limb < 3; limb++) {
            load4(partials + (base + limb) * LIMBS, value + limb * LIMBS);
        }
        jac_add(acc, value, tmp);
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
            store4(out + ((unsigned long long)row * 3 + limb) * LIMBS, scratch + limb * LIMBS);
        }
    }
}
