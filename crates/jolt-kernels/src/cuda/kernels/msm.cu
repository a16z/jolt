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
