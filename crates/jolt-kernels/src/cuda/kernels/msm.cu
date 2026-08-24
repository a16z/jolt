#define POINT_BLOCK 128
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

#define FQ2_LIMBS (2 * LIMBS)
#define G2_LIMBS (3 * FQ2_LIMBS)

__device__ __forceinline__ int fq2_is_zero(const u64 *a) {
    return fq_is_zero(a) && fq_is_zero(a + LIMBS);
}

__device__ __forceinline__ void fq2_copy(const u64 *a, u64 *out) {
    for (int i = 0; i < FQ2_LIMBS; i++) out[i] = a[i];
}

__device__ __forceinline__ void fq2_add(const u64 *a, const u64 *b, u64 *out) {
    fq_add(a, b, out);
    fq_add(a + LIMBS, b + LIMBS, out + LIMBS);
}

__device__ __forceinline__ void fq2_sub(const u64 *a, const u64 *b, u64 *out) {
    fq_sub(a, b, out);
    fq_sub(a + LIMBS, b + LIMBS, out + LIMBS);
}

__device__ __forceinline__ void fq2_double(const u64 *a, u64 *out) {
    fq2_add(a, a, out);
}

__device__ __noinline__ void fq2_mul(const u64 *a, const u64 *b, u64 *out) {
    u64 v0[LIMBS], v1[LIMBS], s[LIMBS], t[LIMBS], c1[LIMBS];
    fq_mul(a, b, v0);
    fq_mul(a + LIMBS, b + LIMBS, v1);
    fq_add(a, a + LIMBS, s);
    fq_add(b, b + LIMBS, t);
    fq_mul(s, t, c1);
    fq_sub(c1, v0, s);
    fq_sub(s, v1, c1);
    fq_sub(v0, v1, out);
    fq_copy(c1, out + LIMBS);
}

__device__ __noinline__ void fq2_sqr(const u64 *a, u64 *out) {
    u64 s[LIMBS], d[LIMBS], m[LIMBS], c0[LIMBS];
    fq_add(a, a + LIMBS, s);
    fq_sub(a, a + LIMBS, d);
    fq_mul(a, a + LIMBS, m);
    fq_mul(s, d, c0);
    fq_double(m, out + LIMBS);
    fq_copy(c0, out);
}

__device__ __forceinline__ int jac2_is_zero(const u64 *p) {
    return fq2_is_zero(p + 2 * FQ2_LIMBS);
}

__device__ __forceinline__ void jac2_set_zero(u64 *p) {
    for (int i = 0; i < G2_LIMBS; i++) p[i] = 0;
}

__device__ __forceinline__ void jac2_copy(const u64 *p, u64 *out) {
    for (int i = 0; i < G2_LIMBS; i++) out[i] = p[i];
}

__device__ __noinline__ void jac2_double(const u64 *p, u64 *out) {
    if (jac2_is_zero(p)) {
        jac2_set_zero(out);
        return;
    }
    const u64 *x = p;
    const u64 *y = p + FQ2_LIMBS;
    const u64 *z = p + 2 * FQ2_LIMBS;
    u64 a[FQ2_LIMBS], b[FQ2_LIMBS], c[FQ2_LIMBS], d[FQ2_LIMBS];
    u64 e[FQ2_LIMBS], f[FQ2_LIMBS], t0[FQ2_LIMBS], t1[FQ2_LIMBS], z3[FQ2_LIMBS];
    fq2_sqr(x, a);
    fq2_sqr(y, b);
    fq2_sqr(b, c);
    fq2_add(x, b, t0);
    fq2_sqr(t0, t1);
    fq2_sub(t1, a, t0);
    fq2_sub(t0, c, t1);
    fq2_double(t1, d);
    fq2_add(a, a, t0);
    fq2_add(t0, a, e);
    fq2_sqr(e, f);
    fq2_mul(y, z, t0);
    fq2_double(t0, z3);
    fq2_double(d, t0);
    fq2_sub(f, t0, out);
    fq2_sub(d, out, t0);
    fq2_mul(e, t0, t1);
    fq2_double(c, t0);
    fq2_double(t0, c);
    fq2_double(c, t0);
    fq2_sub(t1, t0, out + FQ2_LIMBS);
    fq2_copy(z3, out + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void jac2_add(const u64 *p, const u64 *q, u64 *out) {
    if (jac2_is_zero(p)) {
        jac2_copy(q, out);
        return;
    }
    if (jac2_is_zero(q)) {
        jac2_copy(p, out);
        return;
    }
    u64 z1z1[FQ2_LIMBS], z2z2[FQ2_LIMBS], u1[FQ2_LIMBS], s1[FQ2_LIMBS];
    u64 h[FQ2_LIMBS], i[FQ2_LIMBS], j[FQ2_LIMBS], r[FQ2_LIMBS], v[FQ2_LIMBS];
    u64 t0[FQ2_LIMBS], t1[FQ2_LIMBS], z3[FQ2_LIMBS];
    fq2_sqr(p + 2 * FQ2_LIMBS, z1z1);
    fq2_sqr(q + 2 * FQ2_LIMBS, z2z2);
    fq2_mul(p, z2z2, u1);
    fq2_mul(q, z1z1, t1);
    fq2_mul(p + FQ2_LIMBS, z2z2, t0);
    fq2_mul(t0, q + 2 * FQ2_LIMBS, s1);
    fq2_mul(q + FQ2_LIMBS, z1z1, t0);
    fq2_mul(t0, p + 2 * FQ2_LIMBS, v);
    fq2_sub(t1, u1, h);
    fq2_sub(v, s1, t0);
    if (fq2_is_zero(h) && fq2_is_zero(t0)) {
        jac2_double(p, out);
        return;
    }
    if (fq2_is_zero(h)) {
        jac2_set_zero(out);
        return;
    }
    fq2_add(p + 2 * FQ2_LIMBS, q + 2 * FQ2_LIMBS, t1);
    fq2_sqr(t1, z3);
    fq2_sub(z3, z1z1, t1);
    fq2_sub(t1, z2z2, z3);
    fq2_mul(z3, h, t1);
    fq2_copy(t1, z3);
    fq2_double(t0, r);
    fq2_double(h, t0);
    fq2_sqr(t0, i);
    fq2_mul(h, i, j);
    fq2_mul(u1, i, v);
    fq2_sqr(r, t0);
    fq2_sub(t0, j, t1);
    fq2_double(v, t0);
    fq2_sub(t1, t0, out);
    fq2_sub(v, out, t0);
    fq2_mul(r, t0, t1);
    fq2_mul(s1, j, t0);
    fq2_double(t0, j);
    fq2_sub(t1, j, out + FQ2_LIMBS);
    fq2_copy(z3, out + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void jac2_scale(const u64 *point, const u64 *scalar,
                                        unsigned int scalar_bits, u64 *out) {
    u64 acc[G2_LIMBS], tmp[G2_LIMBS];
    jac2_set_zero(acc);
    if (jac2_is_zero(point)) {
        jac2_copy(acc, out);
        return;
    }
    for (int bit = (int)scalar_bits - 1; bit >= 0; bit--) {
        jac2_double(acc, tmp);
        if (((scalar[bit >> 6] >> (bit & 63)) & 1ULL) != 0ULL) {
            jac2_add(tmp, point, acc);
        } else {
            jac2_copy(tmp, acc);
        }
    }
    jac2_copy(acc, out);
}

__device__ __forceinline__ void g2_psi_power(const u64 *base, const u64 *coefficients,
                                            unsigned int power, u64 *out) {
    u64 scaled[2 * LIMBS], negated[LIMBS];
    jac2_copy(base, out);
    if ((power & 1u) == 1u) {
        fq_neg(out + LIMBS, negated);
        fq_copy(negated, out + LIMBS);
        fq_neg(out + 3 * LIMBS, negated);
        fq_copy(negated, out + 3 * LIMBS);
        fq_neg(out + 5 * LIMBS, negated);
        fq_copy(negated, out + 5 * LIMBS);
    }
    const u64 *coef_x = coefficients + (unsigned long long)(power - 1) * 4 * LIMBS;
    const u64 *coef_y = coef_x + 2 * LIMBS;
    fq2_mul(out, coef_x, scaled);
    fq2_copy(scaled, out);
    fq2_mul(out + 2 * LIMBS, coef_y, scaled);
    fq2_copy(scaled, out + 2 * LIMBS);
}

__device__ __forceinline__ void g2_negate_y(u64 *point) {
    u64 negated[LIMBS];
    fq_neg(point + 2 * LIMBS, negated);
    fq_copy(negated, point + 2 * LIMBS);
    fq_neg(point + 3 * LIMBS, negated);
    fq_copy(negated, point + 3 * LIMBS);
}

extern "C" __global__ void msm_g2_axpy_glv_kernel(
    u64 *buf, const u64 *__restrict__ coeffs, const u64 *__restrict__ frobenius,
    unsigned int signs, unsigned int max_bits, unsigned int a_offset, unsigned int b_offset,
    unsigned int out_offset, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const u64 *a = buf + ((unsigned long long)a_offset + i) * G2_LIMBS;
    const u64 *b = buf + ((unsigned long long)b_offset + i) * G2_LIMBS;
    u64 *out = buf + ((unsigned long long)out_offset + i) * G2_LIMBS;

    u64 bases[4][G2_LIMBS];
    jac2_copy(a, bases[0]);
    for (unsigned int power = 1; power < 4; power++) {
        g2_psi_power(bases[0], frobenius, power, bases[power]);
    }
    for (unsigned int j = 0; j < 4; j++) {
        if (((signs >> j) & 1u) != 0u) g2_negate_y(bases[j]);
    }

    u64 acc[G2_LIMBS], tmp[G2_LIMBS];
    jac2_set_zero(acc);
    for (int bit = (int)max_bits - 1; bit >= 0; bit--) {
        jac2_double(acc, tmp);
        jac2_copy(tmp, acc);
        for (unsigned int j = 0; j < 4; j++) {
            const u64 *coeff = coeffs + (unsigned long long)j * LIMBS;
            if (((coeff[bit >> 6] >> (bit & 63)) & 1ULL) != 0ULL) {
                jac2_add(acc, bases[j], tmp);
                jac2_copy(tmp, acc);
            }
        }
    }
    jac2_add(acc, b, tmp);
    jac2_copy(tmp, out);
}

extern "C" __global__ void msm_g2_fixed_base_kernel(u64 *buf,
                                                   const u64 *__restrict__ scalars,
                                                   unsigned int base_offset,
                                                   unsigned int out_offset, unsigned int count,
                                                   unsigned int scalar_bits) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const u64 *base = buf + (unsigned long long)base_offset * G2_LIMBS;
    u64 *out = buf + ((unsigned long long)out_offset + i) * G2_LIMBS;

    u64 scaled[G2_LIMBS];
    jac2_scale(base, scalars + (unsigned long long)i * LIMBS, scalar_bits, scaled);
    jac2_copy(scaled, out);
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

extern "C" __global__ void msm_g2_segment_sum_small_kernel(const u64 *__restrict__ bases,
                                                          const unsigned int *__restrict__ indices,
                                                          const unsigned int *__restrict__ offsets,
                                                          const unsigned int *__restrict__ counts,
                                                          unsigned int segments,
                                                          u64 *__restrict__ out) {
    unsigned int segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    unsigned int start = offsets[segment];
    unsigned int end = start + counts[segment];

    u64 acc[G2_LIMBS], tmp[G2_LIMBS], base[G2_LIMBS], negated[LIMBS];
    jac2_set_zero(acc);
    for (unsigned int i = start; i < end; i++) {
        unsigned int index = indices[i];
        unsigned int negate = index >> 31;
        index &= 0x7fffffffu;
        jac2_copy(bases + (unsigned long long)index * G2_LIMBS, base);
        if (negate != 0) {
            fq_neg(base + 2 * LIMBS, negated);
            fq_copy(negated, base + 2 * LIMBS);
            fq_neg(base + 3 * LIMBS, negated);
            fq_copy(negated, base + 3 * LIMBS);
        }
        jac2_add(acc, base, tmp);
        jac2_copy(tmp, acc);
    }
    jac2_copy(acc, out + (unsigned long long)segment * G2_LIMBS);
}

extern "C" __global__ void msm_g2_bucket_reduce_chunked_kernel(
    const u64 *__restrict__ buckets_points, unsigned int rows, unsigned int buckets,
    unsigned int chunks, u64 *__restrict__ out) {
    extern __shared__ u64 scratch2[];
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

    u64 acc[G2_LIMBS], running[G2_LIMBS], tmp[G2_LIMBS], bucket[G2_LIMBS];
    jac2_set_zero(acc);
    jac2_set_zero(running);
    if (lo < hi) {
        for (int b = (int)hi - 1; b >= (int)lo; b--) {
            jac2_copy(buckets_points +
                          ((unsigned long long)row * buckets + (unsigned int)b) * G2_LIMBS,
                      bucket);
            jac2_add(acc, bucket, tmp);
            jac2_copy(tmp, acc);
            jac2_add(running, acc, tmp);
            jac2_copy(tmp, running);
        }
        unsigned int weight = lo - 1;
        if (weight != 0) {
            u64 scaled[G2_LIMBS], addend[G2_LIMBS];
            jac2_set_zero(scaled);
            jac2_copy(acc, addend);
            while (weight != 0) {
                if ((weight & 1u) != 0) {
                    jac2_add(scaled, addend, tmp);
                    jac2_copy(tmp, scaled);
                }
                jac2_double(addend, tmp);
                jac2_copy(tmp, addend);
                weight >>= 1;
            }
            jac2_add(running, scaled, tmp);
            jac2_copy(tmp, running);
        }
    }

    u64 *slot = scratch2 + (unsigned long long)threadIdx.x * G2_LIMBS;
    jac2_copy(running, slot);
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            jac2_add(slot, scratch2 + (unsigned long long)(threadIdx.x + stride) * G2_LIMBS, tmp);
            jac2_copy(tmp, slot);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        jac2_copy(scratch2, out + ((unsigned long long)row * chunks + chunk) * G2_LIMBS);
    }
}

extern "C" __global__ void msm_g2_point_rows_sum_kernel(const u64 *__restrict__ partials,
                                                       unsigned int rows, unsigned int count,
                                                       u64 *__restrict__ out) {
    extern __shared__ u64 scratch2[];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    u64 acc[G2_LIMBS], tmp[G2_LIMBS], value[G2_LIMBS];
    jac2_set_zero(acc);
    for (unsigned int index = threadIdx.x; index < count; index += blockDim.x) {
        jac2_copy(partials + ((unsigned long long)row * count + index) * G2_LIMBS, value);
        jac2_add(acc, value, tmp);
        jac2_copy(tmp, acc);
    }

    u64 *slot = scratch2 + (unsigned long long)threadIdx.x * G2_LIMBS;
    jac2_copy(acc, slot);
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            jac2_add(slot, scratch2 + (unsigned long long)(threadIdx.x + stride) * G2_LIMBS, tmp);
            jac2_copy(tmp, slot);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        jac2_copy(scratch2, out + (unsigned long long)row * G2_LIMBS);
    }
}

extern "C" __global__ void msm_g2_window_fold_kernel(const u64 *__restrict__ window_points,
                                                    unsigned int rows, unsigned int windows,
                                                    unsigned int window_bits,
                                                    u64 *__restrict__ out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    u64 acc[G2_LIMBS], addend[G2_LIMBS], tmp[G2_LIMBS];
    jac2_set_zero(acc);
    for (int window = (int)windows - 1; window >= 0; window--) {
        if ((unsigned int)window + 1 != windows) {
            for (unsigned int i = 0; i < window_bits; i++) {
                jac2_double(acc, tmp);
                jac2_copy(tmp, acc);
            }
        }
        jac2_copy(window_points +
                      ((unsigned long long)(unsigned int)window * rows + row) * G2_LIMBS,
                  addend);
        jac2_add(acc, addend, tmp);
        jac2_copy(tmp, acc);
    }
    jac2_copy(acc, out + (unsigned long long)row * G2_LIMBS);
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

extern "C" __global__ void msm_glv_decompose_4d_kernel(
    const u64 *__restrict__ scalars, const u64 *__restrict__ table,
    const unsigned char *__restrict__ table_signs, unsigned int count,
    u64 *__restrict__ out_scalars, unsigned char *__restrict__ out_signs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    u64 k[LIMBS];
    load4(scalars + (unsigned long long)i * LIMBS, k);

    u64 acc[4][2];
    for (int j = 0; j < 4; j++) {
        acc[j][0] = 0;
        acc[j][1] = 0;
    }

    for (int bit = 0; bit < 254; bit++) {
        unsigned int limb = (unsigned int)(bit >> 6);
        unsigned int shift = (unsigned int)(bit & 63);
        if (((k[limb] >> shift) & 1ull) == 0ull) continue;
        const u64 *row = table + (unsigned long long)bit * 8;
        const unsigned char *row_signs = table_signs + (unsigned long long)bit * 4;
        for (int j = 0; j < 4; j++) {
            u64 lo = row[j * 2];
            u64 hi = row[j * 2 + 1];
            if (row_signs[j] != 0) {
                u64 borrow = 0;
                acc[j][0] = sbb(acc[j][0], lo, &borrow);
                acc[j][1] = sbb(acc[j][1], hi, &borrow);
            } else {
                u64 carry = 0;
                acc[j][0] = adc(acc[j][0], lo, &carry);
                acc[j][1] = adc(acc[j][1], hi, &carry);
            }
        }
    }

    for (int j = 0; j < 4; j++) {
        u64 magnitude[LIMBS];
        unsigned char negative = (unsigned char)((acc[j][1] >> 63) & 1ull);
        if (negative != 0) {
            u64 borrow = 0;
            magnitude[0] = sbb(0ull, acc[j][0], &borrow);
            magnitude[1] = sbb(0ull, acc[j][1], &borrow);
        } else {
            magnitude[0] = acc[j][0];
            magnitude[1] = acc[j][1];
        }
        magnitude[2] = 0;
        magnitude[3] = 0;
        unsigned long long slot = (unsigned long long)j * count + i;
        store4(out_scalars + slot * LIMBS, magnitude);
        out_signs[slot] = negative;
    }
}

extern "C" __global__ void msm_g2_frobenius_kernel(const u64 *__restrict__ jacobian,
                                                  const u64 *__restrict__ coefficients,
                                                  unsigned int count, u64 *__restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    u64 base[G2_LIMBS];
    jac2_copy(jacobian + (unsigned long long)i * G2_LIMBS, base);
    jac2_copy(base, out + (unsigned long long)i * G2_LIMBS);

    for (unsigned int power = 1; power < 4; power++) {
        u64 res[G2_LIMBS];
        g2_psi_power(base, coefficients, power, res);
        jac2_copy(res, out + ((unsigned long long)power * count + i) * G2_LIMBS);
    }
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
                                                const u64 *__restrict__ left,
                                                unsigned long long base, unsigned long long len,
                                                unsigned int sigma, unsigned int rows,
                                                unsigned int columns, u64 *__restrict__ out) {
    unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    u64 acc[LIMBS] = {0, 0, 0, 0};
    unsigned long long span = (unsigned long long)columns;
    unsigned long long mask = span - 1ull;
    unsigned long long first = ((unsigned long long)column + span - (base & mask)) & mask;
    unsigned long long row = (base + first) >> sigma;
    for (unsigned long long local = first; local < len; local += span, row++) {
        if (row >= rows) break;
        u64 weight[LIMBS], value[LIMBS], term[LIMBS], sum[LIMBS];
        load4(left + row * LIMBS, weight);
        load4(table + local * LIMBS, value);
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

extern "C" __global__ void msm_shared_scalar_rows_glv_kernel(
    const u64 *__restrict__ bases, const u64 *__restrict__ coeffs,
    const unsigned char *__restrict__ signs, const u64 *__restrict__ beta, unsigned int rows,
    unsigned int terms, unsigned int max_bits, u64 *__restrict__ out) {
    extern __shared__ u64 scratch[];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    u64 acc[3 * LIMBS], tmp[3 * LIMBS], product[3 * LIMBS];
    u64 low[3 * LIMBS], high[3 * LIMBS], factor[LIMBS], scaled[LIMBS];
    load4(beta, factor);
    jac_set_zero(acc);
    for (unsigned int term = threadIdx.x; term < terms; term += blockDim.x) {
        const u64 *source = bases + ((unsigned long long)term * rows + row) * 3 * LIMBS;
        for (int limb = 0; limb < 3; limb++) {
            load4(source + limb * LIMBS, low + limb * LIMBS);
        }
        if (jac_is_zero(low)) continue;

        jac_copy(low, high);
        fq_mul(low, factor, scaled);
        fq_copy(scaled, high);
        if (signs[term] != 0u) {
            fq_neg(low + LIMBS, scaled);
            fq_copy(scaled, low + LIMBS);
        }
        if (signs[(unsigned long long)terms + term] != 0u) {
            fq_neg(high + LIMBS, scaled);
            fq_copy(scaled, high + LIMBS);
        }

        const u64 *first = coeffs + (unsigned long long)term * LIMBS;
        const u64 *second = coeffs + ((unsigned long long)terms + term) * LIMBS;
        jac_set_zero(product);
        for (int bit = (int)max_bits - 1; bit >= 0; bit--) {
            jac_double(product, tmp);
            jac_copy(tmp, product);
            if (((first[bit >> 6] >> (bit & 63)) & 1ULL) != 0ULL) {
                jac_add(product, low, tmp);
                jac_copy(tmp, product);
            }
            if (((second[bit >> 6] >> (bit & 63)) & 1ULL) != 0ULL) {
                jac_add(product, high, tmp);
                jac_copy(tmp, product);
            }
        }
        jac_add(acc, product, tmp);
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

extern "C" __global__ void msm_g1_axpy_kernel(u64 *buf, const u64 *__restrict__ scalar,
                                              unsigned int a_offset, unsigned int b_offset,
                                              unsigned int out_offset, unsigned int count,
                                              unsigned int scalar_bits) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const u64 *a = buf + ((unsigned long long)a_offset + i) * 3 * LIMBS;
    const u64 *b = buf + ((unsigned long long)b_offset + i) * 3 * LIMBS;
    u64 *out = buf + ((unsigned long long)out_offset + i) * 3 * LIMBS;

    u64 term[3 * LIMBS], acc[3 * LIMBS], tmp[3 * LIMBS];
    for (int limb = 0; limb < 3; limb++) {
        load4(a + limb * LIMBS, term + limb * LIMBS);
    }
    jac_set_zero(acc);
    if (!jac_is_zero(term)) {
        for (int bit = (int)scalar_bits - 1; bit >= 0; bit--) {
            jac_double(acc, tmp);
            if (((scalar[bit >> 6] >> (bit & 63)) & 1ULL) != 0ULL) {
                jac_add(tmp, term, acc);
            } else {
                jac_copy(tmp, acc);
            }
        }
    }
    for (int limb = 0; limb < 3; limb++) {
        load4(b + limb * LIMBS, term + limb * LIMBS);
    }
    jac_add(acc, term, tmp);
    for (int limb = 0; limb < 3; limb++) {
        store4(out + limb * LIMBS, tmp + limb * LIMBS);
    }
}
