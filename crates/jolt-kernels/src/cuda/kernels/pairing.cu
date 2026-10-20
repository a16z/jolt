#define FQ6_LIMBS (3 * FQ2_LIMBS)
#define FQ12_LIMBS (2 * FQ6_LIMBS)

#define PC_TWIST_Q_X 0
#define PC_TWIST_Q_Y 8
#define PC_COEFF_B 16
#define PC_TWO_INV 24
#define PC_WORDS 28

static_assert(FQ2_LIMBS == 8, "Fq2 is two Fq limbs wide");
static_assert(PC_WORDS == 28, "the pairing constant layout must match pairing.rs");

__device__ __forceinline__ void fq2_set_one(u64 *out) {
    fq_copy(FQ_MONT_ONE, out);
    for (int i = 0; i < LIMBS; i++) out[LIMBS + i] = 0;
}

__device__ __forceinline__ void fq2_neg(const u64 *a, u64 *out) {
    fq_neg(a, out);
    fq_neg(a + LIMBS, out + LIMBS);
}

__device__ __forceinline__ void fq2_conj(const u64 *a, u64 *out) {
    fq_copy(a, out);
    fq_neg(a + LIMBS, out + LIMBS);
}

__device__ __forceinline__ void fq2_mul_by_fp(const u64 *a, const u64 *k, u64 *out) {
    fq_mul(a, k, out);
    fq_mul(a + LIMBS, k, out + LIMBS);
}

__device__ void fq2_mul_by_nonresidue(const u64 *a, u64 *out) {
    u64 nine_a0[LIMBS], nine_a1[LIMBS], t[LIMBS], u[LIMBS], c0[LIMBS];
    fq_double(a, t);
    fq_double(t, u);
    fq_double(u, t);
    fq_add(t, a, nine_a0);
    fq_double(a + LIMBS, t);
    fq_double(t, u);
    fq_double(u, t);
    fq_add(t, a + LIMBS, nine_a1);
    fq_sub(nine_a0, a + LIMBS, c0);
    fq_add(nine_a1, a, out + LIMBS);
    fq_copy(c0, out);
}

__device__ __forceinline__ void fq6_copy(const u64 *a, u64 *out) {
    for (int i = 0; i < FQ6_LIMBS; i++) out[i] = a[i];
}

__device__ __forceinline__ void fq6_add(const u64 *a, const u64 *b, u64 *out) {
    for (int i = 0; i < 3; i++) {
        fq2_add(a + i * FQ2_LIMBS, b + i * FQ2_LIMBS, out + i * FQ2_LIMBS);
    }
}

__device__ __forceinline__ void fq6_sub(const u64 *a, const u64 *b, u64 *out) {
    for (int i = 0; i < 3; i++) {
        fq2_sub(a + i * FQ2_LIMBS, b + i * FQ2_LIMBS, out + i * FQ2_LIMBS);
    }
}

__device__ void fq6_mul_by_nonresidue(const u64 *a, u64 *out) {
    u64 c0[FQ2_LIMBS], c1[FQ2_LIMBS];
    fq2_mul_by_nonresidue(a + 2 * FQ2_LIMBS, c0);
    fq2_copy(a, c1);
    fq2_copy(a + FQ2_LIMBS, out + 2 * FQ2_LIMBS);
    fq2_copy(c1, out + FQ2_LIMBS);
    fq2_copy(c0, out);
}

__device__ __noinline__ void fq6_mul(const u64 *lhs, const u64 *rhs, u64 *out) {
    const u64 *d = lhs;
    const u64 *e = lhs + FQ2_LIMBS;
    const u64 *f = lhs + 2 * FQ2_LIMBS;
    const u64 *a = rhs;
    const u64 *b = rhs + FQ2_LIMBS;
    const u64 *c = rhs + 2 * FQ2_LIMBS;

    u64 ad[FQ2_LIMBS], be[FQ2_LIMBS], cf[FQ2_LIMBS];
    u64 x[FQ2_LIMBS], y[FQ2_LIMBS], z[FQ2_LIMBS], s[FQ2_LIMBS], t[FQ2_LIMBS];
    fq2_mul(d, a, ad);
    fq2_mul(e, b, be);
    fq2_mul(f, c, cf);

    fq2_add(e, f, s);
    fq2_add(b, c, t);
    fq2_mul(s, t, x);
    fq2_sub(x, be, x);
    fq2_sub(x, cf, x);

    fq2_add(d, e, s);
    fq2_add(a, b, t);
    fq2_mul(s, t, y);
    fq2_sub(y, ad, y);
    fq2_sub(y, be, y);

    fq2_add(d, f, s);
    fq2_add(a, c, t);
    fq2_mul(s, t, z);
    fq2_sub(z, ad, z);
    fq2_add(z, be, z);
    fq2_sub(z, cf, z);

    fq2_mul_by_nonresidue(x, s);
    fq2_add(ad, s, x);
    fq2_mul_by_nonresidue(cf, s);
    fq2_add(y, s, t);

    fq2_copy(x, out);
    fq2_copy(t, out + FQ2_LIMBS);
    fq2_copy(z, out + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void fq6_mul_by_01(const u64 *a, const u64 *c0, const u64 *c1, u64 *out) {
    u64 a_a[FQ2_LIMBS], b_b[FQ2_LIMBS], t1[FQ2_LIMBS], t2[FQ2_LIMBS], t3[FQ2_LIMBS];
    u64 tmp[FQ2_LIMBS], scratch[FQ2_LIMBS];
    fq2_mul(a, c0, a_a);
    fq2_mul(a + FQ2_LIMBS, c1, b_b);

    fq2_add(a + FQ2_LIMBS, a + 2 * FQ2_LIMBS, tmp);
    fq2_mul(c1, tmp, t1);
    fq2_sub(t1, b_b, t1);
    fq2_mul_by_nonresidue(t1, scratch);
    fq2_add(scratch, a_a, t1);

    fq2_add(a, a + 2 * FQ2_LIMBS, tmp);
    fq2_mul(c0, tmp, t3);
    fq2_sub(t3, a_a, t3);
    fq2_add(t3, b_b, t3);

    fq2_add(c0, c1, scratch);
    fq2_add(a, a + FQ2_LIMBS, tmp);
    fq2_mul(scratch, tmp, t2);
    fq2_sub(t2, a_a, t2);
    fq2_sub(t2, b_b, t2);

    fq2_copy(t1, out);
    fq2_copy(t2, out + FQ2_LIMBS);
    fq2_copy(t3, out + 2 * FQ2_LIMBS);
}

__device__ void fq6_mul_by_fq2(const u64 *a, const u64 *k, u64 *out) {
    for (int i = 0; i < 3; i++) {
        fq2_mul(a + i * FQ2_LIMBS, k, out + i * FQ2_LIMBS);
    }
}

__device__ __forceinline__ void fq12_copy(const u64 *a, u64 *out) {
    for (int i = 0; i < FQ12_LIMBS; i++) out[i] = a[i];
}

__device__ __forceinline__ void fq12_set_one(u64 *out) {
    for (int i = 0; i < FQ12_LIMBS; i++) out[i] = 0;
    fq2_set_one(out);
}

__device__ __noinline__ void fq12_mul(const u64 *a, const u64 *b, u64 *out) {
    const u64 *a0 = a;
    const u64 *a1 = a + FQ6_LIMBS;
    const u64 *b0 = b;
    const u64 *b1 = b + FQ6_LIMBS;

    u64 v0[FQ6_LIMBS], v1[FQ6_LIMBS], s[FQ6_LIMBS], t[FQ6_LIMBS], c1[FQ6_LIMBS];
    fq6_mul(a0, b0, v0);
    fq6_mul(a1, b1, v1);
    fq6_add(a1, a0, s);
    fq6_add(b0, b1, t);
    fq6_mul(s, t, c1);
    fq6_sub(c1, v0, c1);
    fq6_sub(c1, v1, c1);
    fq6_mul_by_nonresidue(v1, s);
    fq6_add(s, v0, t);
    fq6_copy(t, out);
    fq6_copy(c1, out + FQ6_LIMBS);
}

__device__ __noinline__ void fq12_sqr(const u64 *a, u64 *out) {
    const u64 *a0 = a;
    const u64 *a1 = a + FQ6_LIMBS;

    u64 v0[FQ6_LIMBS], v2[FQ6_LIMBS], v3[FQ6_LIMBS], s[FQ6_LIMBS], c0[FQ6_LIMBS];
    fq6_sub(a0, a1, v0);
    fq6_mul_by_nonresidue(a1, s);
    fq6_sub(a0, s, v3);
    fq6_mul(a0, a1, v2);
    fq6_mul(v0, v3, v0);

    fq6_mul_by_nonresidue(v2, s);
    fq6_add(s, v2, c0);
    fq6_add(c0, v0, c0);
    fq6_add(v2, v2, s);

    fq6_copy(c0, out);
    fq6_copy(s, out + FQ6_LIMBS);
}

__device__ __noinline__ void fq12_mul_by_034(const u64 *f, const u64 *c0, const u64 *c3,
                                             const u64 *c4, u64 *out) {
    const u64 *f0 = f;
    const u64 *f1 = f + FQ6_LIMBS;

    u64 a[FQ6_LIMBS], b[FQ6_LIMBS], e[FQ6_LIMBS], s[FQ6_LIMBS];
    u64 d0[FQ2_LIMBS];

    fq6_mul_by_fq2(f0, c0, a);
    fq6_mul_by_01(f1, c3, c4, b);

    fq2_add(c0, c3, d0);
    fq6_add(f0, f1, e);
    fq6_mul_by_01(e, d0, c4, s);

    fq6_add(a, b, e);
    fq6_sub(s, e, e);

    fq6_mul_by_nonresidue(b, s);
    fq6_add(s, a, s);

    fq6_copy(s, out);
    fq6_copy(e, out + FQ6_LIMBS);
}

__device__ __noinline__ void g2_double_step(u64 *rx, u64 *ry, u64 *rz, const u64 *consts,
                                            u64 *coeff) {
    const u64 *two_inv = consts + PC_TWO_INV;
    const u64 *coeff_b = consts + PC_COEFF_B;

    u64 a[FQ2_LIMBS], b[FQ2_LIMBS], c[FQ2_LIMBS], e[FQ2_LIMBS], f[FQ2_LIMBS];
    u64 g[FQ2_LIMBS], h[FQ2_LIMBS], i[FQ2_LIMBS], j[FQ2_LIMBS];
    u64 e_square[FQ2_LIMBS], t[FQ2_LIMBS], u[FQ2_LIMBS];

    fq2_mul(rx, ry, t);
    fq2_mul_by_fp(t, two_inv, a);
    fq2_sqr(ry, b);
    fq2_sqr(rz, c);
    fq2_double(c, t);
    fq2_add(t, c, t);
    fq2_mul(coeff_b, t, e);
    fq2_double(e, t);
    fq2_add(t, e, f);
    fq2_add(b, f, t);
    fq2_mul_by_fp(t, two_inv, g);
    fq2_add(ry, rz, t);
    fq2_sqr(t, h);
    fq2_add(b, c, t);
    fq2_sub(h, t, h);
    fq2_sub(e, b, i);
    fq2_sqr(rx, j);
    fq2_sqr(e, e_square);

    fq2_sub(b, f, t);
    fq2_mul(a, t, u);
    fq2_sqr(g, t);
    fq2_double(e_square, a);
    fq2_add(a, e_square, a);
    fq2_sub(t, a, t);
    fq2_mul(b, h, a);

    fq2_copy(u, rx);
    fq2_copy(t, ry);
    fq2_copy(a, rz);

    fq2_neg(h, coeff);
    fq2_double(j, t);
    fq2_add(t, j, coeff + FQ2_LIMBS);
    fq2_copy(i, coeff + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void g2_add_step(u64 *rx, u64 *ry, u64 *rz, const u64 *qx, const u64 *qy,
                                         u64 *coeff) {
    u64 theta[FQ2_LIMBS], lambda[FQ2_LIMBS], c[FQ2_LIMBS], d[FQ2_LIMBS];
    u64 e[FQ2_LIMBS], f[FQ2_LIMBS], g[FQ2_LIMBS], h[FQ2_LIMBS];
    u64 j[FQ2_LIMBS], t[FQ2_LIMBS], u[FQ2_LIMBS];

    fq2_mul(qy, rz, t);
    fq2_sub(ry, t, theta);
    fq2_mul(qx, rz, t);
    fq2_sub(rx, t, lambda);
    fq2_sqr(theta, c);
    fq2_sqr(lambda, d);
    fq2_mul(lambda, d, e);
    fq2_mul(rz, c, f);
    fq2_mul(rx, d, g);
    fq2_double(g, t);
    fq2_add(e, f, h);
    fq2_sub(h, t, h);

    fq2_mul(theta, qx, t);
    fq2_mul(lambda, qy, u);
    fq2_sub(t, u, j);

    fq2_mul(lambda, h, t);
    fq2_sub(g, h, u);
    fq2_mul(theta, u, u);
    fq2_mul(e, ry, c);
    fq2_sub(u, c, u);
    fq2_mul(rz, e, d);

    fq2_copy(t, rx);
    fq2_copy(u, ry);
    fq2_copy(d, rz);

    fq2_copy(lambda, coeff);
    fq2_neg(theta, coeff + FQ2_LIMBS);
    fq2_copy(j, coeff + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void ell(u64 *f, const u64 *coeff, const u64 *px, const u64 *py) {
    u64 c0[FQ2_LIMBS], c1[FQ2_LIMBS], scratch[FQ12_LIMBS];
    fq2_mul_by_fp(coeff, py, c0);
    fq2_mul_by_fp(coeff + FQ2_LIMBS, px, c1);
    fq12_mul_by_034(f, c0, c1, coeff + 2 * FQ2_LIMBS, scratch);
    fq12_copy(scratch, f);
}

__device__ __noinline__ void g2_mul_by_char(const u64 *qx, const u64 *qy, const u64 *consts,
                                            u64 *outx, u64 *outy) {
    u64 t[FQ2_LIMBS];
    fq2_conj(qx, t);
    fq2_mul(t, consts + PC_TWIST_Q_X, outx);
    fq2_conj(qy, t);
    fq2_mul(t, consts + PC_TWIST_Q_Y, outy);
}

__device__ __forceinline__ void mw_fq2_mul(const u64 *a, const u64 *b, u64 *out) {
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

__device__ __forceinline__ void mw_fq2_sqr(const u64 *a, u64 *out) {
    u64 s[LIMBS], d[LIMBS], m[LIMBS], c0[LIMBS];
    fq_add(a, a + LIMBS, s);
    fq_sub(a, a + LIMBS, d);
    fq_mul(a, a + LIMBS, m);
    fq_mul(s, d, c0);
    fq_double(m, out + LIMBS);
    fq_copy(c0, out);
}

__device__ __noinline__ void mw_g2_double_step(u64 *rx, u64 *ry, u64 *rz, const u64 *consts,
                                                  u64 *coeff) {
    const u64 *two_inv = consts + PC_TWO_INV;
    const u64 *coeff_b = consts + PC_COEFF_B;

    u64 a[FQ2_LIMBS], b[FQ2_LIMBS], c[FQ2_LIMBS], e[FQ2_LIMBS], f[FQ2_LIMBS];
    u64 g[FQ2_LIMBS], h[FQ2_LIMBS], i[FQ2_LIMBS], j[FQ2_LIMBS];
    u64 e_square[FQ2_LIMBS], t[FQ2_LIMBS], u[FQ2_LIMBS];

    mw_fq2_mul(rx, ry, t);
    fq2_mul_by_fp(t, two_inv, a);
    mw_fq2_sqr(ry, b);
    mw_fq2_sqr(rz, c);
    fq2_double(c, t);
    fq2_add(t, c, t);
    mw_fq2_mul(coeff_b, t, e);
    fq2_double(e, t);
    fq2_add(t, e, f);
    fq2_add(b, f, t);
    fq2_mul_by_fp(t, two_inv, g);
    fq2_add(ry, rz, t);
    mw_fq2_sqr(t, h);
    fq2_add(b, c, t);
    fq2_sub(h, t, h);
    fq2_sub(e, b, i);
    mw_fq2_sqr(rx, j);
    mw_fq2_sqr(e, e_square);

    fq2_sub(b, f, t);
    mw_fq2_mul(a, t, u);
    mw_fq2_sqr(g, t);
    fq2_double(e_square, a);
    fq2_add(a, e_square, a);
    fq2_sub(t, a, t);
    mw_fq2_mul(b, h, a);

    fq2_copy(u, rx);
    fq2_copy(t, ry);
    fq2_copy(a, rz);

    fq2_neg(h, coeff);
    fq2_double(j, t);
    fq2_add(t, j, coeff + FQ2_LIMBS);
    fq2_copy(i, coeff + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void mw_g2_add_step(u64 *rx, u64 *ry, u64 *rz, const u64 *qx,
                                               const u64 *qy, u64 *coeff) {
    u64 theta[FQ2_LIMBS], lambda[FQ2_LIMBS], c[FQ2_LIMBS], d[FQ2_LIMBS];
    u64 e[FQ2_LIMBS], f[FQ2_LIMBS], g[FQ2_LIMBS], h[FQ2_LIMBS];
    u64 j[FQ2_LIMBS], t[FQ2_LIMBS], u[FQ2_LIMBS];

    mw_fq2_mul(qy, rz, t);
    fq2_sub(ry, t, theta);
    mw_fq2_mul(qx, rz, t);
    fq2_sub(rx, t, lambda);
    mw_fq2_sqr(theta, c);
    mw_fq2_sqr(lambda, d);
    mw_fq2_mul(lambda, d, e);
    mw_fq2_mul(rz, c, f);
    mw_fq2_mul(rx, d, g);
    fq2_double(g, t);
    fq2_add(e, f, h);
    fq2_sub(h, t, h);

    mw_fq2_mul(theta, qx, t);
    mw_fq2_mul(lambda, qy, u);
    fq2_sub(t, u, j);

    mw_fq2_mul(lambda, h, t);
    fq2_sub(g, h, u);
    mw_fq2_mul(theta, u, u);
    mw_fq2_mul(e, ry, c);
    fq2_sub(u, c, u);
    mw_fq2_mul(rz, e, d);

    fq2_copy(t, rx);
    fq2_copy(u, ry);
    fq2_copy(d, rz);

    fq2_copy(lambda, coeff);
    fq2_neg(theta, coeff + FQ2_LIMBS);
    fq2_copy(j, coeff + 2 * FQ2_LIMBS);
}

__device__ __noinline__ void mw_g2_mul_by_char(const u64 *qx, const u64 *qy, const u64 *consts,
                                                  u64 *outx, u64 *outy) {
    u64 t[FQ2_LIMBS];
    fq2_conj(qx, t);
    mw_fq2_mul(t, consts + PC_TWIST_Q_X, outx);
    fq2_conj(qy, t);
    mw_fq2_mul(t, consts + PC_TWIST_Q_Y, outy);
}

extern "C" __global__ void pairing_fq12_product_kernel(const u64 *__restrict__ values,
                                                       unsigned int count,
                                                       unsigned int chunk_size,
                                                       u64 *__restrict__ out) {
    extern __shared__ u64 shared[];
    u64 *slot = shared + (unsigned long long)threadIdx.x * FQ12_LIMBS;
    const unsigned int start = blockIdx.x * chunk_size;
    const unsigned int chunk_count = min(chunk_size, count - start);
    const u64 *lane =
        values + ((unsigned long long)blockIdx.y * count + start) * FQ12_LIMBS;

    u64 acc[FQ12_LIMBS], scratch[FQ12_LIMBS];
    fq12_set_one(acc);
    for (unsigned int i = threadIdx.x; i < chunk_count; i += blockDim.x) {
        fq12_mul(acc, lane + (unsigned long long)i * FQ12_LIMBS, scratch);
        fq12_copy(scratch, acc);
    }
    fq12_copy(acc, slot);
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            fq12_mul(slot, shared + (unsigned long long)(threadIdx.x + stride) * FQ12_LIMBS,
                     scratch);
            fq12_copy(scratch, slot);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        u64 *target =
            out + ((unsigned long long)blockIdx.y * gridDim.x + blockIdx.x) * FQ12_LIMBS;
        for (int i = 0; i < FQ12_LIMBS; i++) target[i] = shared[i];
    }
}

extern "C" __global__ void fq6_mul_probe(const u64 *__restrict__ in, u64 *__restrict__ out,
                                        unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *a = in + (unsigned long long)i * 2 * FQ6_LIMBS;
    fq6_mul(a, a + FQ6_LIMBS, out + (unsigned long long)i * FQ6_LIMBS);
}

extern "C" __global__ void fq6_mul_by_01_probe(const u64 *__restrict__ in, u64 *__restrict__ out,
                                              unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *a = in + (unsigned long long)i * (FQ6_LIMBS + 2 * FQ2_LIMBS);
    fq6_mul_by_01(a, a + FQ6_LIMBS, a + FQ6_LIMBS + FQ2_LIMBS,
                  out + (unsigned long long)i * FQ6_LIMBS);
}

extern "C" __global__ void fq6_mul_by_fq2_probe(const u64 *__restrict__ in, u64 *__restrict__ out,
                                               unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *a = in + (unsigned long long)i * (FQ6_LIMBS + FQ2_LIMBS);
    fq6_mul_by_fq2(a, a + FQ6_LIMBS, out + (unsigned long long)i * FQ6_LIMBS);
}

extern "C" __global__ void fq6_mul_by_nonresidue_probe(const u64 *__restrict__ in,
                                                      u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fq6_mul_by_nonresidue(in + (unsigned long long)i * FQ6_LIMBS,
                          out + (unsigned long long)i * FQ6_LIMBS);
}

extern "C" __global__ void fq12_mul_probe(const u64 *__restrict__ in, u64 *__restrict__ out,
                                         unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *a = in + (unsigned long long)i * 2 * FQ12_LIMBS;
    fq12_mul(a, a + FQ12_LIMBS, out + (unsigned long long)i * FQ12_LIMBS);
}

extern "C" __global__ void fq12_sqr_probe(const u64 *__restrict__ in, u64 *__restrict__ out,
                                         unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fq12_sqr(in + (unsigned long long)i * FQ12_LIMBS, out + (unsigned long long)i * FQ12_LIMBS);
}

extern "C" __global__ void fq12_mul_by_034_probe(const u64 *__restrict__ in,
                                                u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *a = in + (unsigned long long)i * (FQ12_LIMBS + 3 * FQ2_LIMBS);
    fq12_mul_by_034(a, a + FQ12_LIMBS, a + FQ12_LIMBS + FQ2_LIMBS,
                    a + FQ12_LIMBS + 2 * FQ2_LIMBS, out + (unsigned long long)i * FQ12_LIMBS);
}

extern "C" __global__ void ell_probe(const u64 *__restrict__ in, u64 *__restrict__ out,
                                    unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *src = in + (unsigned long long)i * (FQ12_LIMBS + 3 * FQ2_LIMBS + 2 * LIMBS);
    u64 f[FQ12_LIMBS];
    for (int k = 0; k < FQ12_LIMBS; k++) f[k] = src[k];
    ell(f, src + FQ12_LIMBS, src + FQ12_LIMBS + 3 * FQ2_LIMBS,
        src + FQ12_LIMBS + 3 * FQ2_LIMBS + LIMBS);
    u64 *dst = out + (unsigned long long)i * FQ12_LIMBS;
    for (int k = 0; k < FQ12_LIMBS; k++) dst[k] = f[k];
}

extern "C" __global__ void g2_double_step_probe(const u64 *__restrict__ in,
                                               const u64 *__restrict__ consts,
                                               u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *src = in + (unsigned long long)i * 3 * FQ2_LIMBS;
    u64 *dst = out + (unsigned long long)i * 12 * FQ2_LIMBS;
    u64 rx[FQ2_LIMBS], ry[FQ2_LIMBS], rz[FQ2_LIMBS], coeff[3 * FQ2_LIMBS];

    fq2_copy(src, rx);
    fq2_copy(src + FQ2_LIMBS, ry);
    fq2_copy(src + 2 * FQ2_LIMBS, rz);
    g2_double_step(rx, ry, rz, consts, coeff);
    fq2_copy(rx, dst);
    fq2_copy(ry, dst + FQ2_LIMBS);
    fq2_copy(rz, dst + 2 * FQ2_LIMBS);
    for (int k = 0; k < 3 * FQ2_LIMBS; k++) dst[3 * FQ2_LIMBS + k] = coeff[k];

    fq2_copy(src, rx);
    fq2_copy(src + FQ2_LIMBS, ry);
    fq2_copy(src + 2 * FQ2_LIMBS, rz);
    mw_g2_double_step(rx, ry, rz, consts, coeff);
    fq2_copy(rx, dst + 6 * FQ2_LIMBS);
    fq2_copy(ry, dst + 7 * FQ2_LIMBS);
    fq2_copy(rz, dst + 8 * FQ2_LIMBS);
    for (int k = 0; k < 3 * FQ2_LIMBS; k++) dst[9 * FQ2_LIMBS + k] = coeff[k];
}

extern "C" __global__ void g2_add_step_probe(const u64 *__restrict__ in,
                                            const u64 *__restrict__ consts,
                                            u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *src = in + (unsigned long long)i * 5 * FQ2_LIMBS;
    const u64 *qx = src + 3 * FQ2_LIMBS;
    const u64 *qy = src + 4 * FQ2_LIMBS;
    u64 *dst = out + (unsigned long long)i * 12 * FQ2_LIMBS;
    u64 rx[FQ2_LIMBS], ry[FQ2_LIMBS], rz[FQ2_LIMBS], coeff[3 * FQ2_LIMBS];

    fq2_copy(src, rx);
    fq2_copy(src + FQ2_LIMBS, ry);
    fq2_copy(src + 2 * FQ2_LIMBS, rz);
    g2_add_step(rx, ry, rz, qx, qy, coeff);
    fq2_copy(rx, dst);
    fq2_copy(ry, dst + FQ2_LIMBS);
    fq2_copy(rz, dst + 2 * FQ2_LIMBS);
    for (int k = 0; k < 3 * FQ2_LIMBS; k++) dst[3 * FQ2_LIMBS + k] = coeff[k];

    fq2_copy(src, rx);
    fq2_copy(src + FQ2_LIMBS, ry);
    fq2_copy(src + 2 * FQ2_LIMBS, rz);
    mw_g2_add_step(rx, ry, rz, qx, qy, coeff);
    fq2_copy(rx, dst + 6 * FQ2_LIMBS);
    fq2_copy(ry, dst + 7 * FQ2_LIMBS);
    fq2_copy(rz, dst + 8 * FQ2_LIMBS);
    for (int k = 0; k < 3 * FQ2_LIMBS; k++) dst[9 * FQ2_LIMBS + k] = coeff[k];
}

extern "C" __global__ void g2_mul_by_char_probe(const u64 *__restrict__ in,
                                               const u64 *__restrict__ consts,
                                               u64 *__restrict__ out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 *src = in + (unsigned long long)i * 2 * FQ2_LIMBS;
    u64 *dst = out + (unsigned long long)i * 4 * FQ2_LIMBS;
    g2_mul_by_char(src, src + FQ2_LIMBS, consts, dst, dst + FQ2_LIMBS);
    mw_g2_mul_by_char(src, src + FQ2_LIMBS, consts, dst + 2 * FQ2_LIMBS, dst + 3 * FQ2_LIMBS);
}
