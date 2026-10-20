#define FQ6_LIMBS (3 * FQ2_LIMBS)

static_assert(FQ2_LIMBS == 8, "Fq2 is two Fq limbs wide");

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
