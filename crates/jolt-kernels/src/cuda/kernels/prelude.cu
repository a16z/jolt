typedef unsigned long long u64;
typedef unsigned __int128 u128;

#define LIMBS 4
#define BLOCK 256

__device__ __constant__ u64 MODULUS[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
__device__ __constant__ u64 INV = 0xc2e1f593efffffffULL;

__device__ __forceinline__ u64 mac(u64 a, u64 b, u64 c, u64 *carry) {
    u128 t = (u128)a + (u128)b * (u128)c + (u128)(*carry);
    *carry = (u64)(t >> 64);
    return (u64)t;
}

__device__ __forceinline__ u64 adc(u64 a, u64 b, u64 *carry) {
    u64 lo, hi;
    asm("{\n\t"
        ".reg .u64 l, h;\n\t"
        "add.cc.u64 l, %2, %3;\n\t"
        "addc.u64 h, 0, 0;\n\t"
        "add.cc.u64 l, l, %4;\n\t"
        "addc.u64 h, h, 0;\n\t"
        "mov.u64 %0, l;\n\t"
        "mov.u64 %1, h;\n\t"
        "}"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b), "l"(*carry));
    *carry = hi;
    return lo;
}

__device__ __forceinline__ u64 sbb(u64 a, u64 b, u64 *borrow) {
    u64 lo, hi;
    asm("{\n\t"
        ".reg .u64 l, h;\n\t"
        "sub.cc.u64 l, %2, %3;\n\t"
        "subc.u64 h, 0, 0;\n\t"
        "sub.cc.u64 l, l, %4;\n\t"
        "subc.u64 h, h, 0;\n\t"
        "mov.u64 %0, l;\n\t"
        "sub.u64 %1, 0, h;\n\t"
        "}"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b), "l"(*borrow));
    *borrow = hi;
    return lo;
}

// Keep carries in CC across all four limbs instead of materializing a
// 128-bit intermediate and a separate carry at every limb boundary.
__device__ __forceinline__ void add4(const u64 *a, const u64 *b, u64 *out) {
    asm("{\n\t"
        ".reg .u64 t0, t1, t2, t3;\n\t"
        "add.cc.u64 t0, %4, %8;\n\t"
        "addc.cc.u64 t1, %5, %9;\n\t"
        "addc.cc.u64 t2, %6, %10;\n\t"
        "addc.u64 t3, %7, %11;\n\t"
        "mov.u64 %0, t0;\n\t"
        "mov.u64 %1, t1;\n\t"
        "mov.u64 %2, t2;\n\t"
        "mov.u64 %3, t3;\n\t"
        "}"
        : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
}

__device__ __forceinline__ u64 sub4(const u64 *a, const u64 *b, u64 *out) {
    u64 borrow;
    asm("{\n\t"
        ".reg .u64 t0, t1, t2, t3, mask;\n\t"
        "sub.cc.u64 t0, %5, %9;\n\t"
        "subc.cc.u64 t1, %6, %10;\n\t"
        "subc.cc.u64 t2, %7, %11;\n\t"
        "subc.cc.u64 t3, %8, %12;\n\t"
        "subc.u64 mask, 0, 0;\n\t"
        "mov.u64 %0, t0;\n\t"
        "mov.u64 %1, t1;\n\t"
        "mov.u64 %2, t2;\n\t"
        "mov.u64 %3, t3;\n\t"
        "mov.u64 %4, mask;\n\t"
        "}"
        : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3]), "=l"(borrow)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
    return borrow;
}

__device__ __forceinline__ void field_add(const u64 *a, const u64 *b,
                                           const u64 *modulus, u64 *out) {
    u64 sum[LIMBS], reduced[LIMBS];
    // Both BN254 moduli are below 2^254, so a canonical sum cannot overflow.
    add4(a, b, sum);
    u64 borrow = sub4(sum, modulus, reduced);
    for (int i = 0; i < LIMBS; i++) out[i] = borrow ? sum[i] : reduced[i];
}

__device__ __forceinline__ void field_sub(const u64 *a, const u64 *b,
                                           const u64 *modulus, u64 *out) {
    u64 difference[LIMBS], correction[LIMBS];
    u64 borrow = sub4(a, b, difference);
    for (int i = 0; i < LIMBS; i++) correction[i] = modulus[i] & borrow;
    add4(difference, correction, out);
}

// Any 256-bit word is below 8p. Subtracting 4p, 2p, then p conditionally
// canonicalizes it without a Montgomery multiplication.
__device__ __forceinline__ void fr_reduce_256(const u64 *value, u64 *out) {
    const u64 four_p[LIMBS] = {
        0x0f87d64fc0000004ULL, 0xa0cfa121e6e5c245ULL,
        0xe14116da06056174ULL, 0xc19139cb84c680a6ULL
    };
    const u64 two_p[LIMBS] = {
        0x87c3eb27e0000002ULL, 0x5067d090f372e122ULL,
        0x70a08b6d0302b0baULL, 0x60c89ce5c2634053ULL
    };
    u64 t[LIMBS], reduced[LIMBS];
    u64 borrow = sub4(value, four_p, reduced);
    for (int i = 0; i < LIMBS; i++) t[i] = borrow ? value[i] : reduced[i];
    borrow = sub4(t, two_p, reduced);
    for (int i = 0; i < LIMBS; i++) t[i] = borrow ? t[i] : reduced[i];
    borrow = sub4(t, MODULUS, reduced);
    for (int i = 0; i < LIMBS; i++) out[i] = borrow ? t[i] : reduced[i];
}

// CIOS Montgomery multiplication. For canonical inputs, each iteration ends
// below a + modulus < 2^255; the two unshifted sums fit in five limbs.
__device__ __forceinline__ void mont_accumulate(u64 *t, const u64 *a, u64 b) {
    asm("{\n\t"
        ".reg .u64 l0,l1,l2,l3,h0,h1,h2,h3,r0,r1,r2,r3,r4;\n\t"
        "mul.lo.u64 l0, %10, %14;\n\t"
        "mul.lo.u64 l1, %11, %14;\n\t"
        "mul.lo.u64 l2, %12, %14;\n\t"
        "mul.lo.u64 l3, %13, %14;\n\t"
        "mul.hi.u64 h0, %10, %14;\n\t"
        "mul.hi.u64 h1, %11, %14;\n\t"
        "mul.hi.u64 h2, %12, %14;\n\t"
        "mul.hi.u64 h3, %13, %14;\n\t"
        "add.cc.u64 r0, %5, l0;\n\t"
        "addc.cc.u64 r1, %6, l1;\n\t"
        "addc.cc.u64 r2, %7, l2;\n\t"
        "addc.cc.u64 r3, %8, l3;\n\t"
        "addc.u64 r4, %9, 0;\n\t"
        "add.cc.u64 r1, r1, h0;\n\t"
        "addc.cc.u64 r2, r2, h1;\n\t"
        "addc.cc.u64 r3, r3, h2;\n\t"
        "addc.u64 r4, r4, h3;\n\t"
        "mov.u64 %0, r0;\n\t"
        "mov.u64 %1, r1;\n\t"
        "mov.u64 %2, r2;\n\t"
        "mov.u64 %3, r3;\n\t"
        "mov.u64 %4, r4;\n\t"
        "}"
        : "=l"(t[0]), "=l"(t[1]), "=l"(t[2]), "=l"(t[3]), "=l"(t[4])
        : "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(t[4]),
          "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b));
}

__device__ __forceinline__ void field_mont_mul(const u64 *a, const u64 *b,
                                                const u64 *modulus, u64 inv, u64 *out) {
    u64 t[5] = {0, 0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < LIMBS; i++) {
        mont_accumulate(t, a, b[i]);
        mont_accumulate(t, modulus, t[0] * inv);
#pragma unroll
        for (int j = 0; j < LIMBS; j++) t[j] = t[j + 1];
        t[4] = 0;
    }
    u64 reduced[LIMBS];
    u64 borrow = sub4(t, modulus, reduced);
#pragma unroll
    for (int i = 0; i < LIMBS; i++) out[i] = borrow ? t[i] : reduced[i];
}

__device__ __forceinline__ void load4(const u64 *__restrict__ p, u64 *r) {
    ulonglong4 v = *reinterpret_cast<const ulonglong4 *>(p);
    r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
}

__device__ __forceinline__ void store4(u64 *p, const u64 *r) {
    ulonglong4 v;
    v.x = r[0]; v.y = r[1]; v.z = r[2]; v.w = r[3];
    *reinterpret_cast<ulonglong4 *>(p) = v;
}

__device__ void fr_add(const u64 *a, const u64 *b, u64 *out) {
    field_add(a, b, MODULUS, out);
}

__device__ void fr_sub(const u64 *a, const u64 *b, u64 *out) {
    field_sub(a, b, MODULUS, out);
}

__device__ void fr_mul(const u64 *a, const u64 *b, u64 *out) {
    field_mont_mul(a, b, MODULUS, INV, out);
}
