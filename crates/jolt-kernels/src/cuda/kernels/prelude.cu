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
    u128 t = (u128)a + (u128)b + (u128)(*carry);
    *carry = (u64)(t >> 64);
    return (u64)t;
}

__device__ __forceinline__ u64 sbb(u64 a, u64 b, u64 *borrow) {
    u128 t = ((u128)1 << 64) + (u128)a - (u128)b - (u128)(*borrow);
    *borrow = (t >> 64) == 0 ? 1 : 0;
    return (u64)t;
}

__device__ __forceinline__ int geq_modulus(const u64 *a) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != MODULUS[i]) return a[i] > MODULUS[i];
    }
    return 1;
}

__device__ __forceinline__ void sub_modulus(u64 *a) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) a[i] = sbb(a[i], MODULUS[i], &borrow);
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
    u64 carry = 0;
    for (int i = 0; i < 4; i++) out[i] = adc(a[i], b[i], &carry);
    if (carry != 0 || geq_modulus(out)) sub_modulus(out);
}

__device__ void fr_sub(const u64 *a, const u64 *b, u64 *out) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) out[i] = sbb(a[i], b[i], &borrow);
    if (borrow != 0) {
        u64 carry = 0;
        for (int i = 0; i < 4; i++) out[i] = adc(out[i], MODULUS[i], &carry);
    }
}

__device__ void fr_mul(const u64 *a, const u64 *b, u64 *out) {
    u64 t[6];
    for (int i = 0; i < 6; i++) t[i] = 0;

    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) t[j] = mac(t[j], a[j], b[i], &carry);
        u64 c = 0;
        t[4] = adc(t[4], carry, &c);
        t[5] = adc(t[5], 0, &c);

        u64 m = t[0] * INV;
        u64 c2 = 0;
        mac(t[0], m, MODULUS[0], &c2);
        for (int j = 1; j < 4; j++) t[j - 1] = mac(t[j], m, MODULUS[j], &c2);
        u64 c3 = 0;
        t[3] = adc(t[4], c2, &c3);
        t[4] = adc(t[5], 0, &c3);
        t[5] = 0;
    }

    for (int i = 0; i < 4; i++) out[i] = t[i];
    if (t[4] != 0 || geq_modulus(out)) sub_modulus(out);
}
