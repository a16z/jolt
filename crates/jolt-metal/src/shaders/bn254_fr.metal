#ifndef JOLT_METAL_BN254_FR_H
#define JOLT_METAL_BN254_FR_H

// BN254 scalar field (Fr) arithmetic for Metal compute shaders.
//
// Representation: 8 × 32-bit limbs, little-endian (limb[0] = LSB).
// Values are in Montgomery form: a_mont = a · R mod r, where R = 2^256.
//
// Multiplication uses CIOS (Coarsely Integrated Operand Scanning) with
// a 32-bit digit width. Each of the 8 outer iterations performs one
// multiply-accumulate pass and one reduction pass across 8 limbs.
//
// All outputs of mul/sqr are fully reduced to [0, r). This ensures
// add/sub inputs are in [0, r), where a single conditional correction
// (one modulus addition/subtraction) is always sufficient.

#include "common.metal"

// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
constant uint FR_MODULUS[8] = {
    0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u,
};

// -r^{-1} mod 2^{32}
constant uint FR_INV32 = 0xefffffffu;

// R^2 mod r  (used for to_mont conversion)
constant uint FR_R2[8] = {
    0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u,
    0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u,
};

// R mod r  (Montgomery representation of 1)
constant uint FR_ONE[8] = {
    0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u,
    0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u,
};

struct Fr {
    uint limbs[8];
};

inline bool fr_gte(Fr a, thread const uint (&m)[8]) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > m[i]) return true;
        if (a.limbs[i] < m[i]) return false;
    }
    return true; // equal
}

inline Fr fr_select(bool cond, Fr if_true, Fr if_false) {
    Fr r;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = cond ? if_true.limbs[i] : if_false.limbs[i];
    }
    return r;
}

inline Fr fr_reduce(Fr a) {
    Fr reduced;
    uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(a.limbs[i], FR_MODULUS[i], borrow);
        reduced.limbs[i] = r.x;
        borrow = r.y;
    }
    // If borrow=1, subtraction underflowed → a < modulus → keep a.
    return fr_select(borrow == 0, reduced, a);
}

inline Fr fr_add(Fr a, Fr b) {
    Fr result;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = adc(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = r.x;
        carry = r.y;
    }
    // Subtract modulus; if the subtraction doesn't underflow, use it.
    Fr reduced;
    uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(result.limbs[i], FR_MODULUS[i], borrow);
        reduced.limbs[i] = r.x;
        borrow = r.y;
    }
    // Use reduced when: carry from add >= borrow from sub.
    return fr_select(carry >= borrow, reduced, result);
}

inline Fr fr_sub(Fr a, Fr b) {
    Fr result;
    uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = sbb(a.limbs[i], b.limbs[i], borrow);
        result.limbs[i] = r.x;
        borrow = r.y;
    }
    // If underflow, add modulus back.
    Fr corrected;
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint addend = borrow ? FR_MODULUS[i] : 0u;
        uint2 r = adc(result.limbs[i], addend, carry);
        corrected.limbs[i] = r.x;
        carry = r.y;
    }
    return corrected;
}

inline Fr fr_neg(Fr a) {
    Fr zero;
    for (int i = 0; i < 8; i++) zero.limbs[i] = 0;
    return fr_sub(zero, a);
}

// CIOS iteration: T += a[] * b_j, then reduce+shift.
// After 8 iterations, T[0..7] holds the reduced product.
#define FR_CIOS_ROUND(T, a, b_j, T9_out)                                    \
{                                                                            \
    ulong carry = 0;                                                         \
    for (int i = 0; i < 8; i++) {                                            \
        ulong sum = ulong(T[i]) + ulong(a.limbs[i]) * ulong(b_j) + carry;   \
        T[i] = uint(sum);                                                    \
        carry = sum >> 32;                                                   \
    }                                                                        \
    ulong t8c = ulong(T[8]) + carry;                                         \
    T[8] = uint(t8c);                                                        \
    T9_out = uint(t8c >> 32);                                                \
    uint m = T[0] * FR_INV32;                                                \
    carry = (ulong(T[0]) + ulong(m) * ulong(FR_MODULUS[0])) >> 32;          \
    for (int i = 1; i < 8; i++) {                                            \
        ulong sum2 = ulong(T[i]) + ulong(m) * ulong(FR_MODULUS[i]) + carry; \
        T[i - 1] = uint(sum2);                                              \
        carry = sum2 >> 32;                                                  \
    }                                                                        \
    ulong fs = ulong(T[8]) + carry;                                          \
    T[7] = uint(fs);                                                         \
    T[8] = T9_out + uint(fs >> 32);                                          \
}

inline Fr fr_mul(Fr a, Fr b) {
    uint T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint T9;
    FR_CIOS_ROUND(T, a, b.limbs[0], T9);
    FR_CIOS_ROUND(T, a, b.limbs[1], T9);
    FR_CIOS_ROUND(T, a, b.limbs[2], T9);
    FR_CIOS_ROUND(T, a, b.limbs[3], T9);
    FR_CIOS_ROUND(T, a, b.limbs[4], T9);
    FR_CIOS_ROUND(T, a, b.limbs[5], T9);
    FR_CIOS_ROUND(T, a, b.limbs[6], T9);
    FR_CIOS_ROUND(T, a, b.limbs[7], T9);
    Fr result;
    for (int i = 0; i < 8; i++) result.limbs[i] = T[i];
    return fr_reduce(result);
}

inline Fr fr_sqr(Fr a) {
    return fr_mul(a, a);
}

inline Fr fr_zero() {
    Fr z;
    for (int i = 0; i < 8; i++) z.limbs[i] = 0;
    return z;
}

inline Fr fr_one() {
    Fr o;
    for (int i = 0; i < 8; i++) o.limbs[i] = FR_ONE[i];
    return o;
}

// Convert a raw integer (in standard form) to Montgomery form: a_mont = a * R^2 * R^{-1} = a * R
inline Fr fr_to_mont(Fr a) {
    Fr r2;
    for (int i = 0; i < 8; i++) r2.limbs[i] = FR_R2[i];
    return fr_mul(a, r2);
}

// Convert from Montgomery form to standard form: a * 1 * R^{-1} = a / R
inline Fr fr_from_mont(Fr a) {
    Fr one_std;
    one_std.limbs[0] = 1;
    for (int i = 1; i < 8; i++) one_std.limbs[i] = 0;
    return fr_reduce(fr_mul(a, one_std));
}

// Convert a u64 scalar to Fr in Montgomery form.
// The u64 is placed in limbs[0..1] (little-endian 32-bit), then multiplied by R^2.
inline Fr fr_from_u64(ulong val) {
    Fr a;
    a.limbs[0] = uint(val);
    a.limbs[1] = uint(val >> 32);
    for (int i = 2; i < 8; i++) a.limbs[i] = 0;
    return fr_to_mont(a);
}

inline bool fr_eq(Fr a, Fr b) {
    Fr ar = fr_reduce(a);
    Fr br = fr_reduce(b);
    bool eq = true;
    for (int i = 0; i < 8; i++) {
        eq = eq && (ar.limbs[i] == br.limbs[i]);
    }
    return eq;
}

#endif // JOLT_METAL_BN254_FR_H
