// BN254 scalar-field arithmetic in Montgomery form, 32-bit limbs, little-endian.
//
// This file is compiled at runtime with a generated preamble prepended (see
// `metal::field::constants_preamble`) that defines:
//   FR_LIMBS            — number of 32-bit limbs (8 for BN254)
//   FR_MOD[FR_LIMBS]    — the modulus p, LE u32 limbs
//   FR_INV32            — -p^{-1} mod 2^32 (Montgomery reduction constant)
//   JK_TG_SIZE          — compute threadgroup size shared with the Rust side
//   JK_MAX_EVAL_POINTS  — bind_eval's eval-point capacity
//
// All functions take and return canonical residues (< p). Multiplication is
// CIOS (coarsely integrated operand scanning, Koç–Acar): interleaved
// multiply and Montgomery reduction over an (FR_LIMBS+2)-word accumulator,
// with 64-bit intermediates (uint mul → ulong) since a*b + acc + carry for
// 32-bit operands fits exactly in 64 bits. Conditional corrections are
// branchless mask selects so warps stay uniform on random data.

#include <metal_stdlib>
using namespace metal;

struct Fr256 {
    uint v[FR_LIMBS];
};

inline Fr256 fr_zero() {
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = 0u;
    }
    return r;
}

inline Fr256 fr_load(device const uint* p, uint idx) {
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = p[idx * FR_LIMBS + i];
    }
    return r;
}

inline void fr_store(device uint* p, uint idx, Fr256 x) {
    for (uint i = 0; i < FR_LIMBS; i++) {
        p[idx * FR_LIMBS + i] = x.v[i];
    }
}

inline Fr256 fr_load_const(constant const uint* p, uint idx) {
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = p[idx * FR_LIMBS + i];
    }
    return r;
}

// Threadgroup scratch uses a limb-major layout (limb l of lane i at
// [l * JK_TG_SIZE + i]) so lanes touch consecutive words — bank-conflict-free.
inline Fr256 fr_load_tg(threadgroup const uint* p, uint lane) {
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = p[i * JK_TG_SIZE + lane];
    }
    return r;
}

inline void fr_store_tg(threadgroup uint* p, uint lane, Fr256 x) {
    for (uint i = 0; i < FR_LIMBS; i++) {
        p[i * JK_TG_SIZE + lane] = x.v[i];
    }
}

// r = a - b + (borrow ? p : 0); canonical for canonical inputs.
inline Fr256 fr_sub(Fr256 a, Fr256 b) {
    Fr256 r;
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - borrow;
        r.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    // Add p back when the subtraction wrapped, selected by mask (no branch).
    uint mask = (uint)(0u - (uint)borrow);
    ulong carry = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong s = (ulong)r.v[i] + (ulong)(FR_MOD[i] & mask) + carry;
        r.v[i] = (uint)s;
        carry = s >> 32;
    }
    return r;
}

// r = (a + b) mod p via wide add + branchless trial subtraction.
inline Fr256 fr_add(Fr256 a, Fr256 b) {
    Fr256 sum;
    ulong carry = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + carry;
        sum.v[i] = (uint)s;
        carry = s >> 32;
    }
    Fr256 diff;
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)sum.v[i] - (ulong)FR_MOD[i] - borrow;
        diff.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    // sum >= p iff the wide add carried out or the trial subtract didn't borrow.
    bool take_diff = (carry != 0) || (borrow == 0);
    uint mask = take_diff ? 0xffffffffu : 0u;
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = (diff.v[i] & mask) | (sum.v[i] & ~mask);
    }
    return r;
}

// Montgomery product abR^{-1} mod p, CIOS. Canonical output for canonical
// inputs: T = (ab + mp)/R < p^2/R + p < 2p, so one trial subtraction
// suffices and the accumulator's top word stays clear.
inline Fr256 fr_mont_mul(Fr256 a, Fr256 b) {
    uint t[FR_LIMBS + 2];
    for (uint i = 0; i < FR_LIMBS + 2; i++) {
        t[i] = 0u;
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong carry = 0;
        for (uint j = 0; j < FR_LIMBS; j++) {
            ulong cur = (ulong)t[j] + (ulong)a.v[i] * (ulong)b.v[j] + carry;
            t[j] = (uint)cur;
            carry = cur >> 32;
        }
        ulong cur = (ulong)t[FR_LIMBS] + carry;
        t[FR_LIMBS] = (uint)cur;
        t[FR_LIMBS + 1] = (uint)(cur >> 32);

        uint m = t[0] * FR_INV32;
        cur = (ulong)t[0] + (ulong)m * (ulong)FR_MOD[0];
        carry = cur >> 32;
        for (uint j = 1; j < FR_LIMBS; j++) {
            cur = (ulong)t[j] + (ulong)m * (ulong)FR_MOD[j] + carry;
            t[j - 1] = (uint)cur;
            carry = cur >> 32;
        }
        cur = (ulong)t[FR_LIMBS] + carry;
        t[FR_LIMBS - 1] = (uint)cur;
        t[FR_LIMBS] = t[FR_LIMBS + 1] + (uint)(cur >> 32);
        t[FR_LIMBS + 1] = 0u;
    }
    Fr256 sum;
    for (uint i = 0; i < FR_LIMBS; i++) {
        sum.v[i] = t[i];
    }
    Fr256 diff;
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)sum.v[i] - (ulong)FR_MOD[i] - borrow;
        diff.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    bool take_diff = (t[FR_LIMBS] != 0u) || (borrow == 0);
    uint mask = take_diff ? 0xffffffffu : 0u;
    Fr256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = (diff.v[i] & mask) | (sum.v[i] & ~mask);
    }
    return r;
}
