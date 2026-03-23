#ifndef JOLT_METAL_RNS_H
#define JOLT_METAL_RNS_H

#include <metal_stdlib>
using namespace metal;

// RNS-Montgomery arithmetic for BN254 Fr on Metal GPU.
//
// Dual-basis approach (Bajard/VROOM):
// - Base B (primary): 9 pseudo-Mersenne primes, product M > r
// - Base B' (secondary): 9 pseudo-Mersenne primes, product M' > r
// - Elements stored as ã = a·M mod r, decomposed in both bases
//
// All operations are independent 32-bit per-residue computations.

constant constexpr uint BASIS_SIZE = 9;
constant constexpr uint RNS_NUM_PRIMES = 18;

// Primary basis B: primes m_i = 2^31 - c_i
constant constexpr uint B_PRIMES[BASIS_SIZE] = {
    2147483647, 2147483629, 2147483587, 2147483579, 2147483563,
    2147483549, 2147483543, 2147483497, 2147483489,
};

constant constexpr uint B_C[BASIS_SIZE] = {
    1, 19, 61, 69, 85, 99, 105, 151, 159,
};

// Secondary basis B': primes m'_j = 2^31 - c'_j
constant constexpr uint BP_PRIMES[BASIS_SIZE] = {
    2147483477, 2147483423, 2147483399, 2147483353, 2147483323,
    2147483269, 2147483249, 2147483237, 2147483179,
};

constant constexpr uint BP_C[BASIS_SIZE] = {
    171, 225, 249, 295, 325, 379, 399, 411, 469,
};

// Combined arrays for looping over all 18 primes
constant constexpr uint RNS_PRIMES[RNS_NUM_PRIMES] = {
    2147483647, 2147483629, 2147483587, 2147483579, 2147483563,
    2147483549, 2147483543, 2147483497, 2147483489,
    2147483477, 2147483423, 2147483399, 2147483353, 2147483323,
    2147483269, 2147483249, 2147483237, 2147483179,
};

constant constexpr uint RNS_C[RNS_NUM_PRIMES] = {
    1, 19, 61, 69, 85, 99, 105, 151, 159,
    171, 225, 249, 295, 325, 379, 399, 411, 469,
};

// ---------------------------------------------------------------------------
// Pseudo-Mersenne reduction
// ---------------------------------------------------------------------------

// x mod (2^31 - c). Input: x < 2^63. Output: [0, p).
inline uint rns_reduce(ulong x, uint p, uint c) {
    ulong r = (x & 0x7FFFFFFFul) + (ulong)c * (x >> 31);
    uint r2 = (uint)(r & 0x7FFFFFFFul) + c * (uint)(r >> 31);
    return r2 >= p ? r2 - p : r2;
}

// (a + b) mod p. Inputs in [0, p).
inline uint rns_add(uint a, uint b, uint p) {
    uint s = a + b;
    return s >= p ? s - p : s;
}

// (a - b) mod p. Inputs in [0, p).
inline uint rns_sub(uint a, uint b, uint p) {
    return a >= b ? a - b : p - (b - a);
}

// (a * b) mod p via pseudo-Mersenne. Inputs in [0, p).
inline uint rns_mul(uint a, uint b, uint p, uint c) {
    return rns_reduce((ulong)a * (ulong)b, p, c);
}

// Accumulate a*b into ulong. Caller reduces periodically.
inline void rns_fmadd(thread ulong &acc, uint a, uint b) {
    acc += (ulong)a * (ulong)b;
}

// Reduce ulong accumulator mod p.
inline uint rns_acc_reduce(ulong acc, uint p, uint c) {
    return rns_reduce(acc, p, c);
}

// Simdgroup sum of u32.
inline uint simd_sum_u32(uint val) {
    val += simd_shuffle_xor(val, 1);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 16);
    return val;
}

#endif // JOLT_METAL_RNS_H
