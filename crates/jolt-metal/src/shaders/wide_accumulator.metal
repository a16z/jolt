#ifndef JOLT_METAL_WIDE_ACCUMULATOR_H
#define JOLT_METAL_WIDE_ACCUMULATOR_H

// Wide-integer accumulator for deferred-reduction fused multiply-add.
//
// In sumcheck inner loops, thousands of `acc += a * b` steps execute
// before the final field element is needed. Rather than reducing mod r
// after every multiply (~40ns), we accumulate products as wide integers
// and reduce once at the end.
//
// The accumulator stores 18 × 32-bit limbs (576 bits). An Fr × Fr
// product is at most 512 bits (16 limbs). With 2 extra limbs of
// headroom, we can accumulate 2^32 products without overflow — far
// more than any sumcheck round requires.
//
// The accumulator layout matches the Rust-side WideAccumulator (9×64-bit
// limbs = 18×32-bit limbs).

#include "bn254_fr.metal"

// 18 limbs = 9 × 64-bit limbs = 576 bits
constant int ACC_LIMBS = 18;

struct WideAcc {
    uint limbs[ACC_LIMBS];
};

inline WideAcc acc_zero() {
    WideAcc a;
    for (int i = 0; i < ACC_LIMBS; i++) a.limbs[i] = 0;
    return a;
}

// Add a field element to the accumulator without multiplication.
//
// For summation: acc += a_mont. The accumulated wide value is Σ a_i_mont.
// After acc_reduce this gives Σ a_i (standard form); use fr_to_mont to
// convert back to Montgomery representation.
inline void acc_add_fr(thread WideAcc &acc, Fr a) {
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint2 r = adc(acc.limbs[i], a.limbs[i], carry);
        acc.limbs[i] = r.x;
        carry = r.y;
    }
    for (int i = 8; i < ACC_LIMBS && carry; i++) {
        uint2 r = adc(acc.limbs[i], 0u, carry);
        acc.limbs[i] = r.x;
        carry = r.y;
    }
}

// Fused multiply-add: acc += a * b (schoolbook, 8×8 → 16 limbs, accumulated into 18).
// Pure 32-bit: uses mul + mulhi instead of ulong emulation.
inline void acc_fmadd(thread WideAcc &acc, Fr a, Fr b) {
    for (int j = 0; j < 8; j++) {
        uint carry = 0;
        for (int i = 0; i < 8; i++) {
            int idx = i + j;
            uint p_lo = a.limbs[i] * b.limbs[j];
            uint p_hi = mulhi(a.limbs[i], b.limbs[j]);
            uint s1 = acc.limbs[idx] + p_lo;
            uint c1 = uint(s1 < acc.limbs[idx]);
            uint s2 = s1 + carry;
            uint c2 = uint(s2 < s1);
            acc.limbs[idx] = s2;
            carry = p_hi + c1 + c2;
        }
        int k = j + 8;
        while (carry != 0 && k < ACC_LIMBS) {
            uint old = acc.limbs[k];
            uint s = old + carry;
            carry = uint(s < old);
            acc.limbs[k] = s;
            k++;
        }
    }
}

// Merge two accumulators: dst += src.
inline void acc_merge(thread WideAcc &dst, WideAcc src) {
    uint carry = 0;
    for (int i = 0; i < ACC_LIMBS; i++) {
        uint2 r = adc(dst.limbs[i], src.limbs[i], carry);
        dst.limbs[i] = r.x;
        carry = r.y;
    }
}

// Threadgroup overload for reduction kernels using shared memory.
inline void acc_merge(threadgroup WideAcc &dst, threadgroup WideAcc &src) {
    uint carry = 0;
    for (int i = 0; i < ACC_LIMBS; i++) {
        uint2 r = adc(dst.limbs[i], src.limbs[i], carry);
        dst.limbs[i] = r.x;
        carry = r.y;
    }
}

// Reduce accumulator to Fr via Montgomery reduction.
//
// The accumulator holds Σ (a_i_mont × b_i_mont), which is Σ (a_i·R × b_i·R).
// Montgomery reduction divides by R, giving Σ (a_i × b_i) · R = the
// Montgomery form of the desired sum.
//
// After 8 CIOS rounds the result lives in limbs[8..17]. For a single
// multiply the value fits in 8 limbs, but after many accumulated products
// it can exceed 256 bits (e.g. 256 products → ~260 bits). We subtract
// the modulus in a loop to bring it into [0, r).
inline Fr acc_reduce(WideAcc acc) {
    for (int round = 0; round < 8; round++) {
        uint m = acc.limbs[round] * FR_INV32;
        // First limb: acc[round] + m * MODULUS[0] ≡ 0 mod 2^32 (CIOS property)
        uint m_hi = mulhi(m, (uint)FR_MODULUS[0]);
        uint s0 = acc.limbs[round] + m * (uint)FR_MODULUS[0];
        uint carry = m_hi + uint(s0 < acc.limbs[round]);
        for (int i = 1; i < 8; i++) {
            uint p_lo = m * (uint)FR_MODULUS[i];
            uint p_hi = mulhi(m, (uint)FR_MODULUS[i]);
            uint s1 = acc.limbs[round + i] + p_lo;
            uint c1 = uint(s1 < acc.limbs[round + i]);
            uint s2 = s1 + carry;
            uint c2 = uint(s2 < s1);
            acc.limbs[round + i] = s2;
            carry = p_hi + c1 + c2;
        }
        int k = round + 8;
        while (carry != 0 && k < ACC_LIMBS) {
            uint old = acc.limbs[k];
            uint s = old + carry;
            carry = uint(s < old);
            acc.limbs[k] = s;
            k++;
        }
    }

    // limbs[8..17] holds the unreduced result. Subtract r until < r.
    // With N accumulated products the value is at most ~N·r, so this
    // loop runs at most N times (typically 1-2 for practical sizes).
    while (true) {
        bool gte = (acc.limbs[16] != 0) || (acc.limbs[17] != 0);
        if (!gte) {
            gte = true;
            for (int i = 7; i >= 0; i--) {
                if (acc.limbs[i + 8] > FR_MODULUS[i]) break;
                if (acc.limbs[i + 8] < FR_MODULUS[i]) { gte = false; break; }
            }
        }
        if (!gte) break;

        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint2 d = sbb(acc.limbs[i + 8], FR_MODULUS[i], borrow);
            acc.limbs[i + 8] = d.x;
            borrow = d.y;
        }
        // Propagate borrow through overflow limbs
        for (int i = 16; i < ACC_LIMBS; i++) {
            uint2 d = sbb(acc.limbs[i], 0, borrow);
            acc.limbs[i] = d.x;
            borrow = d.y;
        }
    }

    Fr result;
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = acc.limbs[i + 8];
    }
    return result;
}

// Copy a threadgroup accumulator to thread-local memory and reduce.
inline Fr acc_reduce_tg(threadgroup WideAcc &tg_acc) {
    WideAcc acc;
    for (int i = 0; i < ACC_LIMBS; i++) acc.limbs[i] = tg_acc.limbs[i];
    return acc_reduce(acc);
}

#endif // JOLT_METAL_WIDE_ACCUMULATOR_H
