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
FR_FUNC_ATTR void acc_add_fr(thread WideAcc &acc, Fr a) {
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

// 4×4 schoolbook multiply: out[0..7] += a[0..3] * b[0..3].
// `out` must be zero-initialized before the first call.
inline void schoolbook_4x4(thread uint (&out)[8], const thread uint *a, const thread uint *b) {
    for (int j = 0; j < 4; j++) {
        uint carry = 0;
        for (int i = 0; i < 4; i++) {
            int idx = i + j;
            uint p_lo = a[i] * b[j];
            uint p_hi = mulhi(a[i], b[j]);
            uint s1 = out[idx] + p_lo;
            uint c1 = uint(s1 < out[idx]);
            uint s2 = s1 + carry;
            uint c2 = uint(s2 < s1);
            out[idx] = s2;
            carry = p_hi + c1 + c2;
        }
        // Propagate carry through remaining limbs
        for (int k = j + 4; carry != 0 && k < 8; k++) {
            uint old = out[k];
            out[k] = old + carry;
            carry = uint(out[k] < old);
        }
    }
}

// 4×4 schoolbook multiply into 9-limb output (for sums that may have a carry bit).
inline void schoolbook_4x4_wide(thread uint (&out)[9], const thread uint *a, const thread uint *b) {
    for (int j = 0; j < 4; j++) {
        uint carry = 0;
        for (int i = 0; i < 4; i++) {
            int idx = i + j;
            uint p_lo = a[i] * b[j];
            uint p_hi = mulhi(a[i], b[j]);
            uint s1 = out[idx] + p_lo;
            uint c1 = uint(s1 < out[idx]);
            uint s2 = s1 + carry;
            uint c2 = uint(s2 < s1);
            out[idx] = s2;
            carry = p_hi + c1 + c2;
        }
        for (int k = j + 4; carry != 0 && k < 9; k++) {
            uint old = out[k];
            out[k] = old + carry;
            carry = uint(out[k] < old);
        }
    }
}

// Add N limbs from `src` into WideAcc at `offset`, with carry propagation.
inline void acc_add_limbs(thread WideAcc &acc, const thread uint *src, int n, int offset) {
    uint carry = 0;
    for (int i = 0; i < n; i++) {
        uint2 r = adc(acc.limbs[offset + i], src[i], carry);
        acc.limbs[offset + i] = r.x;
        carry = r.y;
    }
    for (int i = offset + n; carry != 0 && i < ACC_LIMBS; i++) {
        uint2 r = adc(acc.limbs[i], 0u, carry);
        acc.limbs[i] = r.x;
        carry = r.y;
    }
}

// Fused multiply-add: acc += a * b.
//
// Uses Karatsuba on 4-limb halves: 3 × (4×4) = 48 mul+mulhi instead of
// schoolbook's 64. The cross-term P1 = (aL+aH)*(bL+bH) - aL*bL - aH*bH
// is always non-negative (it equals aH*bL + aL*bH).
//
// Operation sequence minimizes peak register pressure:
// 1. aS = aL+aH, bS = bL+bH (10 regs, freed after step 2)
// 2. Pm = aS * bS (9 regs)
// 3. P0 = aL * bL (8 regs), Pm -= P0, accumulate P0, free P0
// 4. P2 = aH * bH (8 regs), Pm -= P2, accumulate P2, free P2
// 5. Accumulate Pm (the cross-term P1)
FR_FUNC_ATTR void acc_fmadd(thread WideAcc &acc, Fr a, Fr b) {
    // Step 1: aS = aL + aH, bS = bL + bH (4 limbs + carry bit each)
    uint aS[4], bS[4];
    uint aC, bC; // carry bits (0 or 1)
    {
        uint carry = 0;
        for (int i = 0; i < 4; i++) {
            uint2 r = adc(a.limbs[i], a.limbs[i + 4], carry);
            aS[i] = r.x;
            carry = r.y;
        }
        aC = carry;
    }
    {
        uint carry = 0;
        for (int i = 0; i < 4; i++) {
            uint2 r = adc(b.limbs[i], b.limbs[i + 4], carry);
            bS[i] = r.x;
            carry = r.y;
        }
        bC = carry;
    }

    // Step 2: Pm = aS[0..3] * bS[0..3] (4x4 into 9 limbs)
    // Then handle carry bits branchlessly.
    uint Pm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    schoolbook_4x4_wide(Pm, aS, bS);

    // Pm += aC * bS at offset 4 (branchless: mask is 0xFFFFFFFF or 0)
    {
        uint mask = uint(-int(aC));
        uint carry = 0;
        for (int i = 0; i < 4; i++) {
            uint2 r = adc(Pm[i + 4], bS[i] & mask, carry);
            Pm[i + 4] = r.x;
            carry = r.y;
        }
        Pm[8] += carry;
    }
    // Pm += bC * aS at offset 4
    {
        uint mask = uint(-int(bC));
        uint carry = 0;
        for (int i = 0; i < 4; i++) {
            uint2 r = adc(Pm[i + 4], aS[i] & mask, carry);
            Pm[i + 4] = r.x;
            carry = r.y;
        }
        Pm[8] += carry;
    }
    // Pm[8] += aC & bC (branchless)
    Pm[8] += aC & bC;
    // aS, bS, aC, bC are now dead.

    // Step 3: P0 = aL * bL, subtract from Pm, accumulate at offset 0
    {
        uint P0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        schoolbook_4x4(P0, a.limbs, b.limbs);

        // Pm -= P0 (safe: Pm = P0 + P2 + cross >= P0)
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint2 d = sbb(Pm[i], P0[i], borrow);
            Pm[i] = d.x;
            borrow = d.y;
        }
        Pm[8] -= borrow;

        acc_add_limbs(acc, P0, 8, 0);
    } // P0 freed

    // Step 4: P2 = aH * bH, subtract from Pm, accumulate at offset 8
    {
        uint P2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        schoolbook_4x4(P2, &a.limbs[4], &b.limbs[4]);

        // Pm -= P2 (safe: Pm now holds cross + P2 >= P2)
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint2 d = sbb(Pm[i], P2[i], borrow);
            Pm[i] = d.x;
            borrow = d.y;
        }
        Pm[8] -= borrow;

        acc_add_limbs(acc, P2, 8, 8);
    } // P2 freed

    // Step 5: Pm now holds P1 = aH*bL + aL*bH. Accumulate at offset 4.
    acc_add_limbs(acc, Pm, 9, 4);
}

// Merge two accumulators: dst += src.
FR_FUNC_ATTR void acc_merge(thread WideAcc &dst, WideAcc src) {
    uint carry = 0;
    for (int i = 0; i < ACC_LIMBS; i++) {
        uint2 r = adc(dst.limbs[i], src.limbs[i], carry);
        dst.limbs[i] = r.x;
        carry = r.y;
    }
}

// Threadgroup overload for reduction kernels using shared memory.
FR_FUNC_ATTR void acc_merge(threadgroup WideAcc &dst, threadgroup WideAcc &src) {
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
FR_FUNC_ATTR Fr acc_reduce(WideAcc acc) {
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
FR_FUNC_ATTR Fr acc_reduce_tg(threadgroup WideAcc &tg_acc) {
    WideAcc acc;
    for (int i = 0; i < ACC_LIMBS; i++) acc.limbs[i] = tg_acc.limbs[i];
    return acc_reduce(acc);
}

#endif // JOLT_METAL_WIDE_ACCUMULATOR_H
