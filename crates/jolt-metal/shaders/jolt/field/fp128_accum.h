// Deferred-reduction accumulators for jolt::Fp128<C>, the counterparts of
// crates/jolt-field/src/fp128_accumulators.rs.
//
// - Fp128Accumulator<C>: a 288-bit unsigned integer in nine carried 32-bit
//   words. fmadd adds the 256-bit product from mul_wide with one carry chain.
// - Fp128SignedAccumulator<C>: a 224-bit two's-complement integer in seven
//   words. A signed scalar product is added or subtracted in place.
//
// Both reduce once, at the end, to the canonical element.
//
// The layouts differ from the CPU's u128 slots and pos/neg pair. On an
// Apple M4 Max, a paired A/B (specs/jolt-metal-field.md, step 3) found the
// carried 288-bit form within 3% of eight ulong slots and of eight uncarried
// column sums, and it needs 9 words instead of 16. Signed accumulation in
// place was 1.16x faster than the pos/neg pair, and it needs half the words.

#ifndef JOLT_FIELD_FP128_ACCUM_H
#define JOLT_FIELD_FP128_ACCUM_H

#include <metal_stdlib>

namespace jolt {
namespace fp128_detail {

// Reduces lo + 2^128 hi into [0, p), for the 128-bit lo = w[0..4) and the
// 160-bit hi = w[4..9).
//
// First fold: t = lo + C (hi mod 2^128), one carried step per word as in
// reduce_4. The carry out plus C w[8] is at most
// (2^32 - 1) + (2^32 - 1)^2 < 2^64, so it is a valid t2 for
// fold2_canonicalize.
template <uint C>
inline uint4 reduce_288(Words<9> w) {
    uint4 t;
    ulong u = 0;
    for (int i = 0; i < 4; i++) {
        u = ulong(w[4 + i]) * C + w[i] + (u >> 32);
        t[i] = uint(u);
    }
    return fold2_canonicalize<C>(t, ulong(w[8]) * C + (u >> 32));
}

// Magnitude of a signed scalar in unsigned arithmetic, so LONG_MIN gives
// 2^63 without overflow.
inline ulong magnitude(long s) {
    return s < 0 ? 0ul - ulong(s) : ulong(s);
}

} // namespace fp128_detail

// Sum of field elements and products of two field elements, as an unsigned
// integer below 2^288 in nine little-endian words.
//
// CAPACITY: every term is at most (p - 1)^2 < 2^256: a product is at most
// (p - 1)^2 and an added element at most p - 1. So 2^32 terms sum below
// 2^288. The bound is tight: 2^32 + 1 products (p - 1)^2 exceed 2^288.
template <uint C>
struct Fp128Accumulator {
    using F = Fp128<C>;
    static constexpr constant ulong CAPACITY = 1ul << 32;

    fp128_detail::Words<9> w;

    static Fp128Accumulator zero() {
        Fp128Accumulator z;
        for (int k = 0; k < 9; k++) {
            z.w[k] = 0;
        }
        return z;
    }

    // w += p. The sum stays below 2^288 within CAPACITY, so the carry into
    // w[8] never wraps.
    void add_words(fp128_detail::Words<8> p) {
        ulong t = 0;
        for (int k = 0; k < 8; k++) {
            t = ulong(w[k]) + p[k] + (t >> 32);
            w[k] = uint(t);
        }
        w[8] += uint(t >> 32);
    }

    void add(F v) {
        add_words(fp128_detail::Words<8>{v.limb.x, v.limb.y, v.limb.z, v.limb.w, 0u, 0u, 0u, 0u});
    }

    void fmadd(F a, F b) { add_words(fp128_detail::mul_wide(a.limb, b.limb)); }

    void merge(Fp128Accumulator other) {
        ulong t = 0;
        for (int k = 0; k < 9; k++) {
            t = ulong(w[k]) + other.w[k] + (t >> 32);
            w[k] = uint(t);
        }
    }

    F reduce() { return F{fp128_detail::reduce_288<C>(w)}; }

    friend Fp128Accumulator simd_shuffle_xor(Fp128Accumulator a, ushort mask) {
        Fp128Accumulator r;
        for (int k = 0; k < 9; k++) {
            r.w[k] = metal::simd_shuffle_xor(a.w[k], mask);
        }
        return r;
    }
};

// Sum of field elements times signed 64-bit scalars, as a two's-complement
// integer in [-2^223, 2^223) in seven little-endian words.
//
// CAPACITY: every term has magnitude below p 2^64 < 2^192: a scalar product
// is at most (p - 1)(2^64 - 1) and an added element at most p - 1. So 2^31
// terms of either sign stay in (-2^223, 2^223), and bit 223 is the sign. The
// bound is tight: 2^31 + 1 terms (p - 1)(2^64 - 1) exceed 2^223.
template <uint C>
struct Fp128SignedAccumulator {
    using F = Fp128<C>;
    static constexpr constant ulong CAPACITY = 1ul << 31;

    fp128_detail::Words<7> w;

    static Fp128SignedAccumulator zero() {
        Fp128SignedAccumulator z;
        for (int k = 0; k < 7; k++) {
            z.w[k] = 0;
        }
        return z;
    }

    // w += p, or w -= p when `negative`, modulo 2^224: subtracting adds the
    // complement of p, extended to seven words, plus one.
    void add_signed(fp128_detail::Words<6> p, bool negative) {
        uint mask = negative ? ~0u : 0u;
        ulong t = negative ? 1 : 0;
        for (int k = 0; k < 6; k++) {
            t = ulong(w[k]) + (p[k] ^ mask) + t;
            w[k] = uint(t);
            t >>= 32;
        }
        w[6] += mask + uint(t);
    }

    void add(F v) {
        add_signed(fp128_detail::Words<6>{v.limb.x, v.limb.y, v.limb.z, v.limb.w, 0u, 0u}, false);
    }

    void fmadd_u64(F a, ulong s) { add_signed(fp128_detail::mul_wide_u64(a.limb, s), false); }

    void fmadd_i64(F a, long s) {
        add_signed(fp128_detail::mul_wide_u64(a.limb, fp128_detail::magnitude(s)), s < 0);
    }

    void fmadd_signed_u64(F a, ulong magnitude, bool is_positive) {
        add_signed(fp128_detail::mul_wide_u64(a.limb, magnitude), !is_positive);
    }

    void merge(Fp128SignedAccumulator other) {
        ulong t = 0;
        for (int k = 0; k < 7; k++) {
            t = ulong(w[k]) + other.w[k] + (t >> 32);
            w[k] = uint(t);
        }
    }

    // Reduces the magnitude, below 2^223 + 1, and negates the result when
    // bit 223 is set.
    F reduce() {
        bool negative = (w[6] >> 31) != 0;
        uint mask = negative ? ~0u : 0u;
        fp128_detail::Words<9> m;
        ulong t = negative ? 1 : 0;
        for (int k = 0; k < 7; k++) {
            t = ulong(w[k] ^ mask) + t;
            m[k] = uint(t);
            t >>= 32;
        }
        m[7] = 0;
        m[8] = 0;
        uint4 r = fp128_detail::reduce_288<C>(m);
        uint4 negated = fp128_detail::sub<C>(uint4(0u), r);
        return F{negative ? negated : r};
    }

    friend Fp128SignedAccumulator simd_shuffle_xor(Fp128SignedAccumulator a, ushort mask) {
        Fp128SignedAccumulator r;
        for (int k = 0; k < 7; k++) {
            r.w[k] = metal::simd_shuffle_xor(a.w[k], mask);
        }
        return r;
    }
};

template <uint C>
struct WithAccumulator<Fp128<C>> {
    using Accumulator = Fp128Accumulator<C>;
    using SmallScalarAccumulator = Fp128SignedAccumulator<C>;
};

} // namespace jolt

#endif // JOLT_FIELD_FP128_ACCUM_H
