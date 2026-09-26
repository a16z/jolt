// jolt::Fp64<C>: the prime field p = 2^64 - C for odd C < 2^32, bit-exact
// with jolt_field::solinas::Fp64<P> for 64-bit P.
//
// The algorithms and their names follow crates/jolt-field/src/solinas/word.rs:
// a widening product, a first fold of the high 64 bits through
// 2^64 = C (mod p), then fold2_canonicalize. Each bound the code relies on is
// argued next to it. Moduli below 2^63 (jolt_field's sub-word Fp64) fold at
// a different bit and are not covered.
//
// Representation: one ulong holding the canonical value in [0, p), the bytes
// of the CPU's u64. Every operation takes canonical inputs and returns a
// canonical output.

#ifndef JOLT_FIELD_FP64_H
#define JOLT_FIELD_FP64_H

#include <metal_stdlib>

namespace jolt {
namespace fp64_detail {

// A 128-bit unreduced intermediate.
struct Wide {
    ulong lo;
    ulong hi;
};

// a * b over 128 bits, no reduction: four 32 x 32-bit multiplies, summed
// row by row as mul_words in fp128.h does. Each step computes
// x * y + c + d <= (2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1 for 32-bit x, y, c
// and d, so no step carries out of 64 bits, and the last one is the high
// word.
inline Wide mul_wide(ulong a, ulong b) {
    uint a0 = uint(a), a1 = uint(a >> 32);
    uint b0 = uint(b), b1 = uint(b >> 32);
    ulong t = ulong(a0) * b0;
    uint w0 = uint(t);
    t = ulong(a0) * b1 + (t >> 32);
    uint w1 = uint(t);
    uint w2 = uint(t >> 32);
    t = ulong(a1) * b0 + w1;
    w1 = uint(t);
    t = ulong(a1) * b1 + w2 + (t >> 32);
    return Wide{(ulong(w1) << 32) | w0, t};
}

// a^2 over 128 bits, no reduction: three 32 x 32-bit multiplies.
//
// 2 a0 a1 * 2^32 = mid * 2^33 splits into mid << 33 (the low 64 bits) and
// mid >> 31 (the rest). lo carries at most once, and a^2 < 2^128.
inline Wide sqr_wide(ulong a) {
    uint a0 = uint(a), a1 = uint(a >> 32);
    ulong p00 = ulong(a0) * a0;
    ulong mid = ulong(a0) * a1;
    ulong p11 = ulong(a1) * a1;
    ulong lo = p00 + (mid << 33);
    ulong hi = p11 + (mid >> 31) + (lo < p00 ? 1ul : 0ul);
    return Wide{lo, hi};
}

// Reduces t + t2 * 2^64 into [0, p) for any 64-bit t and any t2 <= C.
//
// C * t2 <= C^2 < 2^64. Let v = t + C * t2 < 2^64 + C^2, so the 64-bit sum s
// wraps at most once (`overflow`).
//
// - No overflow (v < 2^64): s = v, and s + C carries out of bit 63 exactly
//   when s >= p, in which case the wrapped s + C equals s - p.
// - Overflow (v >= 2^64): s = v - 2^64 < C * t2 <= C^2, and the residue is
//   s + C, since 2^64 = C (mod p). s + C < C (C + 1) < p because
//   C < 2^32, so it is canonical and the add does not carry.
//
// Either way the result is s + C when overflow or that add carries, and s
// otherwise.
template <uint C>
inline ulong fold2_canonicalize(ulong t, uint t2) {
    ulong s = t + ulong(t2) * C;
    bool overflow = s < t;
    ulong r = s + C;
    bool carry = r < s;
    return (overflow || carry) ? r : s;
}

// Reduces any 128-bit value into [0, p).
//
// First fold: t = lo + C * hi, in two 32-bit steps. Each step computes
// hi[i] * C + lo[i] + carry <= (2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1.
// t <= (C + 1)(2^64 - 1), so the carry out t2 is at most C; then
// fold2_canonicalize.
template <uint C>
inline ulong reduce_product(Wide x) {
    ulong u = ulong(uint(x.hi)) * C + uint(x.lo);
    uint t0 = uint(u);
    u = ulong(uint(x.hi >> 32)) * C + uint(x.lo >> 32) + (u >> 32);
    return fold2_canonicalize<C>((u << 32) | t0, uint(u >> 32));
}

// With a, b < p, a + b < 2^65 wraps at most once. Without the wrap, s + C
// carries exactly when s >= p, and the wrapped s + C is s - p. With it,
// s = a + b - 2^64 and the residue is s + C < p.
template <uint C>
inline ulong add(ulong a, ulong b) {
    ulong s = a + b;
    bool overflow = s < a;
    ulong r = s + C;
    bool carry = r < s;
    return (overflow || carry) ? r : s;
}

// On borrow, a - b + 2^64 is in [2^64 - p + 1, 2^64), and adding p
// (subtracting C modulo 2^64) gives a - b + p in [1, p).
template <uint C>
inline ulong sub(ulong a, ulong b) {
    ulong d = a - b;
    return a < b ? d - C : d;
}

// v < 2^64 < 2p, so one conditional subtraction of p suffices: v + C carries
// exactly when v >= p, and then the wrapped v + C is v - p.
template <uint C>
inline ulong from_u64(ulong v) {
    ulong r = v + C;
    return r < v ? r : v;
}

// The magnitude is computed in unsigned arithmetic, so v = LONG_MIN gives
// 2^63 without overflow.
template <uint C>
inline ulong from_i64(long v) {
    bool negative = v < 0;
    ulong magnitude = from_u64<C>(negative ? 0ul - ulong(v) : ulong(v));
    ulong negated = sub<C>(0ul, magnitude);
    return negative ? negated : magnitude;
}

// a * s < 2^128 for any 64-bit s, which reduce_product accepts.
template <uint C>
inline ulong mul_u64(ulong a, ulong s) {
    return reduce_product<C>(mul_wide(a, s));
}

template <uint C>
inline ulong mul_i64(ulong a, long s) {
    bool negative = s < 0;
    ulong product = mul_u64<C>(a, negative ? 0ul - ulong(s) : ulong(s));
    ulong negated = sub<C>(0ul, product);
    return negative ? negated : product;
}

} // namespace fp64_detail

// An element of the prime field 2^64 - C. The same C as the CPU type's
// Fp64::<P>::C; the Rust MslType impl spells this type from P, so the two
// cannot drift.
//
// Operators and functions are hidden friends, as for Fp128.
template <uint C>
struct Fp64 {
    static_assert(C % 2u == 1u, "Fp64<C> needs an odd C: p = 2^64 - C must be odd");

    ulong word;

    static Fp64 zero() { return Fp64{0ul}; }
    static Fp64 one() { return Fp64{1ul}; }
    static Fp64 from_u64(ulong v) { return Fp64{fp64_detail::from_u64<C>(v)}; }
    static Fp64 from_i64(long v) { return Fp64{fp64_detail::from_i64<C>(v)}; }

    friend Fp64 operator+(Fp64 a, Fp64 b) {
        return Fp64{fp64_detail::add<C>(a.word, b.word)};
    }
    friend Fp64 operator-(Fp64 a, Fp64 b) {
        return Fp64{fp64_detail::sub<C>(a.word, b.word)};
    }
    friend Fp64 operator-(Fp64 a) {
        return Fp64{fp64_detail::sub<C>(0ul, a.word)};
    }
    friend Fp64 operator*(Fp64 a, Fp64 b) {
        return Fp64{fp64_detail::reduce_product<C>(fp64_detail::mul_wide(a.word, b.word))};
    }
    friend Fp64 square(Fp64 a) {
        return Fp64{fp64_detail::reduce_product<C>(fp64_detail::sqr_wide(a.word))};
    }
    friend Fp64 mul_u64(Fp64 a, ulong s) {
        return Fp64{fp64_detail::mul_u64<C>(a.word, s)};
    }
    friend Fp64 mul_i64(Fp64 a, long s) {
        return Fp64{fp64_detail::mul_i64<C>(a.word, s)};
    }
};

} // namespace jolt

#endif // JOLT_FIELD_FP64_H
