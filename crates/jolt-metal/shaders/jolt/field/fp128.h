// jolt::Fp128<C>: the prime field p = 2^128 - C for odd C < 2^32, bit-exact
// with jolt_field::solinas::Fp128<P>.
//
// The algorithms and their names follow crates/jolt-field/src/solinas/fp128.rs:
// a schoolbook widening product, a first fold of the high 128 bits through
// 2^128 = C (mod p), then fold2_canonicalize. Each bound the code relies on is
// argued next to it, restated for 32-bit words.
//
// Representation: four little-endian 32-bit words holding the canonical value
// in [0, p). On little-endian Apple GPUs these bytes equal the CPU's
// [u64; 2] limbs, so buffers move between host and device without conversion.
// Every operation takes canonical inputs and returns a canonical output.

#ifndef JOLT_FIELD_FP128_H
#define JOLT_FIELD_FP128_H

#include <metal_stdlib>

namespace jolt {
namespace fp128_detail {

// Little-endian 32-bit words of an unreduced intermediate.
template <int N>
using Words = metal::array<uint, N>;

// a + b over 128 bits. `carry` receives the carry out of bit 127.
inline uint4 add128(uint4 a, uint4 b, thread bool& carry) {
    uint4 sum;
    ulong t = 0;
    for (int i = 0; i < 4; i++) {
        t = ulong(a[i]) + b[i] + (t >> 32);
        sum[i] = uint(t);
    }
    carry = (t >> 32) != 0;
    return sum;
}

// a - b over 128 bits. `borrow` receives the borrow out of bit 127.
//
// Each step computes a[i] - b[i] - borrow in [-2^32, 2^32 - 1] modulo 2^64,
// so bit 63 of the difference is set exactly when the step borrows.
inline uint4 sub128(uint4 a, uint4 b, thread bool& borrow) {
    uint4 difference;
    ulong t = 0;
    for (int i = 0; i < 4; i++) {
        t = ulong(a[i]) - b[i] - (t >> 63);
        difference[i] = uint(t);
    }
    borrow = (t >> 63) != 0;
    return difference;
}

// The schoolbook product of a 128-bit `a` and a 32M-bit `b`, with no
// reduction.
//
// Row i adds a[i] * b into out[i .. i + M]. Each step computes
// a[i] * b[j] + out[i + j] + carry <= (2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1,
// so the ulong never overflows. out[i + M] has not been written by an earlier
// row, and the full product is below 2^(128 + 32M), so 4 + M words hold it
// exactly.
template <int M>
inline Words<4 + M> mul_words(uint4 a, Words<M> b) {
    Words<4 + M> out;
    for (int k = 0; k < 4 + M; k++) {
        out[k] = 0;
    }
    for (int i = 0; i < 4; i++) {
        ulong t = 0;
        for (int j = 0; j < M; j++) {
            t = ulong(a[i]) * b[j] + out[i + j] + (t >> 32);
            out[i + j] = uint(t);
        }
        out[i + M] = uint(t >> 32);
    }
    return out;
}

// 128 x 128 -> 256-bit product, no reduction: 16 word multiplies.
inline Words<8> mul_wide(uint4 a, uint4 b) {
    return mul_words<4>(a, Words<4>{b.x, b.y, b.z, b.w});
}

// 128 x 64 -> 192-bit product, no reduction: 8 word multiplies.
inline Words<6> mul_wide_u64(uint4 a, ulong b) {
    return mul_words<2>(a, Words<2>{uint(b), uint(b >> 32)});
}

// a^2 as a 256-bit value, no reduction: 10 word multiplies.
//
// The six cross products a[i] a[j] (i < j) are summed row by row, written
// out: each step computes a[i] a[j] + c + carry with the mul_words row bound,
// and the sum is below a^2 / 2 < 2^255, so doubling it by a one-bit shift
// loses no bit. Word 0 holds no cross product (i + j >= 1). Each square
// a[i]^2 is then added at word 2i in one step,
// a[i]^2 + out[2i] + carry <= (2^32 - 1)^2 + (2^32 - 1) + 1 < 2^64, and its
// carry is propagated through word 2i + 1, leaving a carry of at most 1. The
// final carry is zero because a^2 < 2^256.
//
// Written out rather than as a triangular loop: on an Apple M4 Max the loop
// form ran at half this throughput and slower than a * a (see the Fp128
// benchmarks in specs/jolt-metal-field.md).
inline Words<8> sqr_wide(uint4 a) {
    ulong t = ulong(a.x) * a.y;
    uint c1 = uint(t);
    t = ulong(a.x) * a.z + (t >> 32);
    uint c2 = uint(t);
    t = ulong(a.x) * a.w + (t >> 32);
    uint c3 = uint(t);
    uint c4 = uint(t >> 32);
    t = ulong(a.y) * a.z + c3;
    c3 = uint(t);
    t = ulong(a.y) * a.w + c4 + (t >> 32);
    c4 = uint(t);
    uint c5 = uint(t >> 32);
    t = ulong(a.z) * a.w + c5;
    c5 = uint(t);
    uint c6 = uint(t >> 32);
    Words<8> out{0u,
                 c1 << 1,
                 (c2 << 1) | (c1 >> 31),
                 (c3 << 1) | (c2 >> 31),
                 (c4 << 1) | (c3 >> 31),
                 (c5 << 1) | (c4 >> 31),
                 (c6 << 1) | (c5 >> 31),
                 c6 >> 31};
    ulong carry = 0;
    for (int i = 0; i < 4; i++) {
        t = ulong(a[i]) * a[i] + out[2 * i] + carry;
        out[2 * i] = uint(t);
        t = ulong(out[2 * i + 1]) + (t >> 32);
        out[2 * i + 1] = uint(t);
        carry = t >> 32;
    }
    return out;
}

// Reduces t + t2 * 2^128 into [0, p) for any 128-bit t and any 64-bit t2.
//
// C * t2 < 2^32 * 2^64 = 2^96 fits three words: the low product is below
// 2^64, and the high one plus its carry is at most (2^32 - 1)^2 + 2^32 - 1.
// Let v = t + C * t2 < 2^128 + 2^96, so the 128-bit sum s wraps at most once
// (`overflow`).
//
// - No overflow (v < 2^128): s = v, and s + C carries out of bit 127 exactly
//   when s >= p, in which case the wrapped s + C equals s - p.
// - Overflow (v >= 2^128): s = v - 2^128 < C * t2, and the residue is
//   s + C, since 2^128 = C (mod p). s + C < C (t2 + 1) <= C * 2^64 < 2^96 < p,
//   so it is canonical and the add does not carry.
//
// Either way the result is s + C when overflow or that add carries, and s
// otherwise.
template <uint C>
inline uint4 fold2_canonicalize(uint4 t, ulong t2) {
    ulong low = ulong(uint(t2)) * C;
    ulong high = ulong(uint(t2 >> 32)) * C + (low >> 32);
    bool overflow;
    uint4 s = add128(t, uint4(uint(low), uint(high), uint(high >> 32), 0u), overflow);
    bool carry;
    uint4 r = add128(s, uint4(C, 0u, 0u, 0u), carry);
    return (overflow || carry) ? r : s;
}

// Reduces any 256-bit value into [0, p).
//
// First fold: t = lo + C * hi. Each step computes
// hi[i] * C + lo[i] + carry <= (2^32 - 1)^2 + 2 (2^32 - 1) = 2^64 - 1.
// t <= (C + 1)(2^128 - 1), so the carry out t2 is at most C; then
// fold2_canonicalize.
template <uint C>
inline uint4 reduce_4(Words<8> w) {
    uint4 t;
    ulong u = 0;
    for (int i = 0; i < 4; i++) {
        u = ulong(w[4 + i]) * C + w[i] + (u >> 32);
        t[i] = uint(u);
    }
    return fold2_canonicalize<C>(t, u >> 32);
}

// With a, b < p, a + b < 2^129 wraps at most once. Without the wrap, s + C
// carries exactly when s >= p, and the wrapped s + C is s - p. With it,
// s = a + b - 2^128 and the residue is s + C < p.
template <uint C>
inline uint4 add(uint4 a, uint4 b) {
    bool overflow;
    uint4 s = add128(a, b, overflow);
    bool carry;
    uint4 r = add128(s, uint4(C, 0u, 0u, 0u), carry);
    return (overflow || carry) ? r : s;
}

// On borrow, a - b + 2^128 is in [2^128 - p, 2^128), and adding p
// (subtracting C modulo 2^128) gives a - b + p in [0, p).
template <uint C>
inline uint4 sub(uint4 a, uint4 b) {
    bool borrow;
    uint4 d = sub128(a, b, borrow);
    bool unused;
    uint4 corrected = sub128(d, uint4(C, 0u, 0u, 0u), unused);
    return borrow ? corrected : d;
}

// Every u64 is canonical, since p > 2^127.
inline uint4 from_u64(ulong v) {
    return uint4(uint(v), uint(v >> 32), 0u, 0u);
}

// The magnitude is computed in unsigned arithmetic, so v = LONG_MIN gives
// 2^63 without overflow.
template <uint C>
inline uint4 from_i64(long v) {
    bool negative = v < 0;
    uint4 magnitude = from_u64(negative ? 0ul - ulong(v) : ulong(v));
    uint4 negated = sub<C>(uint4(0u), magnitude);
    return negative ? negated : magnitude;
}

// The 192-bit product [lo, mid, hi] splits as t = [lo, mid] and t2 = hi,
// which fold2_canonicalize reduces for any 64-bit t2.
template <uint C>
inline uint4 mul_u64(uint4 a, ulong s) {
    Words<6> w = mul_wide_u64(a, s);
    return fold2_canonicalize<C>(uint4(w[0], w[1], w[2], w[3]), ulong(w[4]) | (ulong(w[5]) << 32));
}

template <uint C>
inline uint4 mul_i64(uint4 a, long s) {
    bool negative = s < 0;
    uint4 product = mul_u64<C>(a, negative ? 0ul - ulong(s) : ulong(s));
    uint4 negated = sub<C>(uint4(0u), product);
    return negative ? negated : product;
}

} // namespace fp128_detail

// An element of the prime field 2^128 - C. The same C as the CPU type's
// Fp128::<P>::C; the Rust MslType impl spells this type from P, so the two
// cannot drift.
//
// Operators and functions are hidden friends: generic kernels call them
// unqualified (a * b, square(a), mul_u64(a, s)) and argument-dependent lookup
// finds them for any field type.
template <uint C>
struct Fp128 {
    static_assert(C % 2u == 1u, "Fp128<C> needs an odd C: p = 2^128 - C must be odd");

    uint4 limb;

    static Fp128 zero() { return Fp128{uint4(0u)}; }
    static Fp128 one() { return Fp128{uint4(1u, 0u, 0u, 0u)}; }
    static Fp128 from_u64(ulong v) { return Fp128{fp128_detail::from_u64(v)}; }
    static Fp128 from_i64(long v) { return Fp128{fp128_detail::from_i64<C>(v)}; }

    friend Fp128 operator+(Fp128 a, Fp128 b) {
        return Fp128{fp128_detail::add<C>(a.limb, b.limb)};
    }
    friend Fp128 operator-(Fp128 a, Fp128 b) {
        return Fp128{fp128_detail::sub<C>(a.limb, b.limb)};
    }
    friend Fp128 operator-(Fp128 a) {
        return Fp128{fp128_detail::sub<C>(uint4(0u), a.limb)};
    }
    friend Fp128 operator*(Fp128 a, Fp128 b) {
        return Fp128{fp128_detail::reduce_4<C>(fp128_detail::mul_wide(a.limb, b.limb))};
    }
    friend Fp128 square(Fp128 a) {
        return Fp128{fp128_detail::reduce_4<C>(fp128_detail::sqr_wide(a.limb))};
    }
    friend Fp128 mul_u64(Fp128 a, ulong s) {
        return Fp128{fp128_detail::mul_u64<C>(a.limb, s)};
    }
    friend Fp128 mul_i64(Fp128 a, long s) {
        return Fp128{fp128_detail::mul_i64<C>(a.limb, s)};
    }
};

} // namespace jolt

#endif // JOLT_FIELD_FP128_H
