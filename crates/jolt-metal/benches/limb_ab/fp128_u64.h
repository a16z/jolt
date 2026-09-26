// jolt::Fp128<C>, 64-bit limb variant for the limb-layout A/B.
//
// Same storage (uint4, four little-endian 32-bit words) and the same
// algorithm as the 32-bit-limb header; the arithmetic runs on two 64-bit
// words, with MSL's native 64-bit `*` and `mulhi` for products, so the
// compiler chooses how to lower them. The bounds are those of the 32-bit
// header restated for 64-bit words.

#ifndef JOLT_FIELD_FP128_H
#define JOLT_FIELD_FP128_H

#include <metal_stdlib>

namespace jolt {
namespace fp128_detail {

template <int N>
using Words = metal::array<ulong, N>;

inline ulong2 wide(uint4 x) { return as_type<ulong2>(x); }
inline uint4 narrow(ulong2 x) { return as_type<uint4>(x); }

// a * b + acc + carry as a 128-bit value: (2^64 - 1)^2 + 2 (2^64 - 1)
// = 2^128 - 1, so it never overflows. Returns the low word; `hi` receives
// the high word.
inline ulong mac(ulong a, ulong b, ulong acc, ulong carry, thread ulong& hi) {
    ulong lo = a * b;
    ulong h = metal::mulhi(a, b);
    lo += acc;
    h += lo < acc ? 1ul : 0ul;
    lo += carry;
    h += lo < carry ? 1ul : 0ul;
    hi = h;
    return lo;
}

inline ulong2 add128(ulong2 a, ulong2 b, thread bool& carry) {
    ulong lo = a.x + b.x;
    ulong c = lo < a.x ? 1ul : 0ul;
    ulong hi = a.y + b.y;
    bool c1 = hi < a.y;
    hi += c;
    carry = c1 || hi < c;
    return ulong2(lo, hi);
}

inline ulong2 sub128(ulong2 a, ulong2 b, thread bool& borrow) {
    ulong lo = a.x - b.x;
    ulong bw = a.x < b.x ? 1ul : 0ul;
    ulong hi = a.y - b.y;
    bool b1 = a.y < b.y;
    borrow = b1 || hi < bw;
    hi -= bw;
    return ulong2(lo, hi);
}

template <int M>
inline Words<2 + M> mul_words(ulong2 a, Words<M> b) {
    Words<2 + M> out;
    for (int k = 0; k < 2 + M; k++) {
        out[k] = 0;
    }
    for (int i = 0; i < 2; i++) {
        ulong carry = 0;
        for (int j = 0; j < M; j++) {
            ulong hi;
            out[i + j] = mac(a[i], b[j], out[i + j], carry, hi);
            carry = hi;
        }
        out[i + M] = carry;
    }
    return out;
}

inline Words<4> mul_wide(ulong2 a, ulong2 b) {
    return mul_words<2>(a, Words<2>{b.x, b.y});
}

inline Words<3> mul_wide_u64(ulong2 a, ulong b) {
    return mul_words<1>(a, Words<1>{b});
}

inline Words<4> sqr_wide(ulong2 a) {
    ulong cross_hi;
    ulong cross_lo = mac(a.x, a.y, 0, 0, cross_hi);
    Words<4> out{0, cross_lo << 1, (cross_hi << 1) | (cross_lo >> 63), cross_hi >> 63};
    ulong hi;
    out[0] = mac(a.x, a.x, 0, 0, hi);
    ulong t = out[1] + hi;
    ulong carry = t < hi ? 1ul : 0ul;
    out[1] = t;
    ulong hi1;
    out[2] = mac(a.y, a.y, out[2], carry, hi1);
    out[3] += hi1;
    return out;
}

template <uint C>
inline ulong2 fold2_canonicalize(ulong2 t, ulong t2) {
    ulong2 ct2 = ulong2(t2 * C, metal::mulhi(t2, ulong(C)));
    bool overflow;
    ulong2 s = add128(t, ct2, overflow);
    bool carry;
    ulong2 r = add128(s, ulong2(C, 0ul), carry);
    return (overflow || carry) ? r : s;
}

template <uint C>
inline ulong2 reduce_4(Words<4> w) {
    ulong hi0;
    ulong t0 = mac(w[2], C, w[0], 0, hi0);
    ulong hi1;
    ulong t1 = mac(w[3], C, w[1], hi0, hi1);
    return fold2_canonicalize<C>(ulong2(t0, t1), hi1);
}

template <uint C>
inline ulong2 add(ulong2 a, ulong2 b) {
    bool overflow;
    ulong2 s = add128(a, b, overflow);
    bool carry;
    ulong2 r = add128(s, ulong2(C, 0ul), carry);
    return (overflow || carry) ? r : s;
}

template <uint C>
inline ulong2 sub(ulong2 a, ulong2 b) {
    bool borrow;
    ulong2 d = sub128(a, b, borrow);
    bool unused;
    ulong2 corrected = sub128(d, ulong2(C, 0ul), unused);
    return borrow ? corrected : d;
}

template <uint C>
inline ulong2 from_i64(long v) {
    bool negative = v < 0;
    ulong2 magnitude = ulong2(negative ? 0ul - ulong(v) : ulong(v), 0ul);
    ulong2 negated = sub<C>(ulong2(0ul), magnitude);
    return negative ? negated : magnitude;
}

template <uint C>
inline ulong2 mul_u64(ulong2 a, ulong s) {
    Words<3> w = mul_wide_u64(a, s);
    return fold2_canonicalize<C>(ulong2(w[0], w[1]), w[2]);
}

template <uint C>
inline ulong2 mul_i64(ulong2 a, long s) {
    bool negative = s < 0;
    ulong2 product = mul_u64<C>(a, negative ? 0ul - ulong(s) : ulong(s));
    ulong2 negated = sub<C>(ulong2(0ul), product);
    return negative ? negated : product;
}

} // namespace fp128_detail

template <uint C>
struct Fp128 {
    static_assert(C % 2u == 1u, "Fp128<C> needs an odd C: p = 2^128 - C must be odd");

    uint4 limb;

    static Fp128 of(ulong2 v) { return Fp128{fp128_detail::narrow(v)}; }

    static Fp128 zero() { return Fp128{uint4(0u)}; }
    static Fp128 one() { return Fp128{uint4(1u, 0u, 0u, 0u)}; }
    static Fp128 from_u64(ulong v) { return of(ulong2(v, 0ul)); }
    static Fp128 from_i64(long v) { return of(fp128_detail::from_i64<C>(v)); }

    friend Fp128 operator+(Fp128 a, Fp128 b) {
        return of(fp128_detail::add<C>(fp128_detail::wide(a.limb), fp128_detail::wide(b.limb)));
    }
    friend Fp128 operator-(Fp128 a, Fp128 b) {
        return of(fp128_detail::sub<C>(fp128_detail::wide(a.limb), fp128_detail::wide(b.limb)));
    }
    friend Fp128 operator-(Fp128 a) {
        return of(fp128_detail::sub<C>(ulong2(0ul), fp128_detail::wide(a.limb)));
    }
    friend Fp128 operator*(Fp128 a, Fp128 b) {
        return of(fp128_detail::reduce_4<C>(
            fp128_detail::mul_wide(fp128_detail::wide(a.limb), fp128_detail::wide(b.limb))));
    }
    friend Fp128 square(Fp128 a) {
        return of(fp128_detail::reduce_4<C>(fp128_detail::sqr_wide(fp128_detail::wide(a.limb))));
    }
    friend Fp128 mul_u64(Fp128 a, ulong s) {
        return of(fp128_detail::mul_u64<C>(fp128_detail::wide(a.limb), s));
    }
    friend Fp128 mul_i64(Fp128 a, long s) {
        return of(fp128_detail::mul_i64<C>(fp128_detail::wide(a.limb), s));
    }
};

} // namespace jolt

#endif // JOLT_FIELD_FP128_H
