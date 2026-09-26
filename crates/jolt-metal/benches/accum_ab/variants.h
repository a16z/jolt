// Accumulator representations for jolt::Fp128<C>, for the A/B in
// benches/accum_ab.rs (not for merge). Each variant defines the same
// interface; exactly one is bound to jolt::WithAccumulator per library by
// benches/accum_ab.rs.
//
// Product accumulators (fmadd of two field elements):
// - Carried9: nine carried 32-bit words; the 256-bit product from mul_wide is
//   added with a carry chain.
// - Slots8: eight ulong slots, one per word of the carried 256-bit product;
//   the carry chain runs once, in reduce.
// - Columns8: eight ulong column sums of the partial products' low and high
//   halves; mul_wide's carry chain is skipped entirely.
// - NaiveProduct: a field element; every fmadd reduces.
//
// Small-scalar accumulators (fmadd_i64):
// - Signed7: a two's-complement 224-bit integer.
// - PosNeg7: two carried 224-bit magnitudes, one per sign (the CPU layout).
// - NaiveSigned: a field element; every fmadd reduces.

#include <metal_stdlib>

namespace jolt_ab {

using jolt::Fp128;
using jolt::fp128_detail::fold2_canonicalize;
using jolt::fp128_detail::mul_wide;
using jolt::fp128_detail::mul_wide_u64;
using jolt::fp128_detail::Words;

// Reduces lo + 2^128 hi for the 128-bit lo = w[0..4) and the 160-bit
// hi = w[4..9). t = lo + C (hi mod 2^128) needs one carry per step as in
// reduce_4; the carry out plus C w[8] is at most
// (2^32 - 1) + (2^32 - 1)^2 < 2^64, which fold2_canonicalize accepts.
template <uint C>
inline uint4 reduce_9(Words<9> w) {
    uint4 t;
    ulong u = 0;
    for (int i = 0; i < 4; i++) {
        u = ulong(w[4 + i]) * C + w[i] + (u >> 32);
        t[i] = uint(u);
    }
    return fold2_canonicalize<C>(t, ulong(w[8]) * C + (u >> 32));
}

// Carry-propagates ulong slots s[k] of weight 2^(32k) into nine words. With
// every slot at most m (2^32 - 1), each carry is below m, so each step is below
// m 2^32; that fits a ulong when m <= 2^32.
inline Words<9> carry_slots(metal::array<ulong, 8> s) {
    Words<9> w;
    ulong t = 0;
    for (int k = 0; k < 8; k++) {
        t = s[k] + (t >> 32);
        w[k] = uint(t);
    }
    w[8] = uint(t >> 32);
    return w;
}

template <typename F>
struct Carried9;
template <uint C>
struct Carried9<Fp128<C>> {
    using F = Fp128<C>;
    // Terms below (p - 1)^2 < 2^256 sum below 2^288 for up to 2^32 terms.
    static constexpr constant ulong CAPACITY = 1ul << 32;
    Words<9> w;

    static Carried9 zero() {
        Carried9 z;
        for (int k = 0; k < 9; k++) {
            z.w[k] = 0;
        }
        return z;
    }
    void add_words(Words<8> p) {
        ulong t = 0;
        for (int k = 0; k < 8; k++) {
            t = ulong(w[k]) + p[k] + (t >> 32);
            w[k] = uint(t);
        }
        w[8] += uint(t >> 32);
    }
    void add(F v) {
        add_words(Words<8>{v.limb.x, v.limb.y, v.limb.z, v.limb.w, 0u, 0u, 0u, 0u});
    }
    void fmadd(F a, F b) { add_words(mul_wide(a.limb, b.limb)); }
    void merge(Carried9 o) {
        ulong t = 0;
        for (int k = 0; k < 9; k++) {
            t = ulong(w[k]) + o.w[k] + (t >> 32);
            w[k] = uint(t);
        }
    }
    F reduce() { return F{reduce_9<C>(w)}; }
};

template <typename F>
struct Slots8;
template <uint C>
struct Slots8<Fp128<C>> {
    using F = Fp128<C>;
    // Each slot grows by below 2^32 per term: 2^32 terms (carry_slots).
    static constexpr constant ulong CAPACITY = 1ul << 32;
    metal::array<ulong, 8> s;

    static Slots8 zero() {
        Slots8 z;
        for (int k = 0; k < 8; k++) {
            z.s[k] = 0;
        }
        return z;
    }
    void add(F v) {
        for (int k = 0; k < 4; k++) {
            s[k] += v.limb[k];
        }
    }
    void fmadd(F a, F b) {
        Words<8> p = mul_wide(a.limb, b.limb);
        for (int k = 0; k < 8; k++) {
            s[k] += p[k];
        }
    }
    void merge(Slots8 o) {
        for (int k = 0; k < 8; k++) {
            s[k] += o.s[k];
        }
    }
    F reduce() { return F{reduce_9<C>(carry_slots(s))}; }
};

template <typename F>
struct Columns8;
template <uint C>
struct Columns8<Fp128<C>> {
    using F = Fp128<C>;
    // Column k receives the low halves of the products with i + j = k and the
    // high halves of those with i + j = k - 1: at most 7 values below 2^32
    // per term (k = 3, 4). carry_slots needs 7 n (2^32 - 1) <= 2^32 (2^32 - 1).
    static constexpr constant ulong CAPACITY = (1ul << 32) / 7;
    metal::array<ulong, 8> s;

    static Columns8 zero() {
        Columns8 z;
        for (int k = 0; k < 8; k++) {
            z.s[k] = 0;
        }
        return z;
    }
    void add(F v) {
        for (int k = 0; k < 4; k++) {
            s[k] += v.limb[k];
        }
    }
    void fmadd(F a, F b) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                ulong p = ulong(a.limb[i]) * b.limb[j];
                s[i + j] += uint(p);
                s[i + j + 1] += p >> 32;
            }
        }
    }
    void merge(Columns8 o) {
        for (int k = 0; k < 8; k++) {
            s[k] += o.s[k];
        }
    }
    F reduce() { return F{reduce_9<C>(carry_slots(s))}; }
};

template <typename F>
struct NaiveProduct {
    static constexpr constant ulong CAPACITY = ~0ul;
    F sum;

    static NaiveProduct zero() { return NaiveProduct{F::zero()}; }
    void add(F v) { sum = sum + v; }
    void fmadd(F a, F b) { sum = sum + a * b; }
    void merge(NaiveProduct o) { sum = sum + o.sum; }
    F reduce() { return sum; }
};

// Magnitude of a signed scalar, computed in unsigned arithmetic.
inline ulong magnitude(long s) {
    return s < 0 ? 0ul - ulong(s) : ulong(s);
}

template <typename F>
struct Signed7;
template <uint C>
struct Signed7<Fp128<C>> {
    using F = Fp128<C>;
    // Terms have magnitude below p 2^64 < 2^192; 2^31 of them stay within
    // (-2^223, 2^223), so bit 223 is the sign.
    static constexpr constant ulong CAPACITY = 1ul << 31;
    Words<7> w;

    static Signed7 zero() {
        Signed7 z;
        for (int k = 0; k < 7; k++) {
            z.w[k] = 0;
        }
        return z;
    }
    // w += p when !negative, w -= p when negative, modulo 2^224.
    void add_signed(Words<6> p, bool negative) {
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
        add_signed(Words<6>{v.limb.x, v.limb.y, v.limb.z, v.limb.w, 0u, 0u}, false);
    }
    void fmadd_i64(F a, long s) { add_signed(mul_wide_u64(a.limb, magnitude(s)), s < 0); }
    void merge(Signed7 o) {
        ulong t = 0;
        for (int k = 0; k < 7; k++) {
            t = ulong(w[k]) + o.w[k] + (t >> 32);
            w[k] = uint(t);
        }
    }
    F reduce() {
        bool negative = (w[6] >> 31) != 0;
        uint mask = negative ? ~0u : 0u;
        Words<9> m;
        ulong t = negative ? 1 : 0;
        for (int k = 0; k < 7; k++) {
            t = ulong(w[k] ^ mask) + t;
            m[k] = uint(t);
            t >>= 32;
        }
        m[7] = 0;
        m[8] = 0;
        uint4 r = reduce_9<C>(m);
        uint4 negated = jolt::fp128_detail::sub<C>(uint4(0u), r);
        return F{negative ? negated : r};
    }
};

template <typename F>
struct PosNeg7;
template <uint C>
struct PosNeg7<Fp128<C>> {
    using F = Fp128<C>;
    // Each magnitude grows by below 2^192 per term.
    static constexpr constant ulong CAPACITY = 1ul << 32;
    Words<7> pos;
    Words<7> neg;

    static PosNeg7 zero() {
        PosNeg7 z;
        for (int k = 0; k < 7; k++) {
            z.pos[k] = 0;
            z.neg[k] = 0;
        }
        return z;
    }
    static void add_to(thread Words<7>& w, Words<6> p) {
        ulong t = 0;
        for (int k = 0; k < 6; k++) {
            t = ulong(w[k]) + p[k] + (t >> 32);
            w[k] = uint(t);
        }
        w[6] += uint(t >> 32);
    }
    static void merge_into(thread Words<7>& w, Words<7> o) {
        ulong t = 0;
        for (int k = 0; k < 7; k++) {
            t = ulong(w[k]) + o[k] + (t >> 32);
            w[k] = uint(t);
        }
    }
    static uint4 reduce_7(Words<7> w) {
        return reduce_9<C>(Words<9>{w[0], w[1], w[2], w[3], w[4], w[5], w[6], 0u, 0u});
    }
    void add(F v) { add_to(pos, Words<6>{v.limb.x, v.limb.y, v.limb.z, v.limb.w, 0u, 0u}); }
    void fmadd_i64(F a, long s) {
        Words<6> p = mul_wide_u64(a.limb, magnitude(s));
        if (s < 0) {
            add_to(neg, p);
        } else {
            add_to(pos, p);
        }
    }
    void merge(PosNeg7 o) {
        merge_into(pos, o.pos);
        merge_into(neg, o.neg);
    }
    F reduce() { return F{jolt::fp128_detail::sub<C>(reduce_7(pos), reduce_7(neg))}; }
};

template <typename F>
struct NaiveSigned {
    static constexpr constant ulong CAPACITY = ~0ul;
    F sum;

    static NaiveSigned zero() { return NaiveSigned{F::zero()}; }
    void add(F v) { sum = sum + v; }
    void fmadd_i64(F a, long s) { sum = sum + mul_i64(a, s); }
    void merge(NaiveSigned o) { sum = sum + o.sum; }
    F reduce() { return sum; }
};

} // namespace jolt_ab
