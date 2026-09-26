// jolt::Ext2<F>: the quadratic extension F[u] / (u^2 - 2), bit-exact with
// jolt_field::solinas::Ext2<F> (FpExt2<F, TwoNr>).
//
// The formulas follow crates/jolt-field/src/solinas/ext.rs. F[u] / (u^2 - 2)
// is a field exactly when p = +-3 (mod 8); every registered pseudo-Mersenne
// prime has p = 5 (mod 8). The arithmetic here does not depend on that, and
// agrees with jolt_field for any F.
//
// Representation: the coefficients c0 and c1 of c0 + c1 u, in that order,
// the layout of the CPU's [F; 2]. F is any field type with the operators of
// Fp64 and Fp128. Every operation takes canonical inputs and returns a
// canonical output.
//
// Multiply and square dispatch to ext2_mul and ext2_square: generic
// Karatsuba forms, and forms over Fp64<C> with C < 2^31 that sum each
// coefficient's products unreduced and reduce once. The dispatch is by
// overload; results agree either way, since both are canonical. The Fp64
// forms need jolt/field/fp64.h first, as FIELD_HEADERS orders them.

#ifndef JOLT_FIELD_EXT2_H
#define JOLT_FIELD_EXT2_H

#include <metal_stdlib>

namespace jolt {

template <typename F>
struct Ext2 {
    using Base = F;

    F c0;
    F c1;

    static Ext2 zero() { return Ext2{F::zero(), F::zero()}; }
    static Ext2 one() { return Ext2{F::one(), F::zero()}; }
    static Ext2 from_u64(ulong v) { return Ext2{F::from_u64(v), F::zero()}; }
    static Ext2 from_i64(long v) { return Ext2{F::from_i64(v), F::zero()}; }

    // Multiplication by the non-residue 2 is a doubling.
    static F mul_non_residue(F x) { return x + x; }

    friend Ext2 operator+(Ext2 a, Ext2 b) { return Ext2{a.c0 + b.c0, a.c1 + b.c1}; }
    friend Ext2 operator-(Ext2 a, Ext2 b) { return Ext2{a.c0 - b.c0, a.c1 - b.c1}; }
    friend Ext2 operator-(Ext2 a) { return Ext2{-a.c0, -a.c1}; }

    friend Ext2 operator*(Ext2 a, Ext2 b) { return ext2_mul(a, b); }
    friend Ext2 square(Ext2 a) { return ext2_square(a); }

    // Multiplication by a base-field element, coefficient-wise.
    friend Ext2 mul_base(Ext2 a, F x) { return Ext2{a.c0 * x, a.c1 * x}; }
    friend Ext2 mul_u64(Ext2 a, ulong s) { return Ext2{mul_u64(a.c0, s), mul_u64(a.c1, s)}; }
    friend Ext2 mul_i64(Ext2 a, long s) { return Ext2{mul_i64(a.c0, s), mul_i64(a.c1, s)}; }
};

// Karatsuba: three base multiplies.
// (a0 + a1 u)(b0 + b1 u) = (v0 + 2 v1) + ((a0 + a1)(b0 + b1) - v0 - v1) u
// with v0 = a0 b0 and v1 = a1 b1.
template <typename F>
Ext2<F> ext2_mul(Ext2<F> a, Ext2<F> b) {
    F v0 = a.c0 * b.c0;
    F v1 = a.c1 * b.c1;
    F cross = (a.c0 + a.c1) * (b.c0 + b.c1);
    return Ext2<F>{v0 + Ext2<F>::mul_non_residue(v1), cross - v0 - v1};
}

// Two base multiplies: (c0 + c1 u)^2 = (c0^2 + 2 c1^2) + (2 c0 c1) u.
template <typename F>
Ext2<F> ext2_square(Ext2<F> a) {
    return Ext2<F>{square(a.c0) + Ext2<F>::mul_non_residue(square(a.c1)),
                   (a.c0 + a.c0) * a.c1};
}

// Over Fp64<C> with C < 2^31, each coefficient is a sum of at most three
// 128-bit products, which fp64_detail::reduce_sum reduces once:
// c0 = a0 b0 + 2 a1 b1 and c1 = a0 b1 + a1 b0, with the non-residue 2 as a
// second add of a1 b1. Four multiplies and two reductions against
// Karatsuba's three and three; faster on the GPU (specs/jolt-metal-field.md).
template <uint C>
metal::enable_if_t<(C < (1u << 31)), Ext2<Fp64<C>>> ext2_mul(Ext2<Fp64<C>> a,
                                                             Ext2<Fp64<C>> b) {
    using namespace fp64_detail;
    Wide v1 = mul_wide(a.c1.word, b.c1.word);
    Sum3 c0 = add(add(sum(mul_wide(a.c0.word, b.c0.word)), v1), v1);
    Sum3 c1 = add(sum(mul_wide(a.c0.word, b.c1.word)), mul_wide(a.c1.word, b.c0.word));
    return Ext2<Fp64<C>>{Fp64<C>{reduce_sum<C>(c0)}, Fp64<C>{reduce_sum<C>(c1)}};
}

// c0 = c0^2 + 2 c1^2 reduced once; c1 = 2 c0 c1 as in the generic form.
template <uint C>
metal::enable_if_t<(C < (1u << 31)), Ext2<Fp64<C>>> ext2_square(Ext2<Fp64<C>> a) {
    using namespace fp64_detail;
    Wide v1 = mul_wide(a.c1.word, a.c1.word);
    Sum3 c0 = add(add(sum(mul_wide(a.c0.word, a.c0.word)), v1), v1);
    return Ext2<Fp64<C>>{Fp64<C>{reduce_sum<C>(c0)}, (a.c0 + a.c0) * a.c1};
}

} // namespace jolt

#endif // JOLT_FIELD_EXT2_H
