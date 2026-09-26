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

    // Karatsuba: three base multiplies.
    // (a0 + a1 u)(b0 + b1 u) = (v0 + 2 v1) + ((a0 + a1)(b0 + b1) - v0 - v1) u
    // with v0 = a0 b0 and v1 = a1 b1.
    friend Ext2 operator*(Ext2 a, Ext2 b) {
        F v0 = a.c0 * b.c0;
        F v1 = a.c1 * b.c1;
        F cross = (a.c0 + a.c1) * (b.c0 + b.c1);
        return Ext2{v0 + mul_non_residue(v1), cross - v0 - v1};
    }

    // Two base multiplies: (c0 + c1 u)^2 = (c0^2 + 2 c1^2) + (2 c0 c1) u.
    friend Ext2 square(Ext2 a) {
        return Ext2{square(a.c0) + mul_non_residue(square(a.c1)), (a.c0 + a.c0) * a.c1};
    }

    // Multiplication by a base-field element, coefficient-wise.
    friend Ext2 mul_base(Ext2 a, F x) { return Ext2{a.c0 * x, a.c1 * x}; }
    friend Ext2 mul_u64(Ext2 a, ulong s) { return Ext2{mul_u64(a.c0, s), mul_u64(a.c1, s)}; }
    friend Ext2 mul_i64(Ext2 a, long s) { return Ext2{mul_i64(a.c0, s), mul_i64(a.c1, s)}; }
};

} // namespace jolt

#endif // JOLT_FIELD_EXT2_H
