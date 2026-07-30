// BN254 tower extension Fq12 = Fq6[w]/(w² − v), Fq6 = Fq2[v]/(v³ − ξ),
// ξ = 9 + u — and the Miller-loop kernels built on it (W6).
//
// Compiled after g2.metal, whose Fq2 Karatsuba/complex primitives this file
// extends. Algorithms mirror arkworks exactly (values are what parity pins;
// the algorithms are mirrored so a reviewer can diff them side by side):
// Fq6 mul = Devegili–OhEig–Scott–Dahab §4 Karatsuba, Fq6 sqr = CH-SQR2,
// Fq12 mul/sqr = quadratic-extension Karatsuba over β = v, and the pairing's
// sparse Fq12 multiplication mul_by_034 (D-twist line form).
//
// The generated preamble supplies (see `metal::miller::pairing_preamble`,
// every value read from arkworks BN254 config — no hand-written limbs):
//   JK_ATE_LEN / JK_ATE[]      — ATE_LOOP_COUNT digits (LSB order)
//   JK_ELL_COEFFS              — line coefficients per prepared pair (87)
//   FQ_TWO_INV[FR_LIMBS]       — ½ in Fq, Montgomery form
//   JK_G2B_C0/C1[FR_LIMBS]     — the twist's b coefficient (Fq2)
//   JK_MQX_C0/C1, JK_MQY_C0/C1 — TWIST_MUL_BY_Q_X / _Y (Fq2)
// The ξ = 9 + u shape itself is pinned by a host assert at preamble
// generation (fail-closed at context construction), which licenses the
// shift-add specialization in fq2_mul_by_xi.

inline Fq2El fq2_neg(Fq2El a) {
    Fq2El r;
    r.c0 = fq_sub(fq_zero(), a.c0);
    r.c1 = fq_sub(fq_zero(), a.c1);
    return r;
}

// Component-wise scaling by a base-field element (2 Fq muls) — arkworks
// mul_assign_by_fp, the line-evaluation workhorse (c0·P.y, c1·P.x).
inline Fq2El fq2_mul_by_fq(Fq2El a, Fq256 s) {
    Fq2El r;
    r.c0 = fq_mul(a.c0, s);
    r.c1 = fq_mul(a.c1, s);
    return r;
}

// ξ·a for ξ = 9 + u: (9a0 − a1) + (a0 + 9a1)u, with 9x = 8x + x built from
// three doublings — no Fq multiplication. Valid because the host asserts
// Fq6Config::NONRESIDUE == 9 + u before emitting the preamble.
inline Fq2El fq2_mul_by_xi(Fq2El a) {
    Fq256 nine_a0 = fq_add(fq_dbl(fq_dbl(fq_dbl(a.c0))), a.c0);
    Fq256 nine_a1 = fq_add(fq_dbl(fq_dbl(fq_dbl(a.c1))), a.c1);
    Fq2El r;
    r.c0 = fq_sub(nine_a0, a.c1);
    r.c1 = fq_add(a.c0, nine_a1);
    return r;
}

struct Fq6El {
    Fq2El c0;
    Fq2El c1;
    Fq2El c2;
};

inline Fq6El fq6_add(Fq6El a, Fq6El b) {
    Fq6El r;
    r.c0 = fq2_add(a.c0, b.c0);
    r.c1 = fq2_add(a.c1, b.c1);
    r.c2 = fq2_add(a.c2, b.c2);
    return r;
}

inline Fq6El fq6_sub(Fq6El a, Fq6El b) {
    Fq6El r;
    r.c0 = fq2_sub(a.c0, b.c0);
    r.c1 = fq2_sub(a.c1, b.c1);
    r.c2 = fq2_sub(a.c2, b.c2);
    return r;
}

// v·(a0 + a1v + a2v²) = ξ·a2 + a0·v + a1·v² — the Fq12 nonresidue action.
inline Fq6El fq6_mul_by_v(Fq6El a) {
    Fq6El r;
    r.c0 = fq2_mul_by_xi(a.c2);
    r.c1 = a.c0;
    r.c2 = a.c1;
    return r;
}

// Karatsuba (6 Fq2 muls) — arkworks CubicExtField::mul_assign.
inline Fq6El fq6_mul(Fq6El x, Fq6El y) {
    Fq2El ad = fq2_mul(x.c0, y.c0);
    Fq2El be = fq2_mul(x.c1, y.c1);
    Fq2El cf = fq2_mul(x.c2, y.c2);
    Fq2El t_x = fq2_sub(fq2_sub(fq2_mul(fq2_add(x.c1, x.c2), fq2_add(y.c1, y.c2)), be), cf);
    Fq2El t_y = fq2_sub(fq2_sub(fq2_mul(fq2_add(x.c0, x.c1), fq2_add(y.c0, y.c1)), ad), be);
    Fq2El t_z = fq2_sub(
        fq2_add(fq2_sub(fq2_mul(fq2_add(x.c0, x.c2), fq2_add(y.c0, y.c2)), ad), be), cf);
    Fq6El r;
    r.c0 = fq2_add(ad, fq2_mul_by_xi(t_x));
    r.c1 = fq2_add(t_y, fq2_mul_by_xi(cf));
    r.c2 = t_z;
    return r;
}

// CH-SQR2 (2 muls + 3 squarings) — arkworks CubicExtField::square_in_place.
inline Fq6El fq6_sqr(Fq6El a) {
    Fq2El s0 = fq2_sqr(a.c0);
    Fq2El s1 = fq2_dbl(fq2_mul(a.c0, a.c1));
    Fq2El s2 = fq2_sqr(fq2_add(fq2_sub(a.c0, a.c1), a.c2));
    Fq2El s3 = fq2_dbl(fq2_mul(a.c1, a.c2));
    Fq2El s4 = fq2_sqr(a.c2);
    Fq6El r;
    r.c0 = fq2_add(s0, fq2_mul_by_xi(s3));
    r.c1 = fq2_add(s1, fq2_mul_by_xi(s4));
    r.c2 = fq2_sub(fq2_sub(fq2_add(fq2_add(s1, s2), s3), s0), s4);
    return r;
}

// Sparse Fq6 mul by b0 + b1·v (5 Fq2 muls) — arkworks Fp6::mul_by_01.
inline Fq6El fq6_mul_by_01(Fq6El s, Fq2El b0, Fq2El b1) {
    Fq2El a_a = fq2_mul(s.c0, b0);
    Fq2El b_b = fq2_mul(s.c1, b1);
    Fq6El r;
    r.c0 = fq2_add(
        fq2_mul_by_xi(fq2_sub(fq2_mul(b1, fq2_add(s.c1, s.c2)), b_b)), a_a);
    r.c1 = fq2_sub(fq2_sub(fq2_mul(fq2_add(b0, b1), fq2_add(s.c0, s.c1)), a_a), b_b);
    r.c2 = fq2_add(fq2_sub(fq2_mul(b0, fq2_add(s.c0, s.c2)), a_a), b_b);
    return r;
}

struct Fq12El {
    Fq6El c0;
    Fq6El c1;
};

inline Fq12El fq12_one() {
    Fq12El r;
    r.c0.c0 = fq2_one();
    r.c0.c1 = fq2_zero();
    r.c0.c2 = fq2_zero();
    r.c1.c0 = fq2_zero();
    r.c1.c1 = fq2_zero();
    r.c1.c2 = fq2_zero();
    return r;
}

// Karatsuba over β = v (2 Fq6 muls + 1 for the cross term).
inline Fq12El fq12_mul(Fq12El a, Fq12El b) {
    Fq6El v0 = fq6_mul(a.c0, b.c0);
    Fq6El v1 = fq6_mul(a.c1, b.c1);
    Fq12El r;
    r.c1 = fq6_sub(fq6_sub(fq6_mul(fq6_add(a.c0, a.c1), fq6_add(b.c0, b.c1)), v0), v1);
    r.c0 = fq6_add(v0, fq6_mul_by_v(v1));
    return r;
}

// (c0 + c1w)² = (c0² + βc1², 2c0c1) via the (c0−c1)(c0−βc1) trick
// (2 Fq6 muls) — arkworks QuadExtField::square_in_place, β ≠ −1 branch.
inline Fq12El fq12_sqr(Fq12El a) {
    Fq6El v0 = fq6_sub(a.c0, a.c1);
    Fq6El v3 = fq6_sub(a.c0, fq6_mul_by_v(a.c1));
    Fq6El v2 = fq6_mul(a.c0, a.c1);
    v0 = fq6_mul(v0, v3);
    Fq12El r;
    r.c1 = fq6_add(v2, v2);
    r.c0 = fq6_add(fq6_add(v0, v2), fq6_mul_by_v(v2));
    return r;
}

// f ← f·(c0 + c3·w + c4·v·w), the D-twist line value (13 Fq2 muls) —
// arkworks Fp12::mul_by_034.
inline Fq12El fq12_mul_by_034(Fq12El f, Fq2El c0, Fq2El c3, Fq2El c4) {
    Fq6El a;
    a.c0 = fq2_mul(f.c0.c0, c0);
    a.c1 = fq2_mul(f.c0.c1, c0);
    a.c2 = fq2_mul(f.c0.c2, c0);
    Fq6El b = fq6_mul_by_01(f.c1, c3, c4);
    Fq6El e = fq6_mul_by_01(fq6_add(f.c0, f.c1), fq2_add(c0, c3), c4);
    Fq12El r;
    r.c1 = fq6_sub(e, fq6_add(a, b));
    r.c0 = fq6_add(fq6_mul_by_v(b), a);
    return r;
}

// --- host I/O ----------------------------------------------------------------
//
// An Fq12's device words are its 12 Fq Montgomery limb runs in arkworks
// memory order: c0.c0.c0, c0.c0.c1, c0.c1.c0, … (Fq2 minor, Fq6 middle,
// quadratic top) — layout pinned host-side in `metal::miller`.

inline Fq2El fq2_load_at(device const uint* src) {
    Fq2El r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.c0.v[i] = src[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.c1.v[i] = src[FR_LIMBS + i];
    }
    return r;
}

inline void fq2_store_at(device uint* dst, Fq2El a) {
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[i] = a.c0.v[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[FR_LIMBS + i] = a.c1.v[i];
    }
}

inline Fq6El fq6_load_at(device const uint* src) {
    Fq6El r;
    r.c0 = fq2_load_at(src);
    r.c1 = fq2_load_at(src + 2u * FR_LIMBS);
    r.c2 = fq2_load_at(src + 4u * FR_LIMBS);
    return r;
}

inline void fq6_store_at(device uint* dst, Fq6El a) {
    fq2_store_at(dst, a.c0);
    fq2_store_at(dst + 2u * FR_LIMBS, a.c1);
    fq2_store_at(dst + 4u * FR_LIMBS, a.c2);
}

inline Fq12El fq12_load_at(device const uint* src) {
    Fq12El r;
    r.c0 = fq6_load_at(src);
    r.c1 = fq6_load_at(src + 6u * FR_LIMBS);
    return r;
}

inline void fq12_store_at(device uint* dst, Fq12El a) {
    fq6_store_at(dst, a.c0);
    fq6_store_at(dst + 6u * FR_LIMBS, a.c1);
}

// --- parity / microbench kernels ---------------------------------------------
//
// Elementwise tower ops with a chain count: k = 1 is the parity shape
// (out[i] = a[i] ∘ b[i]), larger k re-applies b[i] serially for the
// compute-bound rate (the dependent chain defeats latency hiding the same
// way FrPow2k does).

struct TowerOpParams {
    uint n;
    uint k;
};

kernel void jk_fq6_mul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant TowerOpParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    Fq6El x = fq6_load_at(a + tid * (6u * FR_LIMBS));
    Fq6El y = fq6_load_at(b + tid * (6u * FR_LIMBS));
    for (uint i = 0; i < p.k; i++) {
        x = fq6_mul(x, y);
    }
    fq6_store_at(out + tid * (6u * FR_LIMBS), x);
}

kernel void jk_fq6_sqr(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant TowerOpParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    Fq6El x = fq6_load_at(a + tid * (6u * FR_LIMBS));
    for (uint i = 0; i < p.k; i++) {
        x = fq6_sqr(x);
    }
    fq6_store_at(out + tid * (6u * FR_LIMBS), x);
}

kernel void jk_fq12_mul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant TowerOpParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    Fq12El x = fq12_load_at(a + tid * (12u * FR_LIMBS));
    Fq12El y = fq12_load_at(b + tid * (12u * FR_LIMBS));
    for (uint i = 0; i < p.k; i++) {
        x = fq12_mul(x, y);
    }
    fq12_store_at(out + tid * (12u * FR_LIMBS), x);
}

kernel void jk_fq12_sqr(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant TowerOpParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    Fq12El x = fq12_load_at(a + tid * (12u * FR_LIMBS));
    for (uint i = 0; i < p.k; i++) {
        x = fq12_sqr(x);
    }
    fq12_store_at(out + tid * (12u * FR_LIMBS), x);
}

// f[i] ← f[i]·line(coeff[i], p[i]) k times: the ell application including
// the two base-field scalings. Points stride JK_G1_AFFINE_STRIDE (host
// G1Affine memory); coeffs are 6 Fq runs (c0 ‖ c1 ‖ c2).
kernel void jk_fq12_mul034(
    device const uint* f_in [[buffer(0)]],
    device const uint* coeffs [[buffer(1)]],
    device const uint* ps [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant TowerOpParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    Fq12El f = fq12_load_at(f_in + tid * (12u * FR_LIMBS));
    device const uint* c = coeffs + tid * (6u * FR_LIMBS);
    Fq2El c0 = fq2_load_at(c);
    Fq2El c1 = fq2_load_at(c + 2u * FR_LIMBS);
    Fq2El c2 = fq2_load_at(c + 4u * FR_LIMBS);
    G1AffinePt pt = g1_load_base(ps, tid);
    for (uint i = 0; i < p.k; i++) {
        f = fq12_mul_by_034(f, fq2_mul_by_fq(c0, pt.y), fq2_mul_by_fq(c1, pt.x), c2);
    }
    fq12_store_at(out + tid * (12u * FR_LIMBS), f);
}

// --- Miller loop, prepared-coefficient form (stage-0 tier-2) ------------------
//
// Thread t runs the full ate ladder over its own pair sub-batch
// [t·pairs_per_thread, …) and writes its partial Miller value; the batch
// value is the (exact, order-free) Fq12 product of the partials, which the
// host folds into the Tier2Accumulator. Identical values to arkworks
// multi_miller_loop by the ladder-distributivity argument in
// jolt_dory::tier2 — (f_A·f_B)²·l_A·l_B = (f_A²·l_A)·(f_B²·l_B).
//
// The G2 side arrives fully prepared (the same ell coefficients arkworks
// computed host-side, shared across every column of the commit pass),
// step-major: coefficient `step` of pair `pair` at
// (step·n_pairs + pair)·6·FR_LIMBS — SIMD-adjacent threads read adjacent
// pairs' coefficients, so each step's loads stripe contiguously.
//
// A (0, 0) G1 coordinate pair is the identity sentinel: its line
// applications are skipped, matching arkworks' pair filtering (the pair
// contributes 1). Coefficient slots for such pairs still exist and are
// never read.
struct MillerTableParams {
    uint n_pairs;
    uint pairs_per_thread;
};

kernel void jk_miller_table(
    device const uint* ps [[buffer(0)]],      // G1Affine memory, JK_G1_AFFINE_STRIDE
    device const uint* coeffs [[buffer(1)]],  // JK_ELL_COEFFS × n_pairs × 6·FR_LIMBS
    device uint* out [[buffer(2)]],           // one Fq12 per thread
    constant MillerTableParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint base = tid * p.pairs_per_thread;
    if (base >= p.n_pairs) {
        return;
    }
    uint count = min(p.pairs_per_thread, p.n_pairs - base);

    Fq12El f = fq12_one();
    uint step = 0;
    for (uint it = 0; it < JK_ATE_LEN; it++) {
        if (it != 0) {
            f = fq12_sqr(f);
        }
        for (uint j = 0; j < count; j++) {
            uint pair = base + j;
            G1AffinePt pt = g1_load_base(ps, pair);
            if (fq_is_zero(pt.x) && fq_is_zero(pt.y)) {
                continue;
            }
            device const uint* c = coeffs + (step * p.n_pairs + pair) * (6u * FR_LIMBS);
            f = fq12_mul_by_034(
                f,
                fq2_mul_by_fq(fq2_load_at(c), pt.y),
                fq2_mul_by_fq(fq2_load_at(c + 2u * FR_LIMBS), pt.x),
                fq2_load_at(c + 4u * FR_LIMBS));
        }
        step++;
        int digit = JK_ATE[JK_ATE_LEN - 1u - it];
        if (digit != 0) {
            for (uint j = 0; j < count; j++) {
                uint pair = base + j;
                G1AffinePt pt = g1_load_base(ps, pair);
                if (fq_is_zero(pt.x) && fq_is_zero(pt.y)) {
                    continue;
                }
                device const uint* c = coeffs + (step * p.n_pairs + pair) * (6u * FR_LIMBS);
                f = fq12_mul_by_034(
                    f,
                    fq2_mul_by_fq(fq2_load_at(c), pt.y),
                    fq2_mul_by_fq(fq2_load_at(c + 2u * FR_LIMBS), pt.x),
                    fq2_load_at(c + 4u * FR_LIMBS));
            }
            step++;
        }
    }
    // The two Frobenius-twisted addition steps (q1, then q2).
    for (uint k = 0; k < 2u; k++) {
        for (uint j = 0; j < count; j++) {
            uint pair = base + j;
            G1AffinePt pt = g1_load_base(ps, pair);
            if (fq_is_zero(pt.x) && fq_is_zero(pt.y)) {
                continue;
            }
            device const uint* c = coeffs + (step * p.n_pairs + pair) * (6u * FR_LIMBS);
            f = fq12_mul_by_034(
                f,
                fq2_mul_by_fq(fq2_load_at(c), pt.y),
                fq2_mul_by_fq(fq2_load_at(c + 2u * FR_LIMBS), pt.x),
                fq2_load_at(c + 4u * FR_LIMBS));
        }
        step++;
    }
    fq12_store_at(out + tid * (12u * FR_LIMBS), f);
}

// --- Miller loop, on-the-fly form (stage-8 reduce rounds) ---------------------
//
// One pair per thread: the G2 side changes every call (dory's folded v₂
// halves), so the thread runs arkworks' G2 preparation ladder
// (homogeneous-projective double/add steps, eprint 2013/722) inline,
// applying each line as it is produced. Per-pair squaring ladders cost
// ~27% extra Fq12 squarings versus sub-batching, paid for zero host
// preparation. Values equal arkworks' by the same partial-product argument.

struct G2Hom {
    Fq2El x;
    Fq2El y;
    Fq2El z;
};

// arkworks G2HomProjective::double_in_place (D twist), line = (−h, 3j, i).
inline void jk_fly_dbl(thread G2Hom &r, thread Fq2El &l0, thread Fq2El &l1, thread Fq2El &l2) {
    Fq256 two_inv;
    for (uint i = 0; i < FR_LIMBS; i++) {
        two_inv.v[i] = FQ_TWO_INV[i];
    }
    Fq2El a = fq2_mul(r.x, r.y);
    a = fq2_mul_by_fq(a, two_inv);
    Fq2El b = fq2_sqr(r.y);
    Fq2El c = fq2_sqr(r.z);
    Fq2El g2b;
    for (uint i = 0; i < FR_LIMBS; i++) {
        g2b.c0.v[i] = JK_G2B_C0[i];
        g2b.c1.v[i] = JK_G2B_C1[i];
    }
    Fq2El e = fq2_mul(g2b, fq2_add(fq2_dbl(c), c));
    Fq2El f6 = fq2_add(fq2_dbl(e), e);
    Fq2El g = fq2_mul_by_fq(fq2_add(b, f6), two_inv);
    Fq2El h = fq2_sub(fq2_sqr(fq2_add(r.y, r.z)), fq2_add(b, c));
    Fq2El i = fq2_sub(e, b);
    Fq2El j = fq2_sqr(r.x);
    Fq2El e_sq = fq2_sqr(e);
    r.x = fq2_mul(a, fq2_sub(b, f6));
    r.y = fq2_sub(fq2_sqr(g), fq2_add(fq2_dbl(e_sq), e_sq));
    r.z = fq2_mul(b, h);
    l0 = fq2_neg(h);
    l1 = fq2_add(fq2_dbl(j), j);
    l2 = i;
}

// arkworks G2HomProjective::add_in_place (D twist), line = (λ, −θ, j).
inline void jk_fly_add(
    thread G2Hom &r,
    Fq2El qx,
    Fq2El qy,
    thread Fq2El &l0,
    thread Fq2El &l1,
    thread Fq2El &l2)
{
    Fq2El theta = fq2_sub(r.y, fq2_mul(qy, r.z));
    Fq2El lambda = fq2_sub(r.x, fq2_mul(qx, r.z));
    Fq2El c = fq2_sqr(theta);
    Fq2El d = fq2_sqr(lambda);
    Fq2El e = fq2_mul(lambda, d);
    Fq2El ff = fq2_mul(r.z, c);
    Fq2El g = fq2_mul(r.x, d);
    Fq2El h = fq2_sub(fq2_add(e, ff), fq2_dbl(g));
    r.x = fq2_mul(lambda, h);
    r.y = fq2_sub(fq2_mul(theta, fq2_sub(g, h)), fq2_mul(e, r.y));
    r.z = fq2_mul(r.z, e);
    l0 = lambda;
    l1 = fq2_neg(theta);
    l2 = fq2_sub(fq2_mul(theta, qx), fq2_mul(lambda, qy));
}

// arkworks mul_by_char: coordinate-wise Frobenius (conjugation, since
// u^q = −u for q ≡ 3 mod 4) times the twist constants.
inline void jk_mul_by_char(thread Fq2El &qx, thread Fq2El &qy) {
    Fq2El mqx;
    Fq2El mqy;
    for (uint i = 0; i < FR_LIMBS; i++) {
        mqx.c0.v[i] = JK_MQX_C0[i];
        mqx.c1.v[i] = JK_MQX_C1[i];
        mqy.c0.v[i] = JK_MQY_C0[i];
        mqy.c1.v[i] = JK_MQY_C1[i];
    }
    qx.c1 = fq_sub(fq_zero(), qx.c1);
    qx = fq2_mul(qx, mqx);
    qy.c1 = fq_sub(fq_zero(), qy.c1);
    qy = fq2_mul(qy, mqy);
}

inline void jk_fly_ell(
    thread Fq12El &f,
    Fq2El l0,
    Fq2El l1,
    Fq2El l2,
    Fq256 px,
    Fq256 py)
{
    f = fq12_mul_by_034(f, fq2_mul_by_fq(l0, py), fq2_mul_by_fq(l1, px), l2);
}

// One (G1, G2) pair per thread, both affine; either side's (0, 0) sentinel
// yields the empty product (f = 1), matching arkworks' pair filter. G2
// points stride JK_G2_AFFINE_STRIDE (host G2Affine memory).
struct MillerFlyParams {
    uint n_pairs;
};

kernel void jk_miller_fly(
    device const uint* ps [[buffer(0)]],   // G1Affine memory
    device const uint* qs [[buffer(1)]],   // G2Affine memory
    device uint* out [[buffer(2)]],        // one Fq12 per pair
    constant MillerFlyParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n_pairs) {
        return;
    }
    Fq12El f = fq12_one();
    G1AffinePt pt = g1_load_base(ps, tid);
    G2AffinePt q = g2_load_base(qs, tid);
    bool live = !(fq_is_zero(pt.x) && fq_is_zero(pt.y)) && !(fq2_is_zero(q.x) && fq2_is_zero(q.y));
    if (live) {
        G2Hom r;
        r.x = q.x;
        r.y = q.y;
        r.z = fq2_one();
        Fq2El nqy = fq2_neg(q.y);
        Fq2El l0;
        Fq2El l1;
        Fq2El l2;
        for (uint it = 0; it < JK_ATE_LEN; it++) {
            if (it != 0) {
                f = fq12_sqr(f);
            }
            jk_fly_dbl(r, l0, l1, l2);
            jk_fly_ell(f, l0, l1, l2, pt.x, pt.y);
            int digit = JK_ATE[JK_ATE_LEN - 1u - it];
            if (digit == 1) {
                jk_fly_add(r, q.x, q.y, l0, l1, l2);
                jk_fly_ell(f, l0, l1, l2, pt.x, pt.y);
            } else if (digit == -1) {
                jk_fly_add(r, q.x, nqy, l0, l1, l2);
                jk_fly_ell(f, l0, l1, l2, pt.x, pt.y);
            }
        }
        // q1 = ψ(q), q2 = ψ²(q) with y negated (X_IS_NEGATIVE = false skips
        // the r.y flip).
        Fq2El q1x = q.x;
        Fq2El q1y = q.y;
        jk_mul_by_char(q1x, q1y);
        Fq2El q2x = q1x;
        Fq2El q2y = q1y;
        jk_mul_by_char(q2x, q2y);
        q2y = fq2_neg(q2y);
        jk_fly_add(r, q1x, q1y, l0, l1, l2);
        jk_fly_ell(f, l0, l1, l2, pt.x, pt.y);
        jk_fly_add(r, q2x, q2y, l0, l1, l2);
        jk_fly_ell(f, l0, l1, l2, pt.x, pt.y);
    }
    fq12_store_at(out + tid * (12u * FR_LIMBS), f);
}
