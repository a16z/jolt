// BN254 G1 point arithmetic over the base field Fq, Montgomery form,
// 32-bit limbs, little-endian — the tier-1 witness-commit kernel.
//
// Compiled behind the generated constants preamble, which additionally
// defines (see `metal::field::constants_preamble`):
//   FQ_MOD[FR_LIMBS]  — the base-field modulus q, LE u32 limbs
//   FQ_INV32          — -q^{-1} mod 2^32
//   JK_G1_AFFINE_STRIDE — u32 stride of a host `ark_bn254::G1Affine`
//
// Fq arithmetic mirrors fr.metal's CIOS exactly (same limb count, different
// modulus constants). Points are Jacobian (X, Y, Z) over Fq with the
// arkworks convention: affine (X/Z^2, Y/Z^3), identity encoded as Z = 0.
// The curve is y^2 = x^3 + 3 (a = 0), so doubling needs no `a` term.
//
// The one kernel, `jk_g1_seg_sum`, computes per-thread sums of selected
// affine bases: thread t accumulates bases[indices[seg_starts[t] ..
// seg_starts[t+1]]] with mixed (Jacobian + affine) additions and writes the
// Jacobian result. Bucket sums larger than the host's segment cap arrive as
// several segments; the host reduces those few partials. Bases are read
// directly from host `ark_bn254::G1Affine` memory (unified), which strides
// JK_G1_AFFINE_STRIDE u32s per point: x limbs at +0, y at +8; the trailing
// `infinity` flag is never set for SRS bases (asserted host-side).

struct Fq256 {
    uint v[FR_LIMBS];
};

inline Fq256 fq_zero() {
    Fq256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = 0u;
    }
    return r;
}

inline bool fq_is_zero(Fq256 a) {
    uint acc = 0u;
    for (uint i = 0; i < FR_LIMBS; i++) {
        acc |= a.v[i];
    }
    return acc == 0u;
}

// r = a - b + (borrow ? q : 0); canonical for canonical inputs.
inline Fq256 fq_sub(Fq256 a, Fq256 b) {
    Fq256 r;
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - borrow;
        r.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    uint mask = (uint)(0u - (uint)borrow);
    ulong carry = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong s = (ulong)r.v[i] + (ulong)(FQ_MOD[i] & mask) + carry;
        r.v[i] = (uint)s;
        carry = s >> 32;
    }
    return r;
}

// r = (a + b) mod q via wide add + branchless trial subtraction.
inline Fq256 fq_add(Fq256 a, Fq256 b) {
    Fq256 sum;
    ulong carry = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + carry;
        sum.v[i] = (uint)s;
        carry = s >> 32;
    }
    Fq256 diff;
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)sum.v[i] - (ulong)FQ_MOD[i] - borrow;
        diff.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    bool take_diff = (carry != 0) || (borrow == 0);
    uint mask = take_diff ? 0xffffffffu : 0u;
    Fq256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = (diff.v[i] & mask) | (sum.v[i] & ~mask);
    }
    return r;
}

inline Fq256 fq_dbl(Fq256 a) {
    return fq_add(a, a);
}

// Montgomery product abR^{-1} mod q, CIOS — fr.metal's fr_mont_mul over q.
inline Fq256 fq_mul(Fq256 a, Fq256 b) {
    uint t[FR_LIMBS + 2];
    for (uint i = 0; i < FR_LIMBS + 2; i++) {
        t[i] = 0u;
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong carry = 0;
        for (uint j = 0; j < FR_LIMBS; j++) {
            ulong cur = (ulong)t[j] + (ulong)a.v[i] * (ulong)b.v[j] + carry;
            t[j] = (uint)cur;
            carry = cur >> 32;
        }
        ulong cur = (ulong)t[FR_LIMBS] + carry;
        t[FR_LIMBS] = (uint)cur;
        t[FR_LIMBS + 1] = (uint)(cur >> 32);

        uint m = t[0] * FQ_INV32;
        cur = (ulong)t[0] + (ulong)m * (ulong)FQ_MOD[0];
        carry = cur >> 32;
        for (uint j = 1; j < FR_LIMBS; j++) {
            cur = (ulong)t[j] + (ulong)m * (ulong)FQ_MOD[j] + carry;
            t[j - 1] = (uint)cur;
            carry = cur >> 32;
        }
        cur = (ulong)t[FR_LIMBS] + carry;
        t[FR_LIMBS - 1] = (uint)cur;
        t[FR_LIMBS] = t[FR_LIMBS + 1] + (uint)(cur >> 32);
        t[FR_LIMBS + 1] = 0u;
    }
    Fq256 sum;
    for (uint i = 0; i < FR_LIMBS; i++) {
        sum.v[i] = t[i];
    }
    Fq256 diff;
    ulong borrow = 0;
    for (uint i = 0; i < FR_LIMBS; i++) {
        ulong d = (ulong)sum.v[i] - (ulong)FQ_MOD[i] - borrow;
        diff.v[i] = (uint)d;
        borrow = (d >> 32) & 1u;
    }
    bool take_diff = (t[FR_LIMBS] != 0u) || (borrow == 0);
    uint mask = take_diff ? 0xffffffffu : 0u;
    Fq256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = (diff.v[i] & mask) | (sum.v[i] & ~mask);
    }
    return r;
}

inline Fq256 fq_sqr(Fq256 a) {
    return fq_mul(a, a);
}

inline Fq256 fq_load_constant(constant uint* src) {
    Fq256 r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.v[i] = src[i];
    }
    return r;
}

// Jacobian point; identity is Z = 0 (X, Y then irrelevant).
struct G1Jac {
    Fq256 x;
    Fq256 y;
    Fq256 z;
};

struct G1AffinePt {
    Fq256 x;
    Fq256 y;
};

// Extended Jacobian (XYZZ): affine coordinates are (X/ZZ, Y/ZZZ). Keeping
// Z² and Z³ removes one Montgomery multiplication from every mixed add.
struct G1Xyzz {
    Fq256 x;
    Fq256 y;
    Fq256 zz;
    Fq256 zzz;
};

inline G1Jac g1_identity() {
    G1Jac p;
    p.x = fq_zero();
    p.y = fq_zero();
    p.z = fq_zero();
    return p;
}

inline G1Xyzz g1_xyzz_identity() {
    G1Xyzz p;
    p.x = fq_zero();
    p.y = fq_zero();
    p.zz = fq_zero();
    p.zzz = fq_zero();
    return p;
}

inline G1AffinePt g1_load_base(device const uint* bases, uint idx) {
    G1AffinePt p;
    device const uint* src = bases + idx * JK_G1_AFFINE_STRIDE;
    for (uint i = 0; i < FR_LIMBS; i++) {
        p.x.v[i] = src[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        p.y.v[i] = src[FR_LIMBS + i];
    }
    return p;
}

inline G1Jac g1_load_jac(device const uint* points, uint idx) {
    device const uint* src = points + idx * (3u * FR_LIMBS);
    G1Jac p;
    for (uint i = 0; i < FR_LIMBS; i++) {
        p.x.v[i] = src[i];
        p.y.v[i] = src[FR_LIMBS + i];
        p.z.v[i] = src[2u * FR_LIMBS + i];
    }
    return p;
}

inline void g1_store_jac(device uint* dst, G1Jac p) {
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[i] = p.x.v[i];
        dst[FR_LIMBS + i] = p.y.v[i];
        dst[2u * FR_LIMBS + i] = p.z.v[i];
    }
}

// Doubling, a = 0 (dbl-2009-l): 2M + 5S.
inline G1Jac g1_dbl(G1Jac p) {
    if (fq_is_zero(p.z)) {
        return p;
    }
    Fq256 a = fq_sqr(p.x);                 // A = X^2
    Fq256 b = fq_sqr(p.y);                 // B = Y^2
    Fq256 c = fq_sqr(b);                   // C = B^2
    // D = 2((X + B)^2 - A - C)
    Fq256 d = fq_sqr(fq_add(p.x, b));
    d = fq_sub(d, a);
    d = fq_sub(d, c);
    d = fq_dbl(d);
    Fq256 e = fq_add(fq_dbl(a), a);        // E = 3A
    Fq256 f = fq_sqr(e);                   // F = E^2
    G1Jac r;
    r.x = fq_sub(fq_sub(f, d), d);         // X3 = F - 2D
    Fq256 c8 = fq_dbl(fq_dbl(fq_dbl(c))); // 8C
    r.y = fq_sub(fq_mul(e, fq_sub(d, r.x)), c8);
    r.z = fq_dbl(fq_mul(p.y, p.z));        // Z3 = 2YZ
    return r;
}

// Mixed addition acc + (x2, y2), affine point assumed NOT infinity
// (madd-2007-bl, 7M + 4S), with the three special cases handled exactly:
// identity accumulator (copy), equal points (double), inverse points
// (identity).
inline G1Jac g1_madd(G1Jac acc, G1AffinePt q) {
    if (fq_is_zero(acc.z)) {
        G1Jac r;
        r.x = q.x;
        r.y = q.y;
        // Z = 1 in Montgomery form is FQ_ONE.
        for (uint i = 0; i < FR_LIMBS; i++) {
            r.z.v[i] = FQ_ONE[i];
        }
        return r;
    }
    Fq256 z1z1 = fq_sqr(acc.z);            // Z1Z1 = Z1^2
    Fq256 u2 = fq_mul(q.x, z1z1);          // U2 = X2·Z1Z1
    Fq256 s2 = fq_mul(fq_mul(q.y, acc.z), z1z1); // S2 = Y2·Z1·Z1Z1
    Fq256 h = fq_sub(u2, acc.x);           // H = U2 - X1
    Fq256 rr = fq_sub(s2, acc.y);          // r' = S2 - Y1 (halved r)
    if (fq_is_zero(h)) {
        if (fq_is_zero(rr)) {
            return g1_dbl(acc);
        }
        return g1_identity();
    }
    Fq256 r2 = fq_dbl(rr);                 // r = 2(S2 - Y1)
    Fq256 hh = fq_sqr(h);                  // HH = H^2
    Fq256 i = fq_dbl(fq_dbl(hh));          // I = 4HH
    Fq256 j = fq_mul(h, i);                // J = H·I
    Fq256 v = fq_mul(acc.x, i);            // V = X1·I
    G1Jac out;
    out.x = fq_sub(fq_sub(fq_sqr(r2), j), fq_dbl(v)); // X3 = r^2 - J - 2V
    Fq256 y1j = fq_mul(acc.y, j);
    out.y = fq_sub(fq_mul(r2, fq_sub(v, out.x)), fq_dbl(y1j)); // Y3 = r(V-X3) - 2Y1·J
    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    out.z = fq_sub(fq_sub(fq_sqr(fq_add(acc.z, h)), z1z1), hh);
    return out;
}

// General Jacobian addition (add-2007-bl), including identity, doubling,
// and inverse-point cases. Resident Dory rounds cannot afford host
// normalization between challenge folds.
inline G1Jac g1_add_jac(G1Jac p, G1Jac q) {
    if (fq_is_zero(p.z)) {
        return q;
    }
    if (fq_is_zero(q.z)) {
        return p;
    }
    Fq256 z1z1 = fq_sqr(p.z);
    Fq256 z2z2 = fq_sqr(q.z);
    Fq256 u1 = fq_mul(p.x, z2z2);
    Fq256 u2 = fq_mul(q.x, z1z1);
    Fq256 s1 = fq_mul(p.y, fq_mul(q.z, z2z2));
    Fq256 s2 = fq_mul(q.y, fq_mul(p.z, z1z1));
    Fq256 h = fq_sub(u2, u1);
    Fq256 rr = fq_sub(s2, s1);
    if (fq_is_zero(h)) {
        return fq_is_zero(rr) ? g1_dbl(p) : g1_identity();
    }
    Fq256 i = fq_sqr(fq_dbl(h));
    Fq256 j = fq_mul(h, i);
    Fq256 r2 = fq_dbl(rr);
    Fq256 v = fq_mul(u1, i);
    G1Jac out;
    out.x = fq_sub(fq_sub(fq_sqr(r2), j), fq_dbl(v));
    out.y = fq_sub(fq_mul(r2, fq_sub(v, out.x)), fq_dbl(fq_mul(s1, j)));
    out.z = fq_mul(
        fq_sub(fq_sub(fq_sqr(fq_add(p.z, q.z)), z1z1), z2z2),
        h
    );
    return out;
}

inline G1Xyzz g1_xyzz_dbl(G1Xyzz p) {
    if (fq_is_zero(p.zz)) {
        return p;
    }
    Fq256 u = fq_dbl(p.y);
    Fq256 v = fq_sqr(u);
    Fq256 w = fq_mul(u, v);
    Fq256 s = fq_mul(p.x, v);
    Fq256 m = fq_sqr(p.x);
    m = fq_add(fq_dbl(m), m);
    G1Xyzz out;
    out.x = fq_sub(fq_sqr(m), fq_dbl(s));
    out.y = fq_sub(fq_mul(m, fq_sub(s, out.x)), fq_mul(w, p.y));
    out.zz = fq_mul(v, p.zz);
    out.zzz = fq_mul(w, p.zzz);
    return out;
}

// Mixed XYZZ + affine addition: 8M + 2S, versus Jacobian's 7M + 4S.
inline G1Xyzz g1_xyzz_madd(G1Xyzz acc, G1AffinePt q) {
    if (fq_is_zero(acc.zz)) {
        G1Xyzz out;
        out.x = q.x;
        out.y = q.y;
        for (uint i = 0; i < FR_LIMBS; i++) {
            out.zz.v[i] = FQ_ONE[i];
            out.zzz.v[i] = FQ_ONE[i];
        }
        return out;
    }
    Fq256 u2 = fq_mul(q.x, acc.zz);
    Fq256 s2 = fq_mul(q.y, acc.zzz);
    Fq256 h = fq_sub(u2, acc.x);
    Fq256 r = fq_sub(s2, acc.y);
    if (fq_is_zero(h)) {
        return fq_is_zero(r) ? g1_xyzz_dbl(acc) : g1_xyzz_identity();
    }
    Fq256 hh = fq_sqr(h);
    Fq256 hhh = fq_mul(h, hh);
    Fq256 v = fq_mul(acc.x, hh);
    G1Xyzz out;
    out.x = fq_sub(fq_sub(fq_sqr(r), hhh), fq_dbl(v));
    out.y = fq_sub(fq_mul(r, fq_sub(v, out.x)), fq_mul(acc.y, hhh));
    out.zz = fq_mul(acc.zz, hh);
    out.zzz = fq_mul(acc.zzz, hhh);
    return out;
}

inline G1Jac g1_xyzz_to_jac(G1Xyzz p) {
    if (fq_is_zero(p.zz)) {
        return g1_identity();
    }
    G1Jac out;
    out.x = fq_mul(p.x, fq_sqr(p.zz));
    out.y = fq_mul(p.y, fq_sqr(p.zzz));
    out.z = p.zzz;
    return out;
}

// --- Stage-8 hint combination (W3b) -----------------------------------------
//
// combined[r] = Σ_p scalar_p · P_{p,r} over the batch opening's ragged hint
// matrix — per row an independent small MSM whose SCALARS are shared by
// every row. One thread per row runs a single high-to-low double-and-add
// across ALL of its live hints, so the ~254 doublings amortize over the
// row's whole point set and the per-bit branches are warp-uniform (the bit
// pattern is the same for every row). Hints arrive sorted by row count
// descending, making each row's live set a PREFIX of the hint order —
// raggedness is one uniform `break`, not a gather structure. Points are
// affine (host batch-normalizes), hint-major, so lane-adjacent rows load
// adjacent points; a (0, 0) coordinate pair — never on y² = x³ + 3 — is the
// host's sentinel for an identity row and contributes nothing.
struct G1CombineParams {
    uint num_rows;
    uint num_hints;
    uint start_bit;  // highest nonzero NAF digit across all scalars
};

// Signed-digit slots per hint scalar in the NAF encoding (host twin:
// `hint_combine::NAF_DIGIT_SLOTS`).
constant uint JK_NAF_DIGIT_SLOTS = 256u;

kernel void jk_g1_combine_rows(
    device const uint* points [[buffer(0)]],   // affine, stride JK_G1_AFFINE_STRIDE
    device const uint* scalars [[buffer(1)]],  // per hint: 64 packed signed NAF words (4 i8 per uint)
    device const uint* lens [[buffer(2)]],     // num_hints row counts, nonincreasing
    device const uint* offsets [[buffer(3)]],  // num_hints starts into `points`
    device uint* out [[buffer(4)]],            // num_rows Jacobian results
    constant G1CombineParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.num_rows) {
        return;
    }
    G1Jac acc = g1_identity();
    for (int bit = (int)p.start_bit; bit >= 0; bit--) {
        acc = g1_dbl(acc);
        uint word_index = (uint)bit >> 2;
        for (uint h = 0; h < p.num_hints; h++) {
            // Nonincreasing lens: every later hint is shorter too.
            if (gid >= lens[h]) {
                break;
            }
            uint word = scalars[h * (JK_NAF_DIGIT_SLOTS / 4u) + word_index];
            int digit = (int)(char)((word >> (((uint)bit & 3u) * 8u)) & 0xffu);
            if (digit != 0) {
                G1AffinePt q = g1_load_base(points, offsets[h] + gid);
                if (!(fq_is_zero(q.x) && fq_is_zero(q.y))) {
                    if (digit < 0) {
                        // -(x, y) = (x, q - y); y is canonical nonzero for
                        // every non-sentinel base, so fq_sub yields the
                        // canonical negation.
                        q.y = fq_sub(fq_zero(), q.y);
                    }
                    acc = g1_madd(acc, q);
                }
            }
        }
    }
    device uint* dst = out + gid * (3u * FR_LIMBS);
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[i] = acc.x.v[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[FR_LIMBS + i] = acc.y.v[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[2u * FR_LIMBS + i] = acc.z.v[i];
    }
}

// --- Stage-8 reduce-round folds (W5a) ----------------------------------------
//
// out[i] = s·P[i] + Q[i] with ONE scalar shared by the whole vector — the
// shape of both `fixed_scalar_mul_bases_then_add` (P = setup bases, Q = the
// running v-vector) and `fixed_scalar_mul_vs_then_add` (P = vL, Q = vR) in
// dory-pcs's reduce-and-fold rounds. Thread-per-element double-and-add: the
// scalar's bit branches are warp-uniform (same challenge for every lane),
// and the host batch-normalizes both inputs to affine so the whole kernel
// is the parity-tested g1_dbl/g1_madd pair — no general Jacobian+Jacobian
// add. A (0, 0) coordinate pair (not on y² = x³ + 3) is the host's identity
// sentinel: a dead P skips the ladder, a dead Q skips the final add.
struct G1MulAddParams {
    uint n;
    uint start_bit;               // highest set bit of the scalar (0 if s = 0)
    uint scalar[FR_LIMBS];        // CANONICAL (integer) LE limbs
};

kernel void jk_g1_scalar_mul_add(
    device const uint* ps [[buffer(0)]],   // affine, stride JK_G1_AFFINE_STRIDE
    device const uint* qs [[buffer(1)]],   // affine, stride JK_G1_AFFINE_STRIDE
    device uint* out [[buffer(2)]],        // n Jacobian results
    constant G1MulAddParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    G1AffinePt base = g1_load_base(ps, tid);
    G1Jac acc = g1_identity();
    if (!(fq_is_zero(base.x) && fq_is_zero(base.y))) {
        for (int bit = (int)p.start_bit; bit >= 0; bit--) {
            acc = g1_dbl(acc);
            if ((p.scalar[(uint)bit >> 5] >> ((uint)bit & 31u)) & 1u) {
                acc = g1_madd(acc, base);
            }
        }
    }
    G1AffinePt addend = g1_load_base(qs, tid);
    if (!(fq_is_zero(addend.x) && fq_is_zero(addend.y))) {
        acc = g1_madd(acc, addend);
    }
    device uint* dst = out + tid * (3u * FR_LIMBS);
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[i] = acc.x.v[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[FR_LIMBS + i] = acc.y.v[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        dst[2u * FR_LIMBS + i] = acc.z.v[i];
    }
}

struct G1ProjectiveMulAddParams {
    uint n;
    uint p_offset;
    uint q_offset;
    int digits[66];
    uint endo[FR_LIMBS];
};

kernel void jk_g1_projective_mul_add(
    device const uint* ps [[buffer(0)]],
    device const uint* qs [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant G1ProjectiveMulAddParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    G1Jac point = g1_load_jac(ps, p.p_offset + tid);
    G1Jac table[8];
    table[0] = point;
    table[1] = g1_dbl(point);
    for (uint k = 2u; k < 8u; k++) {
        table[k] = g1_add_jac(table[k - 1u], point);
    }
    Fq256 endo = fq_load_constant(p.endo);
    G1Jac acc = g1_identity();
    for (int w = 32; w >= 0; w--) {
        if (w != 32) {
            for (uint k = 0u; k < 4u; k++) {
                acc = g1_dbl(acc);
            }
        }
        int d1 = p.digits[w];
        if (d1 != 0) {
            uint magnitude = (uint)(d1 < 0 ? -d1 : d1);
            G1Jac entry = table[magnitude - 1u];
            if (d1 < 0) {
                entry.y = fq_sub(fq_zero(), entry.y);
            }
            acc = g1_add_jac(acc, entry);
        }
        int d2 = p.digits[33 + w];
        if (d2 != 0) {
            uint magnitude = (uint)(d2 < 0 ? -d2 : d2);
            G1Jac entry = table[magnitude - 1u];
            entry.x = fq_mul(entry.x, endo);
            if (d2 < 0) {
                entry.y = fq_sub(fq_zero(), entry.y);
            }
            acc = g1_add_jac(acc, entry);
        }
    }
    acc = g1_add_jac(acc, g1_load_jac(qs, p.q_offset + tid));
    g1_store_jac(out + tid * (3u * FR_LIMBS), acc);
}

constant uint JK_DORY_MSM_WINDOWS = 32u;
constant uint JK_DORY_MSM_BUCKETS = 128u;
constant uint JK_DORY_MSM_BINS = JK_DORY_MSM_WINDOWS * JK_DORY_MSM_BUCKETS;

kernel void jk_dory_msm_hist(
    device const char* digits [[buffer(0)]],
    device atomic_uint* hist [[buffer(1)]],
    constant uint* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint tg_count [[threadgroups_per_grid]])
{
    threadgroup atomic_uint bins[4096];
    for (uint bin = tid; bin < JK_DORY_MSM_BINS; bin += 256u) {
        atomic_store_explicit(&bins[bin], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint len = params[0];
    const uint total = JK_DORY_MSM_WINDOWS * len;
    for (uint item = tg * 256u + tid; item < total; item += tg_count * 256u) {
        const int digit = digits[item];
        if (digit != 0) {
            const uint window = item / len;
            const uint magnitude = (uint)(digit < 0 ? -digit : digit);
            atomic_fetch_add_explicit(
                &bins[window * JK_DORY_MSM_BUCKETS + magnitude - 1u],
                1u,
                memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bin = tid; bin < JK_DORY_MSM_BINS; bin += 256u) {
        const uint count = atomic_load_explicit(&bins[bin], memory_order_relaxed);
        if (count != 0u) {
            atomic_fetch_add_explicit(&hist[bin], count, memory_order_relaxed);
        }
    }
}

kernel void jk_dory_msm_offsets(
    device const uint* hist [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    device uint* cursors [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint scan[256];
    threadgroup uint carry;
    if (tid == 0u) {
        carry = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0u; base < JK_DORY_MSM_BINS; base += 256u) {
        const uint bin = base + tid;
        scan[tid] = hist[bin];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < 256u; stride <<= 1u) {
            const uint addend = tid >= stride ? scan[tid - stride] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scan[tid] += addend;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint exclusive = carry + (tid == 0u ? 0u : scan[tid - 1u]);
        offsets[bin] = exclusive;
        cursors[bin] = exclusive;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 255u) {
            carry += scan[255];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        offsets[JK_DORY_MSM_BINS] = carry;
    }
}

kernel void jk_dory_msm_scatter(
    device const char* digits [[buffer(0)]],
    device atomic_uint* cursors [[buffer(1)]],
    device uint* order [[buffer(2)]],
    constant uint* params [[buffer(3)]],
    uint item [[thread_position_in_grid]])
{
    const uint len = params[0];
    const uint total = JK_DORY_MSM_WINDOWS * len;
    if (item >= total) {
        return;
    }
    const int digit = digits[item];
    if (digit == 0) {
        return;
    }
    const uint window = item / len;
    const uint index = item - window * len;
    const uint magnitude = (uint)(digit < 0 ? -digit : digit);
    const uint key = window * JK_DORY_MSM_BUCKETS + magnitude - 1u;
    const uint slot = atomic_fetch_add_explicit(&cursors[key], 1u, memory_order_relaxed);
    order[slot] = (index << 1u) | (digit < 0 ? 1u : 0u);
}

kernel void jk_g1_dory_msm_owner(
    device const uint* bases [[buffer(0)]],
    device const uint* order [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device uint* bucket_sums [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup G1Jac sums[128];
    const uint parts = params[0];
    const uint base_offset = params[1];
    const uint buckets_per_group = 128u / parts;
    const uint local_bucket = tid / parts;
    const uint part = tid % parts;
    const uint bucket = tg * buckets_per_group + local_bucket;
    G1Jac acc = g1_identity();
    if (bucket < JK_DORY_MSM_BINS) {
        const uint start = offsets[bucket];
        const uint end = offsets[bucket + 1u];
        for (uint i = start + part; i < end; i += parts) {
            const uint entry = order[i];
            G1Jac point = g1_load_jac(bases, base_offset + (entry >> 1u));
            if ((entry & 1u) != 0u) {
                point.y = fq_sub(fq_zero(), point.y);
            }
            acc = g1_add_jac(acc, point);
        }
    }
    sums[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = parts >> 1u; stride > 0u; stride >>= 1u) {
        if (part < stride) {
            sums[tid] = g1_add_jac(sums[tid], sums[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0u && bucket < JK_DORY_MSM_BINS) {
        g1_store_jac(bucket_sums + bucket * (3u * FR_LIMBS), sums[tid]);
    }
}

kernel void jk_g1_dory_msm_window_fold(
    device const uint* bucket_sums [[buffer(0)]],
    device uint* partials [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint window [[threadgroup_position_in_grid]])
{
    threadgroup G1Jac buckets[128];
    buckets[tid] = g1_load_jac(bucket_sums, window * JK_DORY_MSM_BUCKETS + tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < JK_DORY_MSM_BUCKETS; stride <<= 1u) {
        G1Jac addend;
        const bool active = tid + stride < JK_DORY_MSM_BUCKETS;
        if (active) {
            addend = buckets[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            buckets[tid] = g1_add_jac(buckets[tid], addend);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            buckets[tid] = g1_add_jac(buckets[tid], buckets[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        g1_store_jac(partials + window * (3u * FR_LIMBS), buckets[0]);
    }
}

// One thread per segment, accumulating in XYZZ to remove one field
// multiplication per mixed addition without changing host segmentation.
//   bases      — host G1Affine array (stride JK_G1_AFFINE_STRIDE u32s)
//   indices    — base indices, all segments concatenated; bit 31 selects
//                the NEGATED base (x, -y) — the signed-digit MSM entries of
//                the increment-column path (one-hot indices never set it)
//   seg_bounds — 3 u32s per segment: [start, end, out_slot]. Threads run
//                segments in the host's LENGTH-SORTED order (a simdgroup
//                finishes with its longest segment, so near-equal trip
//                counts per simdgroup ⇒ measured utilization 0.90 → 0.999
//                on production shape); `out_slot` restores bucket-walk
//                order for the host reducers.
//   out        — n_segs Jacobian points (3 * FR_LIMBS u32s each)
//   params[0]  — n_segs
kernel void jk_g1_seg_sum(
    device const uint* bases [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* seg_bounds [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_segs = params[0];
    if (tid >= n_segs) {
        return;
    }
    uint start = seg_bounds[3u * tid];
    uint end = seg_bounds[3u * tid + 1u];
    G1Xyzz acc = g1_xyzz_identity();
    for (uint i = start; i < end; i++) {
        uint raw = indices[i];
        G1AffinePt q = g1_load_base(bases, raw & 0x7fffffffu);
        if (raw >> 31) {
            // -(x, y) = (x, q - y); y is canonical nonzero (no base has
            // y = 0 on y^2 = x^3 + 3), so fq_sub yields the canonical
            // negation.
            q.y = fq_sub(fq_zero(), q.y);
        }
        acc = g1_xyzz_madd(acc, q);
    }
    g1_store_jac(out + seg_bounds[3u * tid + 2u] * (3u * FR_LIMBS), g1_xyzz_to_jac(acc));
}
