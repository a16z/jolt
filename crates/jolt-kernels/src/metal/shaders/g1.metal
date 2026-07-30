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

inline G1Jac g1_identity() {
    G1Jac p;
    p.x = fq_zero();
    p.y = fq_zero();
    p.z = fq_zero();
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
    uint start_bit;  // highest set bit across all scalars
};

kernel void jk_g1_combine_rows(
    device const uint* points [[buffer(0)]],   // affine, stride JK_G1_AFFINE_STRIDE
    device const uint* scalars [[buffer(1)]],  // num_hints × FR_LIMBS CANONICAL (integer) LE limbs
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
        uint word_index = (uint)bit >> 5;
        uint bit_index = (uint)bit & 31u;
        for (uint h = 0; h < p.num_hints; h++) {
            // Nonincreasing lens: every later hint is shorter too.
            if (gid >= lens[h]) {
                break;
            }
            if ((scalars[h * FR_LIMBS + word_index] >> bit_index) & 1u) {
                G1AffinePt q = g1_load_base(points, offsets[h] + gid);
                if (!(fq_is_zero(q.x) && fq_is_zero(q.y))) {
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

// One thread per segment: sum the selected bases into a Jacobian point.
//   bases      — host G1Affine array (stride JK_G1_AFFINE_STRIDE u32s)
//   indices    — base indices, all segments concatenated; bit 31 selects
//                the NEGATED base (x, -y) — the signed-digit MSM entries of
//                the increment-column path (one-hot indices never set it)
//   seg_starts — n_segs + 1 prefix offsets into `indices`
//   out        — n_segs Jacobian points (3 * FR_LIMBS u32s each)
//   params[0]  — n_segs
kernel void jk_g1_seg_sum(
    device const uint* bases [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* seg_starts [[buffer(2)]],
    device uint* out [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_segs = params[0];
    if (tid >= n_segs) {
        return;
    }
    uint start = seg_starts[tid];
    uint end = seg_starts[tid + 1];
    G1Jac acc = g1_identity();
    for (uint i = start; i < end; i++) {
        uint raw = indices[i];
        G1AffinePt q = g1_load_base(bases, raw & 0x7fffffffu);
        if (raw >> 31) {
            // -(x, y) = (x, q - y); y is canonical nonzero (no base has
            // y = 0 on y^2 = x^3 + 3), so fq_sub yields the canonical
            // negation.
            q.y = fq_sub(fq_zero(), q.y);
        }
        acc = g1_madd(acc, q);
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
