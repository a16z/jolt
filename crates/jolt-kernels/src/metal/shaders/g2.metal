// BN254 G2 point arithmetic over Fq2 = Fq[u]/(u² + 1), Montgomery limbs —
// the stage-8 reduce-round G2 fold kernels (W5a).
//
// Compiled behind the generated constants preamble (FQ_MOD / FQ_INV32 /
// FQ_ONE / JK_G2_AFFINE_STRIDE) after g1.metal, whose Fq256 CIOS primitives
// (fq_add / fq_sub / fq_dbl / fq_mul / fq_sqr) this file builds on.
//
// Fq2 multiplication is Karatsuba over the u² = −1 nonresidue (3 Fq muls):
//   c0 = a0·b0 − a1·b1,  c1 = (a0+a1)(b0+b1) − a0·b0 − a1·b1
// and squaring is the complex method (2 Fq muls):
//   c0 = (a0+a1)(a0−a1),  c1 = 2·a0·a1.
//
// Points are Jacobian (X, Y, Z) over Fq2 with the arkworks convention:
// affine (X/Z², Y/Z³), identity encoded as Z = 0. The twist has a = 0 (only
// b differs from G1, and b appears in no dbl/add formula), so the G1
// formulas carry over verbatim with Fq ops replaced by Fq2 ops:
// dbl-2009-l (2M + 5S) and madd-2007-bl (7M + 4S) with the same three
// special cases. A (0, 0) affine coordinate pair — b ≠ 0 keeps it off the
// twist — is the host's identity sentinel, exactly as in G1.

struct Fq2El {
    Fq256 c0;
    Fq256 c1;
};

inline Fq2El fq2_zero() {
    Fq2El r;
    r.c0 = fq_zero();
    r.c1 = fq_zero();
    return r;
}

inline bool fq2_is_zero(Fq2El a) {
    return fq_is_zero(a.c0) && fq_is_zero(a.c1);
}

inline Fq2El fq2_add(Fq2El a, Fq2El b) {
    Fq2El r;
    r.c0 = fq_add(a.c0, b.c0);
    r.c1 = fq_add(a.c1, b.c1);
    return r;
}

inline Fq2El fq2_sub(Fq2El a, Fq2El b) {
    Fq2El r;
    r.c0 = fq_sub(a.c0, b.c0);
    r.c1 = fq_sub(a.c1, b.c1);
    return r;
}

inline Fq2El fq2_dbl(Fq2El a) {
    return fq2_add(a, a);
}

// Karatsuba over u² = −1: 3 base-field muls.
inline Fq2El fq2_mul(Fq2El a, Fq2El b) {
    Fq256 v0 = fq_mul(a.c0, b.c0);
    Fq256 v1 = fq_mul(a.c1, b.c1);
    Fq2El r;
    r.c0 = fq_sub(v0, v1);
    r.c1 = fq_sub(fq_sub(fq_mul(fq_add(a.c0, a.c1), fq_add(b.c0, b.c1)), v0), v1);
    return r;
}

// Complex squaring: 2 base-field muls.
inline Fq2El fq2_sqr(Fq2El a) {
    Fq2El r;
    r.c0 = fq_mul(fq_add(a.c0, a.c1), fq_sub(a.c0, a.c1));
    r.c1 = fq_dbl(fq_mul(a.c0, a.c1));
    return r;
}

// Jacobian point; identity is Z = 0.
struct G2Jac {
    Fq2El x;
    Fq2El y;
    Fq2El z;
};

struct G2AffinePt {
    Fq2El x;
    Fq2El y;
};

inline G2Jac g2_identity() {
    G2Jac p;
    p.x = fq2_zero();
    p.y = fq2_zero();
    p.z = fq2_zero();
    return p;
}

inline Fq2El fq2_load(device const uint* src) {
    Fq2El r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.c0.v[i] = src[i];
    }
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.c1.v[i] = src[FR_LIMBS + i];
    }
    return r;
}

inline G2AffinePt g2_load_base(device const uint* bases, uint idx) {
    device const uint* src = bases + idx * JK_G2_AFFINE_STRIDE;
    G2AffinePt p;
    p.x = fq2_load(src);
    p.y = fq2_load(src + 2u * FR_LIMBS);
    return p;
}

inline G2Jac g2_load_jac(device const uint* points, uint idx) {
    device const uint* src = points + idx * (6u * FR_LIMBS);
    G2Jac p;
    p.x = fq2_load(src);
    p.y = fq2_load(src + 2u * FR_LIMBS);
    p.z = fq2_load(src + 4u * FR_LIMBS);
    return p;
}

inline Fq2El fq2_load_constant(constant uint* src) {
    Fq2El r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.c0.v[i] = src[i];
        r.c1.v[i] = src[FR_LIMBS + i];
    }
    return r;
}

inline Fq2El fq2_one() {
    Fq2El r;
    for (uint i = 0; i < FR_LIMBS; i++) {
        r.c0.v[i] = FQ_ONE[i];
    }
    r.c1 = fq_zero();
    return r;
}

// Doubling, a = 0 (dbl-2009-l over Fq2): 2M + 5S.
inline G2Jac g2_dbl(G2Jac p) {
    if (fq2_is_zero(p.z)) {
        return p;
    }
    Fq2El a = fq2_sqr(p.x);                // A = X²
    Fq2El b = fq2_sqr(p.y);                // B = Y²
    Fq2El c = fq2_sqr(b);                  // C = B²
    // D = 2((X + B)² - A - C)
    Fq2El d = fq2_sqr(fq2_add(p.x, b));
    d = fq2_sub(d, a);
    d = fq2_sub(d, c);
    d = fq2_dbl(d);
    Fq2El e = fq2_add(fq2_dbl(a), a);      // E = 3A
    Fq2El f = fq2_sqr(e);                  // F = E²
    G2Jac r;
    r.x = fq2_sub(fq2_sub(f, d), d);       // X3 = F - 2D
    Fq2El c8 = fq2_dbl(fq2_dbl(fq2_dbl(c))); // 8C
    r.y = fq2_sub(fq2_mul(e, fq2_sub(d, r.x)), c8);
    r.z = fq2_dbl(fq2_mul(p.y, p.z));      // Z3 = 2YZ
    return r;
}

// Mixed addition acc + (x2, y2) (madd-2007-bl over Fq2, 7M + 4S), affine
// point assumed NOT infinity, with the special cases handled exactly:
// identity accumulator (copy, Z = 1), equal points (double), inverse points
// (identity).
inline G2Jac g2_madd(G2Jac acc, G2AffinePt q) {
    if (fq2_is_zero(acc.z)) {
        G2Jac r;
        r.x = q.x;
        r.y = q.y;
        r.z = fq2_one();
        return r;
    }
    Fq2El z1z1 = fq2_sqr(acc.z);           // Z1Z1 = Z1²
    Fq2El u2 = fq2_mul(q.x, z1z1);         // U2 = X2·Z1Z1
    Fq2El s2 = fq2_mul(fq2_mul(q.y, acc.z), z1z1); // S2 = Y2·Z1·Z1Z1
    Fq2El h = fq2_sub(u2, acc.x);          // H = U2 - X1
    Fq2El rr = fq2_sub(s2, acc.y);         // r' = S2 - Y1 (halved r)
    if (fq2_is_zero(h)) {
        if (fq2_is_zero(rr)) {
            return g2_dbl(acc);
        }
        return g2_identity();
    }
    Fq2El r2 = fq2_dbl(rr);                // r = 2(S2 - Y1)
    Fq2El hh = fq2_sqr(h);                 // HH = H²
    Fq2El i = fq2_dbl(fq2_dbl(hh));        // I = 4HH
    Fq2El j = fq2_mul(h, i);               // J = H·I
    Fq2El v = fq2_mul(acc.x, i);           // V = X1·I
    G2Jac out;
    out.x = fq2_sub(fq2_sub(fq2_sqr(r2), j), fq2_dbl(v)); // X3 = r² - J - 2V
    Fq2El y1j = fq2_mul(acc.y, j);
    out.y = fq2_sub(fq2_mul(r2, fq2_sub(v, out.x)), fq2_dbl(y1j)); // Y3 = r(V-X3) - 2Y1·J
    // Z3 = (Z1 + H)² - Z1Z1 - HH
    out.z = fq2_sub(fq2_sub(fq2_sqr(fq2_add(acc.z, h)), z1z1), hh);
    return out;
}

inline G2Jac g2_add_jac(G2Jac p, G2Jac q) {
    if (fq2_is_zero(p.z)) {
        return q;
    }
    if (fq2_is_zero(q.z)) {
        return p;
    }
    Fq2El z1z1 = fq2_sqr(p.z);
    Fq2El z2z2 = fq2_sqr(q.z);
    Fq2El u1 = fq2_mul(p.x, z2z2);
    Fq2El u2 = fq2_mul(q.x, z1z1);
    Fq2El s1 = fq2_mul(p.y, fq2_mul(q.z, z2z2));
    Fq2El s2 = fq2_mul(q.y, fq2_mul(p.z, z1z1));
    Fq2El h = fq2_sub(u2, u1);
    Fq2El rr = fq2_sub(s2, s1);
    if (fq2_is_zero(h)) {
        return fq2_is_zero(rr) ? g2_dbl(p) : g2_identity();
    }
    Fq2El i = fq2_sqr(fq2_dbl(h));
    Fq2El j = fq2_mul(h, i);
    Fq2El r2 = fq2_dbl(rr);
    Fq2El v = fq2_mul(u1, i);
    G2Jac out;
    out.x = fq2_sub(fq2_sub(fq2_sqr(r2), j), fq2_dbl(v));
    out.y = fq2_sub(fq2_mul(r2, fq2_sub(v, out.x)), fq2_dbl(fq2_mul(s1, j)));
    out.z = fq2_mul(
        fq2_sub(fq2_sub(fq2_sqr(fq2_add(p.z, q.z)), z1z1), z2z2),
        h
    );
    return out;
}

inline void g2_store_jac(device uint* dst, G2Jac p) {
    Fq256 coords[6] = { p.x.c0, p.x.c1, p.y.c0, p.y.c1, p.z.c0, p.z.c1 };
    for (uint c = 0; c < 6; c++) {
        for (uint i = 0; i < FR_LIMBS; i++) {
            dst[c * FR_LIMBS + i] = coords[c].v[i];
        }
    }
}

// out[i] = s·P[i] + Q[i], one scalar shared by the whole vector — the G2
// twin of jk_g1_scalar_mul_add, accelerated by the 4-GLV ψ-decomposition:
// the host splits s = Σ_k ±s_k·λ^k (|s_k| ≲ 2^64) and expands P into four
// ψ^k-transformed, sign-folded affine base vectors (block k at k·n), so
// the ladder sweeps only max_bits(s_k) ≈ 64 iterations — 4× fewer
// doublings than a 254-bit ladder, the dominant term of the per-lane
// latency floor. Bit branches stay warp-uniform (subscalars are shared by
// every lane); the identity sentinel survives ψ (it scales coordinates),
// so dead bases still contribute nothing.
struct G2MulAddParams {
    uint n;
    uint start_bit;               // highest set bit across the subscalars
    uint coeffs[4u * FR_LIMBS];   // CANONICAL LE limbs of |s_0|‖|s_1|‖|s_2|‖|s_3|
};

kernel void jk_g2_scalar_mul_add(
    device const uint* ps [[buffer(0)]],   // 4n affine bases, ψ^k block k at k·n
    device const uint* qs [[buffer(1)]],   // n affine addends
    device uint* out [[buffer(2)]],        // n Jacobian results (6·FR_LIMBS u32s)
    constant G2MulAddParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    G2Jac acc = g2_identity();
    for (int bit = (int)p.start_bit; bit >= 0; bit--) {
        acc = g2_dbl(acc);
        for (uint k = 0; k < 4u; k++) {
            if ((p.coeffs[k * FR_LIMBS + ((uint)bit >> 5)] >> ((uint)bit & 31u)) & 1u) {
                G2AffinePt base = g2_load_base(ps, k * p.n + tid);
                if (!(fq2_is_zero(base.x) && fq2_is_zero(base.y))) {
                    acc = g2_madd(acc, base);
                }
            }
        }
    }
    G2AffinePt addend = g2_load_base(qs, tid);
    if (!(fq2_is_zero(addend.x) && fq2_is_zero(addend.y))) {
        acc = g2_madd(acc, addend);
    }
    g2_store_jac(out + tid * (6u * FR_LIMBS), acc);
}

struct G2ProjectiveMulAddParams {
    uint n;
    uint p_offset;
    uint q_offset;
    int digits[66];
    uint endo[2u * FR_LIMBS];
};

kernel void jk_g2_projective_mul_add(
    device const uint* ps [[buffer(0)]],
    device const uint* qs [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant G2ProjectiveMulAddParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    G2Jac point = g2_load_jac(ps, p.p_offset + tid);
    G2Jac table[8];
    table[0] = point;
    table[1] = g2_dbl(point);
    for (uint k = 2u; k < 8u; k++) {
        table[k] = g2_add_jac(table[k - 1u], point);
    }
    Fq2El endo = fq2_load_constant(p.endo);
    G2Jac acc = g2_identity();
    for (int w = 32; w >= 0; w--) {
        if (w != 32) {
            for (uint k = 0u; k < 4u; k++) {
                acc = g2_dbl(acc);
            }
        }
        int d1 = p.digits[w];
        if (d1 != 0) {
            uint magnitude = (uint)(d1 < 0 ? -d1 : d1);
            G2Jac entry = table[magnitude - 1u];
            if (d1 < 0) {
                entry.y = fq2_sub(fq2_zero(), entry.y);
            }
            acc = g2_add_jac(acc, entry);
        }
        int d2 = p.digits[33 + w];
        if (d2 != 0) {
            uint magnitude = (uint)(d2 < 0 ? -d2 : d2);
            G2Jac entry = table[magnitude - 1u];
            entry.x = fq2_mul(entry.x, endo);
            if (d2 < 0) {
                entry.y = fq2_sub(fq2_zero(), entry.y);
            }
            acc = g2_add_jac(acc, entry);
        }
    }
    acc = g2_add_jac(acc, g2_load_jac(qs, p.q_offset + tid));
    g2_store_jac(out + tid * (6u * FR_LIMBS), acc);
}

kernel void jk_g2_dory_msm_owner(
    device const uint* bases [[buffer(0)]],
    device const uint* order [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device uint* bucket_sums [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup G2Jac sums[128];
    const uint parts = params[0];
    const uint base_offset = params[1];
    const uint buckets_per_group = 128u / parts;
    const uint local_bucket = tid / parts;
    const uint part = tid % parts;
    const uint bucket = tg * buckets_per_group + local_bucket;
    G2Jac acc = g2_identity();
    if (bucket < JK_DORY_MSM_BINS) {
        const uint start = offsets[bucket];
        const uint end = offsets[bucket + 1u];
        for (uint i = start + part; i < end; i += parts) {
            const uint entry = order[i];
            G2Jac point = g2_load_jac(bases, base_offset + (entry >> 1u));
            if ((entry & 1u) != 0u) {
                point.y = fq2_sub(fq2_zero(), point.y);
            }
            acc = g2_add_jac(acc, point);
        }
    }
    sums[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = parts >> 1u; stride > 0u; stride >>= 1u) {
        if (part < stride) {
            sums[tid] = g2_add_jac(sums[tid], sums[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0u && bucket < JK_DORY_MSM_BINS) {
        g2_store_jac(bucket_sums + bucket * (6u * FR_LIMBS), sums[tid]);
    }
}

kernel void jk_g2_dory_msm_window_fold(
    device const uint* bucket_sums [[buffer(0)]],
    device uint* partials [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint window [[threadgroup_position_in_grid]])
{
    threadgroup G2Jac buckets[128];
    buckets[tid] = g2_load_jac(bucket_sums, window * JK_DORY_MSM_BUCKETS + tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < JK_DORY_MSM_BUCKETS; stride <<= 1u) {
        G2Jac addend;
        const bool active = tid + stride < JK_DORY_MSM_BUCKETS;
        if (active) {
            addend = buckets[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            buckets[tid] = g2_add_jac(buckets[tid], addend);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            buckets[tid] = g2_add_jac(buckets[tid], buckets[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        g2_store_jac(partials + window * (6u * FR_LIMBS), buckets[0]);
    }
}

// out[i] = base · scalars[i] via a host-built window table
// (table[win·15 + d − 1] = d·16^win·base, affine, d = 1..15): each thread
// sums its scalar's nonzero nibbles with mixed adds only — no doublings,
// the plain ladder's dominant term. Window order is irrelevant to the sum,
// so the walk is low-to-high; add branches diverge per thread (scalars
// differ), and table entries are never the identity (d·16^win < r, base
// nonzero), so no sentinel check is needed. g2_madd's exact equal/inverse
// branches cover colliding partial sums.
struct G2FixedBaseTableParams {
    uint n;
    uint windows;                 // nibble windows encoded in the table
};

kernel void jk_g2_fixed_base_table(
    device const uint* scalars [[buffer(0)]], // n × FR_LIMBS CANONICAL LE limbs
    device const uint* table [[buffer(1)]],   // windows × 15 affine entries
    device uint* out [[buffer(2)]],           // n Jacobian results
    constant G2FixedBaseTableParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    device const uint* s = scalars + tid * FR_LIMBS;
    G2Jac acc = g2_identity();
    for (uint win = 0; win < p.windows; win++) {
        uint d = (s[win >> 3u] >> ((win & 7u) * 4u)) & 0xfu;
        if (d != 0u) {
            acc = g2_madd(acc, g2_load_base(table, win * 15u + d - 1u));
        }
    }
    g2_store_jac(out + tid * (6u * FR_LIMBS), acc);
}

// out[i] = base · scalars[i]: ONE shared affine base (never the sentinel;
// the host handles an identity base without dispatching), one scalar per
// thread — dory-pcs's v₂ = v_vec · Γ2,fin construction. The ladder sweeps
// from the host-computed maximum start bit; add branches diverge per thread
// (scalars differ), doublings stay uniform.
struct G2FixedBaseParams {
    uint n;
    uint start_bit;               // max highest set bit across all scalars
    uint base[4u * FR_LIMBS];     // affine x, y (Fq2 c0 ‖ c1 each), Montgomery
};

kernel void jk_g2_fixed_base_mul(
    device const uint* scalars [[buffer(0)]], // n × FR_LIMBS CANONICAL LE limbs
    device uint* out [[buffer(1)]],           // n Jacobian results
    constant G2FixedBaseParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= p.n) {
        return;
    }
    G2AffinePt base;
    for (uint i = 0; i < FR_LIMBS; i++) {
        base.x.c0.v[i] = p.base[i];
        base.x.c1.v[i] = p.base[FR_LIMBS + i];
        base.y.c0.v[i] = p.base[2u * FR_LIMBS + i];
        base.y.c1.v[i] = p.base[3u * FR_LIMBS + i];
    }
    device const uint* s = scalars + tid * FR_LIMBS;
    G2Jac acc = g2_identity();
    for (int bit = (int)p.start_bit; bit >= 0; bit--) {
        acc = g2_dbl(acc);
        if ((s[(uint)bit >> 5] >> ((uint)bit & 31u)) & 1u) {
            acc = g2_madd(acc, base);
        }
    }
    g2_store_jac(out + tid * (6u * FR_LIMBS), acc);
}
