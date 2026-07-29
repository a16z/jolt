// Compute kernels over BN254 Fr arrays (see fr.metal for the arithmetic).
//
// Buffer convention (mirrored by `metal::runtime::ComputePass::dispatch`):
// data buffers occupy [[buffer(0..k)]] in argument order, the parameter
// struct is bound with setBytes at the next index. All parameter structs are
// sequences of `uint` (field elements as FR_LIMBS little-endian u32 limbs),
// so the Rust side builds them as flat `[u32]` slices with no padding.

struct ElemwiseParams {
    uint n;
};

struct PowParams {
    uint n;
    uint k;
};

struct BindParams {
    uint n_out;
    uint r[FR_LIMBS];
};

struct BindEvalParams {
    uint n_out;
    uint num_points;
    uint num_tgs;
    uint r[FR_LIMBS];
    uint points[FR_LIMBS * JK_MAX_EVAL_POINTS];
};

// Empty kernel for dispatch-latency measurement.
kernel void jk_noop() {}

kernel void jk_fr_mul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant ElemwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    fr_store(out, gid, fr_mont_mul(fr_load(a, gid), fr_load(b, gid)));
}

kernel void jk_fr_add(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant ElemwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    fr_store(out, gid, fr_add(fr_load(a, gid), fr_load(b, gid)));
}

kernel void jk_fr_sub(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant ElemwiseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    fr_store(out, gid, fr_sub(fr_load(a, gid), fr_load(b, gid)));
}

// out[i] = a[i]^(2^k): k dependent Montgomery squarings per element.
// Compute-bound probe — one load, one store, k muls.
kernel void jk_fr_pow2k(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant PowParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    Fr256 x = fr_load(a, gid);
    for (uint i = 0; i < p.k; i++) {
        x = fr_mont_mul(x, x);
    }
    fr_store(out, gid, x);
}

// The sumcheck fold primitive: out[i] = a[2i] + r·(a[2i+1] − a[2i]).
kernel void jk_fr_bind(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant BindParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n_out) {
        return;
    }
    Fr256 lo = fr_load(a, 2 * gid);
    Fr256 diff = fr_sub(fr_load(a, 2 * gid + 1), lo);
    Fr256 r = fr_load_const(p.r, 0);
    fr_store(out, gid, fr_add(lo, fr_mont_mul(r, diff)));
}

// --- Slot-round machinery -------------------------------------------------
//
// The W2 slot kernels fuse one sumcheck round into a single dispatch: bind
// the previous round's challenge out of place (cur → nxt, so there is no
// intra-dispatch hazard) and accumulate the NEW table's round-poly
// evaluations as per-threadgroup partial sums the host finishes. Round 0 has
// no challenge yet (do_bind = 0): pairs are read straight from cur and nxt
// is untouched (callers may bind any buffer there).

// Threadgroup tree reduction of one value per lane into
// partials[slot * num_tgs + tg]. Every lane must call this (inactive lanes
// pass zero) — it barriers. The trailing barrier makes `scratch` reusable by
// a following call.
inline void jk_tg_sum(threadgroup uint* scratch, uint lid, uint tg, Fr256 v,
                      device uint* partials, uint slot, uint num_tgs)
{
    fr_store_tg(scratch, lid, v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = JK_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            Fr256 s = fr_add(fr_load_tg(scratch, lid), fr_load_tg(scratch, lid + stride));
            fr_store_tg(scratch, lid, s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        fr_store(partials, slot * num_tgs + tg, fr_load_tg(scratch, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One table's (lo, hi) sumcheck pair at post-bind group y. Binding reads the
// quad cur[4y..4y+4], folds both pairs with r (low-to-high bind), and writes
// the results to nxt[2y], nxt[2y+1]; without binding the pair is cur's.
// Inactive lanes produce zeros and touch no memory.
inline void jk_round_pair(device const uint* cur, device uint* nxt, bool bind,
                          Fr256 r, uint y, bool active,
                          thread Fr256& lo, thread Fr256& hi)
{
    if (!active) {
        lo = fr_zero();
        hi = fr_zero();
        return;
    }
    if (bind) {
        Fr256 a = fr_load(cur, 4 * y);
        Fr256 b = fr_load(cur, 4 * y + 1);
        Fr256 c = fr_load(cur, 4 * y + 2);
        Fr256 d = fr_load(cur, 4 * y + 3);
        lo = fr_add(a, fr_mont_mul(r, fr_sub(b, a)));
        hi = fr_add(c, fr_mont_mul(r, fr_sub(d, c)));
        fr_store(nxt, 2 * y, lo);
        fr_store(nxt, 2 * y + 1, hi);
    } else {
        lo = fr_load(cur, 2 * y);
        hi = fr_load(cur, 2 * y + 1);
    }
}

// 2·hi − lo: a degree-2 summand's table evaluation at t = 2.
inline Fr256 jk_at_two(Fr256 lo, Fr256 hi)
{
    return fr_sub(fr_add(hi, hi), lo);
}

// Shared parameter head of the slot-round kernels ("groups" because `half`
// is an MSL type name).
struct SlotRoundParams {
    uint groups;   // post-bind group count = active threads
    uint do_bind;  // 1: fold cur→nxt with r first; 0: round 0, read cur
    uint num_tgs;  // partials stride (threadgroup count of this dispatch)
    uint r[FR_LIMBS];
};

// Stage-6b increment claim reduction round: summand
// A·RamInc + B·RdInc over four same-length tables. Partial sums of the
// summand at t ∈ {0, 2}: partials[0·num_tgs..] = s(0), partials[1·num_tgs..]
// = s(2).
kernel void jk_inc_round(
    device const uint* ram_cur [[buffer(0)]],
    device const uint* rd_cur [[buffer(1)]],
    device const uint* aw_cur [[buffer(2)]],
    device const uint* bw_cur [[buffer(3)]],
    device uint* ram_nxt [[buffer(4)]],
    device uint* rd_nxt [[buffer(5)]],
    device uint* aw_nxt [[buffer(6)]],
    device uint* bw_nxt [[buffer(7)]],
    device uint* partials [[buffer(8)]],
    constant SlotRoundParams& p [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0;
    Fr256 r = fr_load_const(p.r, 0);

    Fr256 ram_lo, ram_hi, rd_lo, rd_hi, a_lo, a_hi, b_lo, b_hi;
    jk_round_pair(ram_cur, ram_nxt, bind, r, gid, active, ram_lo, ram_hi);
    jk_round_pair(rd_cur, rd_nxt, bind, r, gid, active, rd_lo, rd_hi);
    jk_round_pair(aw_cur, aw_nxt, bind, r, gid, active, a_lo, a_hi);
    jk_round_pair(bw_cur, bw_nxt, bind, r, gid, active, b_lo, b_hi);

    Fr256 s0 = fr_add(fr_mont_mul(a_lo, ram_lo), fr_mont_mul(b_lo, rd_lo));
    Fr256 s2 = fr_add(fr_mont_mul(jk_at_two(a_lo, a_hi), jk_at_two(ram_lo, ram_hi)),
                      fr_mont_mul(jk_at_two(b_lo, b_hi), jk_at_two(rd_lo, rd_hi)));
    jk_tg_sum(scratch, lid, tg, s0, partials, 0, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, s2, partials, 1, p.num_tgs);
}

// Stage-7 Hamming-weight claim reduction round: summand Σ_i G_i·W_i over N
// table pairs, each pair concatenated into g/w at stride `len` (current
// per-table length). Thread (i, y) is flat gid = i·groups_per_table + y with
// groups_per_table = 1 << log_h; partial sums cross table boundaries freely
// (the host sums every partial anyway).
struct TablePairsParams {
    uint log_h;       // log2(per-table post-bind group count)
    uint num_tables;  // N
    uint len;         // per-table CURRENT (pre-bind) length
    uint do_bind;
    uint num_tgs;
    uint r[FR_LIMBS];
};

kernel void jk_table_pairs_round(
    device const uint* g_cur [[buffer(0)]],
    device const uint* w_cur [[buffer(1)]],
    device uint* g_nxt [[buffer(2)]],
    device uint* w_nxt [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant TablePairsParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    uint h = 1u << p.log_h;
    bool active = gid < p.num_tables * h;
    bool bind = p.do_bind != 0;
    Fr256 r = fr_load_const(p.r, 0);

    // Clamp inactive lanes' table index so their (unused) pointer math stays
    // in bounds.
    uint i = min(gid >> p.log_h, p.num_tables - 1);
    uint y = gid & (h - 1u);
    uint cur_base = i * p.len * FR_LIMBS;
    uint nxt_base = i * (p.len >> 1) * FR_LIMBS;

    Fr256 g_lo, g_hi, w_lo, w_hi;
    jk_round_pair(g_cur + cur_base, g_nxt + nxt_base, bind, r, y, active, g_lo, g_hi);
    jk_round_pair(w_cur + cur_base, w_nxt + nxt_base, bind, r, y, active, w_lo, w_hi);

    Fr256 s0 = fr_mont_mul(g_lo, w_lo);
    Fr256 s2 = fr_mont_mul(jk_at_two(g_lo, g_hi), jk_at_two(w_lo, w_hi));
    jk_tg_sum(scratch, lid, tg, s0, partials, 0, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, s2, partials, 1, p.num_tgs);
}

// Stage-6b RAM Hamming booleanity round: Gruen split-eq inner accumulators
// over the dense Hamming table H. Per group y the quadratic q(t) =
// (h₀ + t·(h₁−h₀))² − (h₀ + t·(h₁−h₀)) is characterized by its constant
// h₀²−h₀ and leading (h₁−h₀)² coefficients, weighted by
// eq(y) = e_out[y >> log_in]·e_in[y & (in_len−1)]. Partials: slot 0 the
// constant sum, slot 1 the leading sum; the host assembles the Gruen cubic.
struct HammingParams {
    uint groups;
    uint do_bind;
    uint num_tgs;
    uint log_in;  // log2(e_in length)
    uint r[FR_LIMBS];
};

kernel void jk_hamming_round(
    device const uint* h_cur [[buffer(0)]],
    device uint* h_nxt [[buffer(1)]],
    device const uint* e_in [[buffer(2)]],
    device const uint* e_out [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant HammingParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0;
    Fr256 r = fr_load_const(p.r, 0);

    Fr256 h0, h1;
    jk_round_pair(h_cur, h_nxt, bind, r, gid, active, h0, h1);

    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    Fr256 q0 = fr_sub(fr_mont_mul(h0, h0), h0);
    Fr256 delta = fr_sub(h1, h0);
    Fr256 q_lead = fr_mont_mul(delta, delta);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, q0), partials, 0, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, q_lead), partials, 1, p.num_tgs);
}

// Fused fold + round-poly evaluation: the bind above, plus per-threadgroup
// partial sums of v_j(i) = a[2i] + t_j·(a[2i+1] − a[2i]) for each runtime
// eval point t_j. One kernel serves every point set — t is data, so the
// shape stays uniform (t = 0 still runs its multiply). Partials are written
// point-major: partials[j * num_tgs + tg], reduced on the host.
kernel void jk_fr_bind_eval(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    device uint* partials [[buffer(2)]],
    constant BindEvalParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];

    // Inactive lanes contribute zero to every partial sum but still execute
    // the uniform arithmetic below (no early return: they must reach the
    // threadgroup barriers).
    bool active = gid < p.n_out;
    Fr256 lo = fr_zero();
    Fr256 diff = fr_zero();
    if (active) {
        lo = fr_load(a, 2 * gid);
        diff = fr_sub(fr_load(a, 2 * gid + 1), lo);
        Fr256 r = fr_load_const(p.r, 0);
        fr_store(out, gid, fr_add(lo, fr_mont_mul(r, diff)));
    }

    for (uint j = 0; j < p.num_points; j++) {
        Fr256 t = fr_load_const(p.points, j);
        Fr256 v = fr_add(lo, fr_mont_mul(t, diff));
        fr_store_tg(scratch, lid, v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = JK_TG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Fr256 s = fr_add(fr_load_tg(scratch, lid), fr_load_tg(scratch, lid + stride));
                fr_store_tg(scratch, lid, s);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) {
            fr_store(partials, j * p.num_tgs + tg, fr_load_tg(scratch, 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
