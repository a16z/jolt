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

struct Bind4Params {
    uint n_out;
    uint l1[FR_LIMBS];
    uint l2[FR_LIMBS];
    uint l3[FR_LIMBS];
};

struct BindEvalParams {
    uint n_out;
    uint num_points;
    uint num_tgs;
    uint r[FR_LIMBS];
    uint points[FR_LIMBS * JK_MAX_EVAL_POINTS];
};

struct IncPrepareParams {
    uint n;
    uint low_bits;
    uint low_mask;
    uint offsets[8];
    uint gamma[FR_LIMBS];
    uint gamma2[FR_LIMBS];
    uint gamma3[FR_LIMBS];
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

// Lagrange weights sum to one, so an affine a[4i] base matches two binary
// binds' three Montgomery products without their intermediate table pass.
kernel void jk_fr_bind4(
    device const uint* a [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant Bind4Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n_out) {
        return;
    }
    Fr256 a0 = fr_load(a, 4 * gid);
    Fr256 result = a0;
    result = fr_add(result, fr_mont_mul(fr_load_const(p.l1, 0),
                                       fr_sub(fr_load(a, 4 * gid + 1), a0)));
    result = fr_add(result, fr_mont_mul(fr_load_const(p.l2, 0),
                                       fr_sub(fr_load(a, 4 * gid + 2), a0)));
    result = fr_add(result, fr_mont_mul(fr_load_const(p.l3, 0),
                                       fr_sub(fr_load(a, 4 * gid + 3), a0)));
    fr_store(out, gid, result);
}

// Direct paired weights avoid four T-sized eq intermediates; balanced
// factors keep the upload at O(sqrt(T)).
kernel void jk_inc_prepare(
    device const uint* factors [[buffer(0)]],
    device uint* ram_weights [[buffer(1)]],
    device uint* rd_weights [[buffer(2)]],
    constant IncPrepareParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) {
        return;
    }
    uint high = gid >> p.low_bits;
    uint low = gid & p.low_mask;
    Fr256 eq[4];
    for (uint point = 0; point < 4u; point++) {
        Fr256 high_factor = fr_load(factors, p.offsets[2u * point] + high);
        Fr256 low_factor = fr_load(factors, p.offsets[2u * point + 1u] + low);
        eq[point] = fr_mont_mul(high_factor, low_factor);
    }
    Fr256 gamma = fr_load_const(p.gamma, 0);
    Fr256 gamma2 = fr_load_const(p.gamma2, 0);
    Fr256 gamma3 = fr_load_const(p.gamma3, 0);
    fr_store(ram_weights, gid, fr_add(eq[0], fr_mont_mul(gamma, eq[1])));
    fr_store(rd_weights, gid, fr_add(fr_mont_mul(gamma2, eq[2]),
                                    fr_mont_mul(gamma3, eq[3])));
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

// Stage-5 registers value-evaluation round: cubic summand Inc·Wa·LT over the
// cycle variables, sampled at t ∈ {0, 2, 3} (s(1) comes from the engine
// hint). Wa is served lazily from the per-cycle rd indices and the address eq
// table until its first bind densifies it into wa_nxt (wa_dense 0 → 1). LT
// stays in the optimized tier's split form — the host binds the ~√T lo table
// before each dispatch and the kernel serves
// lt(j) = lt_hi[j >> log_lo] + eq_hi[j >> log_lo]·lt_lo[j & mask] — until the
// lo variables are exhausted (lt_dense = 1, dense table in lt_lo), which in
// production happens below the device gate.
struct RegistersValRoundParams {
    uint groups;    // post-bind pair count = active threads
    uint do_bind;   // 1: fold cur→nxt with r first; 0: first round, read cur
    uint num_tgs;
    uint wa_dense;  // 0: wa from rd bytes + eq_address; 1: wa from wa_cur
    uint lt_dense;  // 1: LT served densely from lt_lo
    uint log_lo;    // log2(lt_lo length) after this round's host-side LT bind
    uint r[FR_LIMBS];
};

// wa(j) before densification: eq_address[rd[j]], zero on no-write cycles
// (0xFF sentinel byte).
inline Fr256 jk_registers_wa(device const uint* rd, device const uint* eq_address, uint j)
{
    uint reg = (rd[j >> 2] >> ((j & 3u) * 8u)) & 0xFFu;
    if (reg == 0xFFu) {
        return fr_zero();
    }
    return fr_load(eq_address, reg);
}

kernel void jk_registers_val_round(
    device const uint* inc_cur [[buffer(0)]],
    device uint* inc_nxt [[buffer(1)]],
    device const uint* wa_cur [[buffer(2)]],
    device uint* wa_nxt [[buffer(3)]],
    device const uint* rd [[buffer(4)]],
    device const uint* eq_address [[buffer(5)]],
    device const uint* lt_lo [[buffer(6)]],
    device const uint* lt_hi [[buffer(7)]],
    device const uint* eq_hi [[buffer(8)]],
    device uint* partials [[buffer(9)]],
    constant RegistersValRoundParams& p [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0u;
    Fr256 r = fr_load_const(p.r, 0);

    Fr256 inc_0, inc_1;
    jk_round_pair(inc_cur, inc_nxt, bind, r, gid, active, inc_0, inc_1);

    Fr256 wa_0 = fr_zero();
    Fr256 wa_1 = fr_zero();
    if (p.wa_dense != 0u) {
        jk_round_pair(wa_cur, wa_nxt, bind, r, gid, active, wa_0, wa_1);
    } else if (active) {
        if (bind) {
            // First bind: densify straight from the indices — the K × T grid
            // never exists on either tier.
            Fr256 a = jk_registers_wa(rd, eq_address, 4u * gid);
            Fr256 b = jk_registers_wa(rd, eq_address, 4u * gid + 1u);
            Fr256 c = jk_registers_wa(rd, eq_address, 4u * gid + 2u);
            Fr256 d = jk_registers_wa(rd, eq_address, 4u * gid + 3u);
            wa_0 = fr_add(a, fr_mont_mul(r, fr_sub(b, a)));
            wa_1 = fr_add(c, fr_mont_mul(r, fr_sub(d, c)));
            fr_store(wa_nxt, 2u * gid, wa_0);
            fr_store(wa_nxt, 2u * gid + 1u, wa_1);
        } else {
            wa_0 = jk_registers_wa(rd, eq_address, 2u * gid);
            wa_1 = jk_registers_wa(rd, eq_address, 2u * gid + 1u);
        }
    }

    Fr256 lt_0 = fr_zero();
    Fr256 lt_1 = fr_zero();
    if (active) {
        uint j = 2u * gid;
        if (p.lt_dense != 0u) {
            lt_0 = fr_load(lt_lo, j);
            lt_1 = fr_load(lt_lo, j + 1u);
        } else {
            // Adjacent lo indices share the hi part (lo_len ≥ 2).
            uint hi = j >> p.log_lo;
            uint mask = (1u << p.log_lo) - 1u;
            Fr256 base = fr_load(lt_hi, hi);
            Fr256 scale = fr_load(eq_hi, hi);
            lt_0 = fr_add(base, fr_mont_mul(scale, fr_load(lt_lo, j & mask)));
            lt_1 = fr_add(base, fr_mont_mul(scale, fr_load(lt_lo, (j + 1u) & mask)));
        }
    }

    Fr256 inc_m = fr_sub(inc_1, inc_0);
    Fr256 wa_m = fr_sub(wa_1, wa_0);
    Fr256 lt_m = fr_sub(lt_1, lt_0);
    Fr256 inc_2 = fr_add(inc_1, inc_m);
    Fr256 wa_2 = fr_add(wa_1, wa_m);
    Fr256 lt_2 = fr_add(lt_1, lt_m);
    Fr256 s0 = fr_mont_mul(fr_mont_mul(inc_0, wa_0), lt_0);
    Fr256 s2 = fr_mont_mul(fr_mont_mul(inc_2, wa_2), lt_2);
    Fr256 s3 = fr_mont_mul(fr_mont_mul(fr_add(inc_2, inc_m), fr_add(wa_2, wa_m)),
                           fr_add(lt_2, lt_m));
    jk_tg_sum(scratch, lid, tg, s0, partials, 0, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, s2, partials, 1, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, s3, partials, 2, p.num_tgs);
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

// --- Stage-8 joint-opening fold (W3b) ---------------------------------------
//
// The batch opening's one vector-matrix product, distributed per committed
// trace polynomial: result_p[c] = Σ_r left[r]·M_p[r][c] where M_p places one
// coefficient per cycle at grid index cycle + address·2^log_T (cycle-major).
// With σ ≤ log_T the column of every cycle is its own low σ bits, so thread
// c owns output column c outright — no scatter, no atomics: it walks cycles
// c, c+2^σ, c+2·2^σ, … (adjacent threads read adjacent cycles — coalesced)
// and gathers left[j + address·2^(log_T−σ)] per polynomial. Accumulation
// order per column is ascending-cycle; Fr addition is exact, so any
// regrouping against the CPU's range-partial order is byte-identical.

// value·R mod p for a two's-complement i128 given as 4 LE u32 words:
// magnitude < 2^128 < p is already canonical, one mont_mul by R² lifts it to
// Montgomery form, and p − x (branchless in fr_sub) applies the sign.
inline Fr256 fr_from_i128(uint w0, uint w1, uint w2, uint w3) {
    uint neg = w3 >> 31;
    if (neg != 0u) {
        ulong carry = 1;
        ulong s0 = (ulong)(~w0) + carry; w0 = (uint)s0; carry = s0 >> 32;
        ulong s1 = (ulong)(~w1) + carry; w1 = (uint)s1; carry = s1 >> 32;
        ulong s2 = (ulong)(~w2) + carry; w2 = (uint)s2; carry = s2 >> 32;
        ulong s3 = (ulong)(~w3) + carry; w3 = (uint)s3;
    }
    Fr256 x = fr_zero();
    x.v[0] = w0;
    x.v[1] = w1;
    x.v[2] = w2;
    x.v[3] = w3;
    Fr256 r = fr_mont_mul(x, fr_load_const(FR_R2, 0));
    if (neg != 0u) {
        r = fr_sub(fr_zero(), r);
    }
    return r;
}

struct OpeningFoldDenseParams {
    uint sigma;  // log2(output columns) — one thread per column
    uint steps;  // cycles per thread = T >> sigma
};

// One dense increment column (i128 per cycle, address slot 0):
// out[c] = Σ_j left[j] · value(cycle c + j·2^σ). Zero increments are skipped
// exactly like the CPU's entry() — adding zero is the same sum either way.
kernel void jk_opening_fold_dense(
    device const uint* col [[buffer(0)]],   // 4 u32 words per cycle (LE i128)
    device const uint* left [[buffer(1)]],  // Fr per row
    device uint* out [[buffer(2)]],         // Fr per column
    constant OpeningFoldDenseParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (1u << p.sigma)) {
        return;
    }
    Fr256 acc = fr_zero();
    for (uint j = 0; j < p.steps; j++) {
        uint cycle = gid + (j << p.sigma);
        uint w0 = col[4 * cycle];
        uint w1 = col[4 * cycle + 1];
        uint w2 = col[4 * cycle + 2];
        uint w3 = col[4 * cycle + 3];
        if ((w0 | w1 | w2 | w3) == 0u) {
            continue;
        }
        Fr256 v = fr_from_i128(w0, w1, w2, w3);
        acc = fr_add(acc, fr_mont_mul(fr_load(left, j), v));
    }
    fr_store(out, gid, acc);
}

struct OpeningFoldOneHotParams {
    uint sigma;       // log2(output columns) — one thread per column
    uint steps;       // cycles per thread = T >> sigma
    uint row_shift;   // log_T − σ: row = j + (address << row_shift)
    uint elem_words;  // u32 words per column element: 2 (u64) or 4 (u128)
    uint has_cold;    // 1: an all-ones element is a cold cycle (no entry)
    uint sel_count;   // live selectors ≤ JK_OPENING_MAX_SEL
    uint sel_mask;    // (1 << chunk_bits) − 1, chunk_bits < 32
    uint sel_shift[JK_OPENING_MAX_SEL];  // per-selector bit offset
};

// Extract the chunk at bit offset `shift` (width < 32) from up to 4 LE words.
inline uint jk_chunk_bits(uint w0, uint w1, uint w2, uint w3,
                          uint shift, uint mask)
{
    uint ws[4] = { w0, w1, w2, w3 };
    uint word = shift >> 5;
    uint bit = shift & 31u;
    uint lo = ws[word] >> bit;
    // Chunks may straddle a word boundary; shifts of 32 are avoided (MSL
    // shift counts are modular).
    if (bit != 0u && word + 1u < 4u) {
        lo |= ws[word + 1u] << (32u - bit);
    }
    return lo & mask;
}

// A one-hot family's columns from one shared per-cycle stream (lookup index,
// mapped pc, or remapped RAM address): out[s·2^σ + c] accumulates selector
// s's fold, each selector reading its own chunk of the element as the hot
// address. Cold cycles (all-ones sentinel) contribute nothing.
kernel void jk_opening_fold_onehot(
    device const uint* col [[buffer(0)]],   // elem_words u32 per cycle
    device const uint* left [[buffer(1)]],  // Fr per row
    device uint* out [[buffer(2)]],         // Fr per (selector, column)
    constant OpeningFoldOneHotParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (1u << p.sigma)) {
        return;
    }
    Fr256 acc[JK_OPENING_MAX_SEL];
    for (uint s = 0; s < JK_OPENING_MAX_SEL; s++) {
        acc[s] = fr_zero();
    }
    for (uint j = 0; j < p.steps; j++) {
        uint cycle = gid + (j << p.sigma);
        uint base = p.elem_words * cycle;
        uint w0 = col[base];
        uint w1 = col[base + 1];
        uint w2 = p.elem_words > 2u ? col[base + 2] : 0u;
        uint w3 = p.elem_words > 2u ? col[base + 3] : 0u;
        if (p.has_cold != 0u && (w0 & w1) == 0xffffffffu) {
            continue;
        }
        for (uint s = 0; s < JK_OPENING_MAX_SEL; s++) {
            if (s >= p.sel_count) {
                break;
            }
            uint address = jk_chunk_bits(w0, w1, w2, w3, p.sel_shift[s], p.sel_mask);
            uint row = j + (address << p.row_shift);
            acc[s] = fr_add(acc[s], fr_load(left, row));
        }
    }
    for (uint s = 0; s < JK_OPENING_MAX_SEL; s++) {
        if (s >= p.sel_count) {
            break;
        }
        fr_store(out, (s << p.sigma) + gid, acc[s]);
    }
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
