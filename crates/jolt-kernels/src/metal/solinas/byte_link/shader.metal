// Byte-link prover kernels. Layouts shared with byte_link/*.rs:
// - Q holds 32 signed-byte columns, column c of row t at byte c * n + t (64-bit offsets: 31 n passes 2^32);
//   pack p reads the columns columns.c[p][0..3], and pack LINK_RAM is the RAM pack (two address bytes, activity);
// - tuple keys are canonical MSB-first: b0 << 16 | b1 << 8 | b2, RAM b0 << 9 | b1 << 1 | a; W of pack p
//   starts at cell p << 24;
// - tables[pack][slot][code]: slot 0 holds beta - gamma_0 sigma(code), slots 1 and 2 gamma_i sigma(code), so a
//   leaf denominator is t0[b0] - t1[b1] - t2[b2] for trace bytes and table codes alike;
// - tree levels live in one node buffer, level l of pack p at offset[l] + p * stride[l];
// - sumcheck rounds bind the lowest record-index bit first: pair p is (record 2p, record 2p + 1), weighted by
//   w_lo[p & mask] * w_hi[p >> lo_bits] with the high factor shared by a threadgroup.
#define LINK_THREADS 256u
#define LINK_SIMDS (LINK_THREADS / 32u)
#define LINK_SUMS 2u
#define LINK_TABLE 256u
#define LINK_SUM_THREADS 1024u
#define LINK_PACKS 7u
#define LINK_RAM 6u

struct LinkNode {
    SolinasFp128 p;
    SolinasFp128 q;
};

// Children 2x and 2x + 1 of parent x: one record of the parent layer's sumcheck.
struct LinkRecord {
    SolinasFp128 pl;
    SolinasFp128 ql;
    SolinasFp128 pr;
    SolinasFp128 qr;
};

struct LinkColumns {
    uint c[LINK_PACKS][4];
};

struct LinkLayout {
    uint offset[32];
    uint stride[32];
};

inline SolinasFp128 link_from_u64(ulong value) {
    SolinasFp128 result;
    result.limb = uint4((uint)value, (uint)(value >> 32), 0u, 0u);
    return result;
}

// low + overflow * 2^128 with overflow < 2^32, reduced to canonical form.
inline SolinasFp128 link_lazy_reduce(SolinasLazySum sum) {
    SolinasCorrection canonical = solinas_add_offset(sum.low);
    SolinasFp128 low = solinas_select(canonical.carry != 0u, canonical.value, sum.low);
    return solinas_add(low, link_from_u64((ulong)sum.overflow * (ulong)SOLINAS_OFFSET));
}

inline LinkNode link_gate(LinkNode l, LinkNode r) {
    LinkNode node;
    node.p = solinas_add(solinas_mul_wide(l.p, r.q), solinas_mul_wide(r.p, l.q));
    node.q = solinas_mul_wide(l.q, r.q);
    return node;
}

inline LinkNode link_record_gate(LinkRecord r) {
    LinkNode l = {r.pl, r.ql};
    LinkNode h = {r.pr, r.qr};
    return link_gate(l, h);
}

inline SolinasFp128 link_bind1(SolinasFp128 a, SolinasFp128 b, SolinasFp128 r) {
    return solinas_add(a, solinas_mul_wide(r, solinas_sub(b, a)));
}

inline LinkRecord link_bind(LinkRecord a, LinkRecord b, SolinasFp128 r) {
    LinkRecord out;
    out.pl = link_bind1(a.pl, b.pl, r);
    out.ql = link_bind1(a.ql, b.ql, r);
    out.pr = link_bind1(a.pr, b.pr, r);
    out.qr = link_bind1(a.qr, b.qr, r);
    return out;
}

inline SolinasFp128 link_split(device const SolinasFp128* lo, device const SolinasFp128* hi, uint lo_bits, uint x) {
    return solinas_mul_wide(hi[x >> lo_bits], lo[x & ((1u << lo_bits) - 1u)]);
}

inline SolinasFp128 link_denominator(device const SolinasFp128* tables, uint pack, uint b0, uint b1, uint b2) {
    device const SolinasFp128* t = tables + pack * 3u * LINK_TABLE;
    return solinas_sub(solinas_sub(t[b0], t[LINK_TABLE + b1]), t[2u * LINK_TABLE + b2]);
}

inline uint link_key(uint pack, uint b0, uint b1, uint b2) {
    return pack == LINK_RAM ? (b0 << 9) | (b1 << 1) | (b2 & 1u) : (b0 << 16) | (b1 << 8) | b2;
}

inline uint3 link_codes(uint pack, uint key) {
    return pack == LINK_RAM ? uint3(key >> 9, (key >> 1) & 255u, key & 1u)
                            : uint3(key >> 16, (key >> 8) & 255u, key & 255u);
}

inline device const uchar* link_column(device const uchar* q, constant LinkColumns& columns, uint n, uint pack, uint slot) {
    return q + (ulong)columns.c[pack][slot] * n;
}

// The trace leaves of one pack: (eq(r, t), beta - sum_i gamma_i D_i(t)).
struct LinkLeaves {
    device const uchar* c0;
    device const uchar* c1;
    device const uchar* c2;
    device const SolinasFp128* t;
    device const SolinasFp128* eq_lo;
    device const SolinasFp128* eq_hi;
    uint lo_bits;
};

inline LinkLeaves link_leaves(
    device const uchar* q,
    constant LinkColumns& columns,
    uint n,
    device const SolinasFp128* tables,
    device const SolinasFp128* eq_lo,
    device const SolinasFp128* eq_hi,
    uint lo_bits,
    uint pack)
{
    LinkLeaves leaves;
    leaves.c0 = link_column(q, columns, n, pack, 0u);
    leaves.c1 = link_column(q, columns, n, pack, 1u);
    leaves.c2 = link_column(q, columns, n, pack, 2u);
    leaves.t = tables + pack * 3u * LINK_TABLE;
    leaves.eq_lo = eq_lo;
    leaves.eq_hi = eq_hi;
    leaves.lo_bits = lo_bits;
    return leaves;
}

inline SolinasFp128 link_leaf_q(thread const LinkLeaves& s, uint t) {
    return solinas_sub(solinas_sub(s.t[s.c0[t]], s.t[LINK_TABLE + s.c1[t]]), s.t[2u * LINK_TABLE + s.c2[t]]);
}

inline LinkNode link_leaf(thread const LinkLeaves& s, uint t) {
    LinkNode leaf;
    leaf.p = link_split(s.eq_lo, s.eq_hi, s.lo_bits, t);
    leaf.q = link_leaf_q(s, t);
    return leaf;
}

inline LinkNode link_subtree1(thread const LinkLeaves& s, uint t) {
    return link_gate(link_leaf(s, t), link_leaf(s, t + 1u));
}

inline LinkNode link_subtree2(thread const LinkLeaves& s, uint t) {
    return link_gate(link_subtree1(s, t), link_subtree1(s, t + 2u));
}

inline LinkNode link_subtree3(thread const LinkLeaves& s, uint t) {
    return link_gate(link_subtree2(s, t), link_subtree2(s, t + 4u));
}

inline LinkNode link_subtree(thread const LinkLeaves& s, uint k, uint height) {
    uint t = k << height;
    return height == 1u ? link_subtree1(s, t) : height == 2u ? link_subtree2(s, t) : link_subtree3(s, t);
}

struct LinkTreeParams {
    uint table;
    uint count;
    uint first;
    uint levels;
    uint n;
    uint lo_bits;
    uint pack_base;
    uint reserved;
    LinkLayout layout;
};

inline void link_store(device LinkNode* levels, constant LinkTreeParams& params, uint level, uint pack, uint index, LinkNode node) {
    levels[params.layout.offset[level] + pack * params.layout.stride[level] + index] = node;
}

inline void link_levels(
    LinkNode node,
    device LinkNode* levels,
    constant LinkTreeParams& params,
    uint pack,
    uint local,
    uint tid,
    uint width,
    threadgroup LinkNode* staged)
{
    uint level = params.first;
    link_store(levels, params, level, pack, local * width + tid, node);
    for (uint written = 1u; written < params.levels; written++) {
        staged[tid] = node;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        width /= 2u;
        level += 1u;
        if (tid < width) {
            node = link_gate(staged[2u * tid], staged[2u * tid + 1u]);
            link_store(levels, params, level, pack, local * width + tid, node);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Trace trees rebuild node j of level params.first from its 2^first leaves; table trees (first = 1) also store
// both leaves at level 0, which their bottom layer reads.
kernel void byte_link_tree_leaves(
    device const uchar* q [[buffer(0)]],
    device const SolinasFp128* w [[buffer(1)]],
    device const SolinasFp128* tables [[buffer(2)]],
    device const SolinasFp128* eq_lo [[buffer(3)]],
    device const SolinasFp128* eq_hi [[buffer(4)]],
    device LinkNode* levels [[buffer(5)]],
    constant LinkTreeParams& params [[buffer(6)]],
    constant LinkColumns& columns [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup LinkNode staged[LINK_THREADS];
    uint groups = params.count / width;
    uint pack = group / groups;
    uint local = group % groups;
    uint j = local * width + tid;
    LinkNode node;
    if (params.table == 0u) {
        LinkLeaves leaves = link_leaves(q, columns, params.n, tables, eq_lo, eq_hi, params.lo_bits, pack);
        node = link_subtree(leaves, j, params.first);
    } else {
        uint table_pack = params.pack_base + pack;
        LinkNode leaf[2];
        for (uint b = 0u; b < 2u; b++) {
            uint key = 2u * j + b;
            uint3 code = link_codes(table_pack, key);
            leaf[b].p = w[(table_pack << 24) + key];
            leaf[b].q = link_denominator(tables, table_pack, code.x, code.y, code.z);
            link_store(levels, params, 0u, pack, key, leaf[b]);
        }
        node = link_gate(leaf[0], leaf[1]);
    }
    link_levels(node, levels, params, pack, local, tid, width, staged);
}

kernel void byte_link_tree_upper(
    device LinkNode* levels [[buffer(0)]],
    constant LinkTreeParams& params [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup LinkNode staged[LINK_THREADS];
    uint groups = params.count / width;
    uint pack = group / groups;
    uint local = group % groups;
    uint j = local * width + tid;
    uint below = params.first - 1u;
    device const LinkNode* c = levels + params.layout.offset[below] + pack * params.layout.stride[below] + 2u * j;
    link_levels(link_gate(c[0], c[1]), levels, params, pack, local, tid, width, staged);
}

// Every pack's P and Q sums are batched in-kernel with its weights coeffs[2 pack] and coeffs[2 pack + 1], so a
// point costs one accumulator. A round sums the gate at X = 0 and its X^2 coefficient (the gate of the slopes).

struct LinkRoundParams {
    uint pairs;
    uint groups;
    uint pairs_per_thread;
    uint lo_bits;
    uint in_stride;
    uint out_stride;
    uint e_lo_bits;
    uint stream_log;
    uint n;
    uint materialize;
    uint bound;
    uint shift;
    uint height;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    SolinasFp128 challenge;
};

// Reduces `count` lazy sums of the threadgroup, scales them by w_hi and writes them to
// partials[group * LINK_SUMS + k].
inline void link_round_finish(
    thread SolinasLazySum* acc,
    uint count,
    threadgroup SolinasFp128* simd_sums,
    SolinasFp128 w_hi,
    device SolinasFp128* partials,
    uint group,
    uint tid,
    uint lane,
    uint simd)
{
    for (uint k = 0u; k < count; k++) {
        SolinasFp128 total = solinas_simd_sum_32(link_lazy_reduce(acc[k]));
        if (lane == 0u) {
            simd_sums[k * LINK_SIMDS + simd] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < count) {
        SolinasFp128 total = simd_sums[tid * LINK_SIMDS];
        for (uint s = 1u; s < LINK_SIMDS; s++) {
            total = solinas_add(total, simd_sums[tid * LINK_SIMDS + s]);
        }
        partials[group * LINK_SUMS + tid] = solinas_mul_wide(w_hi, total);
    }
}

// sums[pack][k] = sum of the partials of the pack's `groups` threadgroups; one threadgroup per pack.
kernel void byte_link_sum_partials(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* sums [[buffer(1)]],
    constant uint& groups [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint pack [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 simd_sums[LINK_SUMS * (LINK_SUM_THREADS / 32u)];
    for (uint k = 0u; k < LINK_SUMS; k++) {
        SolinasLazySum acc = solinas_lazy_zero();
        for (uint g = tid; g < groups; g += LINK_SUM_THREADS) {
            solinas_lazy_add(acc, partials[(pack * groups + g) * LINK_SUMS + k]);
        }
        SolinasFp128 total = solinas_simd_sum_32(link_lazy_reduce(acc));
        if (lane == 0u) {
            simd_sums[k * (LINK_SUM_THREADS / 32u) + simd] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < LINK_SUMS) {
        SolinasFp128 total = simd_sums[tid * (LINK_SUM_THREADS / 32u)];
        for (uint s = 1u; s < LINK_SUM_THREADS / 32u; s++) {
            total = solinas_add(total, simd_sums[tid * (LINK_SUM_THREADS / 32u) + s]);
        }
        sums[pack * LINK_SUMS + tid] = total;
    }
}

inline void link_point(thread SolinasLazySum& acc, LinkNode g, SolinasFp128 wp, SolinasFp128 wq) {
    solinas_lazy_add(acc, solinas_mul_wide(wp, g.p));
    solinas_lazy_add(acc, solinas_mul_wide(wq, g.q));
}

inline void link_fold(thread SolinasLazySum* acc, LinkRecord lo, LinkRecord hi, SolinasFp128 wp, SolinasFp128 wq) {
    link_point(acc[0], link_record_gate(lo), wp, wq);
    LinkRecord d;
    d.pl = solinas_sub(hi.pl, lo.pl);
    d.ql = solinas_sub(hi.ql, lo.ql);
    d.pr = solinas_sub(hi.pr, lo.pr);
    d.qr = solinas_sub(hi.qr, lo.qr);
    link_point(acc[1], link_record_gate(d), wp, wq);
}

#define LINK_ROUND_ARGS                                              \
    device const SolinasFp128* w_lo [[buffer(1)]],                   \
    device const SolinasFp128* w_hi [[buffer(2)]],                   \
    device SolinasFp128* partials [[buffer(3)]],                     \
    constant LinkRoundParams& params [[buffer(5)]],                  \
    device const SolinasFp128* coeffs [[buffer(6)]],                 \
    uint tid [[thread_index_in_threadgroup]],                        \
    uint group [[threadgroup_position_in_grid]],                     \
    uint lane [[thread_index_in_simdgroup]],                         \
    uint simd [[simdgroup_index_in_threadgroup]]

#define LINK_ROUND_PROLOGUE                                          \
    threadgroup SolinasFp128 simd_sums[LINK_SUMS * LINK_SIMDS];      \
    uint pack = group / params.groups;                               \
    uint base = (group % params.groups) * LINK_THREADS * params.pairs_per_thread; \
    SolinasFp128 c_p = coeffs[2u * pack];                            \
    SolinasFp128 c_q = coeffs[2u * pack + 1u];                       \
    SolinasLazySum acc[2] = {solinas_lazy_zero(), solinas_lazy_zero()};

#define LINK_WEIGHTS(P)                                              \
    SolinasFp128 w = w_lo[(P) & ((1u << params.lo_bits) - 1u)];      \
    SolinasFp128 wp = solinas_mul_wide(w, c_p);                      \
    SolinasFp128 wq = solinas_mul_wide(w, c_q);

#define LINK_ROUND_EPILOGUE                                          \
    link_round_finish(acc, 2u, simd_sums, w_hi[base >> params.lo_bits], partials, group, tid, lane, simd);

kernel void byte_link_round_eval(device const LinkRecord* records [[buffer(0)]], LINK_ROUND_ARGS) {
    LINK_ROUND_PROLOGUE
    device const LinkRecord* in = records + pack * params.in_stride;
    for (uint i = 0u; i < params.pairs_per_thread; i++) {
        uint p = base + i * LINK_THREADS + tid;
        if (p < params.pairs) {
            LINK_WEIGHTS(p)
            link_fold(acc, in[2u * p], in[2u * p + 1u], wp, wq);
        }
    }
    LINK_ROUND_EPILOGUE
}

// Binds the previous round at params.challenge in place and evaluates this round's `pairs` pairs; the last pair
// of an odd-length bound array only binds. Bound records 2p and 2p + 1 overwrite input records 4p and 4p + 1,
// which only this thread reads, so i rounds after the records were stored, record j sits at slot
// ((j & ~1) << i) | (j & 1); params.shift is i - 1.
kernel void byte_link_round_bind_eval(device LinkRecord* records [[buffer(0)]], LINK_ROUND_ARGS) {
    LINK_ROUND_PROLOGUE
    device LinkRecord* in = records + pack * params.in_stride;
    for (uint i = 0u; i < params.pairs_per_thread; i++) {
        uint p = base + i * LINK_THREADS + tid;
        if (p < params.pairs) {
            uint at = (4u * p) << params.shift;
            LinkRecord lo = link_bind(in[at], in[at + 1u], params.challenge);
            in[at] = lo;
            if (2u * p + 1u < params.bound) {
                uint hi_at = at + (2u << params.shift);
                LinkRecord hi = link_bind(in[hi_at], in[hi_at + 1u], params.challenge);
                in[at + 1u] = hi;
                LINK_WEIGHTS(p)
                link_fold(acc, lo, hi, wp, wq);
            }
        }
    }
    LINK_ROUND_EPILOGUE
}

// Round stream_log (0 or 1) of the layer whose children are level params.height (1 or 2) of the trace trees,
// whose nodes are rebuilt from Q instead of stored: record m is (node 2m, node 2m + 1), and in round 1 bound
// record j binds records 2j and 2j + 1 at params.challenge. `materialize` stores this round's bound records,
// which later rounds bind in place.
kernel void byte_link_round_stream(
    device LinkRecord* out [[buffer(0)]],
    LINK_ROUND_ARGS,
    device const uchar* q [[buffer(7)]],
    device const SolinasFp128* tables [[buffer(8)]],
    device const SolinasFp128* eq_lo [[buffer(9)]],
    device const SolinasFp128* eq_hi [[buffer(10)]],
    constant LinkColumns& columns [[buffer(11)]])
{
    LINK_ROUND_PROLOGUE
    LinkLeaves leaves = link_leaves(q, columns, params.n, tables, eq_lo, eq_hi, params.e_lo_bits, pack);
    device LinkRecord* o = out + pack * params.out_stride;
    for (uint i = 0u; i < params.pairs_per_thread; i++) {
        uint p = base + i * LINK_THREADS + tid;
        if (p < params.pairs) {
            LinkRecord record[2];
            for (uint c = 0u; c < 2u; c++) {
                uint m = (2u * p + c) << params.stream_log;
                LinkNode l = link_subtree(leaves, 2u * m, params.height);
                LinkNode r = link_subtree(leaves, 2u * m + 1u, params.height);
                record[c] = LinkRecord{l.p, l.q, r.p, r.q};
                if (params.stream_log != 0u) {
                    LinkNode l1 = link_subtree(leaves, 2u * m + 2u, params.height);
                    LinkNode r1 = link_subtree(leaves, 2u * m + 3u, params.height);
                    LinkRecord next = {l1.p, l1.q, r1.p, r1.q};
                    record[c] = link_bind(record[c], next, params.challenge);
                }
                if (params.materialize != 0u) {
                    o[2u * p + c] = record[c];
                }
            }
            LINK_WEIGHTS(p)
            link_fold(acc, record[0], record[1], wp, wq);
        }
    }
    LINK_ROUND_EPILOGUE
}

// Bottom layer (children are leaves): leaf numerators are eq(r, t), so child b's numerator at X = 0 and at the
// X slope of pair p is k[b] E(p) and k[2 + b] E(p), E from the split tables e_lo/e_hi; gkr.rs derives k per
// round. Only the denominators are data.

struct LinkQLine {
    SolinasFp128 lo;
    SolinasFp128 d;
};

struct LinkQ2 {
    SolinasFp128 q[2];
};

inline LinkQLine link_qline(SolinasFp128 lo, SolinasFp128 hi) {
    LinkQLine c;
    c.lo = lo;
    c.d = solinas_sub(hi, lo);
    return c;
}

inline void link_bottom_fold(thread SolinasLazySum* acc, device const SolinasFp128* k, LinkQLine c0, LinkQLine c1, SolinasFp128 e, SolinasFp128 wp, SolinasFp128 wq) {
    LinkNode x0 = {solinas_mul_wide(k[0], e), c0.lo};
    LinkNode x1 = {solinas_mul_wide(k[1], e), c1.lo};
    link_point(acc[0], link_gate(x0, x1), wp, wq);
    LinkNode d0 = {solinas_mul_wide(k[2], e), c0.d};
    LinkNode d1 = {solinas_mul_wide(k[3], e), c1.d};
    link_point(acc[1], link_gate(d0, d1), wp, wq);
}

#define LINK_BOTTOM_ARGS                                             \
    LINK_ROUND_ARGS,                                                 \
    device const SolinasFp128* e_lo [[buffer(7)]],                   \
    device const SolinasFp128* e_hi [[buffer(8)]],                   \
    device const SolinasFp128* k [[buffer(9)]]

// Denominator of child b of the bound record j in round stream_log (0 or 1), from Q: in round 1 the record
// binds records 2j and 2j + 1 at params.challenge.
inline SolinasFp128 link_stream_q(thread const LinkLeaves& leaves, constant LinkRoundParams& params, uint j, uint b) {
    uint m = j << params.stream_log;
    SolinasFp128 q = link_leaf_q(leaves, 2u * m + b);
    if (params.stream_log != 0u) {
        q = link_bind1(q, link_leaf_q(leaves, 2u * m + 2u + b), params.challenge);
    }
    return q;
}

// Rounds 0 and 1 of the bottom layer from Q; `materialize` stores the bound denominators of this round.
kernel void byte_link_bottom_stream(
    device LinkQ2* bound [[buffer(0)]],
    LINK_BOTTOM_ARGS,
    device const uchar* q [[buffer(10)]],
    device const SolinasFp128* tables [[buffer(11)]],
    constant LinkColumns& columns [[buffer(12)]])
{
    LINK_ROUND_PROLOGUE
    LinkLeaves leaves = link_leaves(q, columns, params.n, tables, e_lo, e_hi, 0u, pack);
    device LinkQ2* out = bound + pack * params.out_stride;
    for (uint i = 0u; i < params.pairs_per_thread; i++) {
        uint p = base + i * LINK_THREADS + tid;
        if (p < params.pairs) {
            LinkQLine c[2];
            for (uint b = 0u; b < 2u; b++) {
                SolinasFp128 q_lo = link_stream_q(leaves, params, 2u * p, b);
                SolinasFp128 q_hi = link_stream_q(leaves, params, 2u * p + 1u, b);
                if (params.materialize != 0u) {
                    out[2u * p].q[b] = q_lo;
                    out[2u * p + 1u].q[b] = q_hi;
                }
                c[b] = link_qline(q_lo, q_hi);
            }
            LINK_WEIGHTS(p)
            link_bottom_fold(acc, k, c[0], c[1], link_split(e_lo, e_hi, params.e_lo_bits, p), wp, wq);
        }
    }
    LINK_ROUND_EPILOGUE
}

// Later bottom rounds over the stored denominators, in place with the slot map of byte_link_round_bind_eval.
kernel void byte_link_bottom_bind_eval(device LinkQ2* records [[buffer(0)]], LINK_BOTTOM_ARGS) {
    LINK_ROUND_PROLOGUE
    device LinkQ2* in = records + pack * params.in_stride;
    for (uint i = 0u; i < params.pairs_per_thread; i++) {
        uint p = base + i * LINK_THREADS + tid;
        if (p >= params.pairs) {
            continue;
        }
        bool has_hi = 2u * p + 1u < params.bound;
        device LinkQ2* a = in + ((4u * p) << params.shift);
        device LinkQ2* h = a + (2u << params.shift);
        LinkQLine c[2];
        for (uint b = 0u; b < 2u; b++) {
            SolinasFp128 q_lo = link_bind1(a[0].q[b], a[1].q[b], params.challenge);
            SolinasFp128 q_hi = q_lo;
            if (has_hi) {
                q_hi = link_bind1(h[0].q[b], h[1].q[b], params.challenge);
            }
            a[0].q[b] = q_lo;
            if (has_hi) {
                a[1].q[b] = q_hi;
            }
            c[b] = link_qline(q_lo, q_hi);
        }
        if (has_hi) {
            LINK_WEIGHTS(p)
            link_bottom_fold(acc, k, c[0], c[1], link_split(e_lo, e_hi, params.e_lo_bits, p), wp, wq);
        }
    }
    LINK_ROUND_EPILOGUE
}

// Entries are N/2 factor pairs (f, g); a round sums f g at X = 0 and its leading coefficient over all pairs.

template <uint N>
struct LinkProd {
    SolinasFp128 v[N];
};

template <uint N>
inline void link_product_accumulate(thread SolinasLazySum* acc, LinkProd<N> a, LinkProd<N> b) {
    for (uint k = 0u; k < N; k += 2u) {
        solinas_lazy_add(acc[0], solinas_mul_wide(a.v[k], a.v[k + 1u]));
        solinas_lazy_add(acc[1], solinas_mul_wide(solinas_sub(b.v[k], a.v[k]), solinas_sub(b.v[k + 1u], a.v[k + 1u])));
    }
}

template <uint N>
inline LinkProd<N> link_product_bind(LinkProd<N> a, LinkProd<N> b, SolinasFp128 r) {
    LinkProd<N> out;
    for (uint k = 0u; k < N; k++) {
        out.v[k] = link_bind1(a.v[k], b.v[k], r);
    }
    return out;
}

#define LINK_PRODUCT_ARGS                                            \
    device SolinasFp128* partials [[buffer(3)]],                     \
    constant LinkRoundParams& params [[buffer(5)]],                  \
    uint tid [[thread_index_in_threadgroup]],                        \
    uint group [[threadgroup_position_in_grid]],                     \
    uint lane [[thread_index_in_simdgroup]],                         \
    uint simd [[simdgroup_index_in_threadgroup]]

#define LINK_PRODUCT_PROLOGUE                                        \
    threadgroup SolinasFp128 simd_sums[LINK_SUMS * LINK_SIMDS];      \
    uint pack = group / params.groups;                               \
    uint base = (group % params.groups) * LINK_THREADS * params.pairs_per_thread; \
    SolinasLazySum acc[2] = {solinas_lazy_zero(), solinas_lazy_zero()};

#define LINK_PRODUCT_EPILOGUE                                        \
    link_round_finish(acc, 2u, simd_sums, link_from_u64(1u), partials, group, tid, lane, simd);

#define LINK_PRODUCT_KERNELS(N)                                      \
kernel void byte_link_product##N##_eval(device const LinkProd<N>* records [[buffer(0)]], LINK_PRODUCT_ARGS) { \
    LINK_PRODUCT_PROLOGUE                                            \
    device const LinkProd<N>* in = records + pack * params.in_stride; \
    for (uint i = 0u; i < params.pairs_per_thread; i++) {            \
        uint p = base + i * LINK_THREADS + tid;                      \
        if (p < params.pairs) {                                      \
            link_product_accumulate<N>(acc, in[2u * p], in[2u * p + 1u]); \
        }                                                            \
    }                                                                \
    LINK_PRODUCT_EPILOGUE                                            \
}                                                                    \
kernel void byte_link_product##N##_bind_eval(device const LinkProd<N>* records [[buffer(0)]], device LinkProd<N>* bound [[buffer(4)]], LINK_PRODUCT_ARGS) { \
    LINK_PRODUCT_PROLOGUE                                            \
    device const LinkProd<N>* in = records + pack * params.in_stride; \
    device LinkProd<N>* out = bound + pack * params.out_stride;      \
    for (uint i = 0u; i < params.pairs_per_thread; i++) {            \
        uint p = base + i * LINK_THREADS + tid;                      \
        if (p < params.pairs) {                                      \
            LinkProd<N> lo = link_product_bind<N>(in[4u * p], in[4u * p + 1u], params.challenge); \
            out[2u * p] = lo;                                        \
            if (2u * p + 1u < params.bound) {                        \
                LinkProd<N> hi = link_product_bind<N>(in[4u * p + 2u], in[4u * p + 3u], params.challenge); \
                out[2u * p + 1u] = hi;                               \
                link_product_accumulate<N>(acc, lo, hi);             \
            }                                                        \
        }                                                            \
    }                                                                \
    LINK_PRODUCT_EPILOGUE                                            \
}

LINK_PRODUCT_KERNELS(2)
LINK_PRODUCT_KERNELS(4)

// Raw entries of round 0 and of the round-1 bind. Query (W of triple pack `pack` against its weights):
// (W(h), U0[h0] + U1[h1] + U2[h2] + Z0[h0] Z1[h1] Z2[h2]), tables holding U then Z per pack. Source (the Q
// point reduction): (eq(z, t), G(t), eq(r, t), F(t)), G and F sums of per-column tables over Q.
struct LinkRawParams {
    uint pairs;
    uint groups;
    uint pairs_per_thread;
    uint bound;
    uint out_stride;
    uint n;
    uint z_lo_bits;
    uint r_lo_bits;
    SolinasFp128 challenge;
};

struct LinkSourceColumns {
    uint g[24];
    uint f[12];
};

inline LinkProd<2> link_query_entry(device const SolinasFp128* w, device const SolinasFp128* tables, uint pack, uint key) {
    device const SolinasFp128* u = tables + pack * 6u * LINK_TABLE;
    device const SolinasFp128* z = u + 3u * LINK_TABLE;
    uint3 code = link_codes(pack, key);
    SolinasFp128 zp = solinas_mul_wide(solinas_mul_wide(z[code.x], z[LINK_TABLE + code.y]), z[2u * LINK_TABLE + code.z]);
    LinkProd<2> entry;
    entry.v[0] = w[(pack << 24) + key];
    entry.v[1] = solinas_add(solinas_add(solinas_add(u[code.x], u[LINK_TABLE + code.y]), u[2u * LINK_TABLE + code.z]), zp);
    return entry;
}

inline LinkProd<4> link_source_entry(
    constant LinkRawParams& params,
    device const uchar* q,
    device const SolinasFp128* g_tables,
    device const SolinasFp128* f_tables,
    device const SolinasFp128* eq,
    constant LinkSourceColumns& columns,
    uint t)
{
    device const SolinasFp128* z_lo = eq;
    device const SolinasFp128* z_hi = z_lo + (1u << params.z_lo_bits);
    device const SolinasFp128* r_lo = z_hi + (params.n >> params.z_lo_bits);
    device const SolinasFp128* r_hi = r_lo + (1u << params.r_lo_bits);
    SolinasLazySum g = solinas_lazy_zero();
    for (uint c = 0u; c < 3u * LINK_PACKS; c++) {
        solinas_lazy_add(g, g_tables[c * LINK_TABLE + q[(ulong)columns.g[c] * params.n + t]]);
    }
    SolinasLazySum f = solinas_lazy_zero();
    for (uint c = 0u; c < 9u; c++) {
        solinas_lazy_add(f, f_tables[c * LINK_TABLE + q[(ulong)columns.f[c] * params.n + t]]);
    }
    LinkProd<4> entry;
    entry.v[0] = link_split(z_lo, z_hi, params.z_lo_bits, t);
    entry.v[1] = link_lazy_reduce(g);
    entry.v[2] = link_split(r_lo, r_hi, params.r_lo_bits, t);
    entry.v[3] = link_lazy_reduce(f);
    return entry;
}

#define LINK_RAW_TAIL(N)                                             \
    device SolinasFp128* partials [[buffer(7)]],                     \
    device LinkProd<N>* bound [[buffer(8)]],                         \
    constant LinkRawParams& params [[buffer(9)]],                    \
    uint tid [[thread_index_in_threadgroup]],                        \
    uint group [[threadgroup_position_in_grid]],                     \
    uint lane [[thread_index_in_simdgroup]],                         \
    uint simd [[simdgroup_index_in_threadgroup]]

#define LINK_QUERY_ARGS                                              \
    device const SolinasFp128* w [[buffer(0)]],                      \
    device const SolinasFp128* tables [[buffer(1)]],                 \
    LINK_RAW_TAIL(2)

#define LINK_SOURCE_ARGS                                             \
    device const uchar* q [[buffer(0)]],                             \
    device const SolinasFp128* g_tables [[buffer(1)]],               \
    device const SolinasFp128* f_tables [[buffer(2)]],               \
    device const SolinasFp128* eq [[buffer(4)]],                     \
    constant LinkSourceColumns& columns [[buffer(5)]],               \
    LINK_RAW_TAIL(4)

// Round 0 sums the raw entries; `bind` stores the round-1 bound entries, which byte_link_product<N>_eval then
// evaluates in a separate pass (see reduce.rs).
#define LINK_RAW_KERNELS(NAME, ARGS, ENTRY, N)                       \
kernel void byte_link_##NAME##_eval(ARGS) {                          \
    threadgroup SolinasFp128 simd_sums[LINK_SUMS * LINK_SIMDS];      \
    uint pack = group / params.groups;                               \
    uint base = (group % params.groups) * LINK_THREADS * params.pairs_per_thread; \
    SolinasLazySum acc[2] = {solinas_lazy_zero(), solinas_lazy_zero()}; \
    for (uint i = 0u; i < params.pairs_per_thread; i++) {            \
        uint p = base + i * LINK_THREADS + tid;                      \
        if (p < params.pairs) {                                      \
            link_product_accumulate<N>(acc, ENTRY(2u * p), ENTRY(2u * p + 1u)); \
        }                                                            \
    }                                                                \
    LINK_PRODUCT_EPILOGUE                                            \
}                                                                    \
kernel void byte_link_##NAME##_bind(ARGS) {                          \
    uint pack = group / params.groups;                               \
    uint base = (group % params.groups) * LINK_THREADS * params.pairs_per_thread; \
    device LinkProd<N>* out = bound + pack * params.out_stride;      \
    for (uint i = 0u; i < params.pairs_per_thread; i++) {            \
        uint j = base + i * LINK_THREADS + tid;                      \
        if (j < params.bound) {                                      \
            out[j] = link_product_bind<N>(ENTRY(2u * j), ENTRY(2u * j + 1u), params.challenge); \
        }                                                            \
    }                                                                \
}

#define LINK_QUERY_ENTRY(X) link_query_entry(w, tables, pack, X)
#define LINK_SOURCE_ENTRY(X) link_source_entry(params, q, g_tables, f_tables, eq, columns, X)
LINK_RAW_KERNELS(query, LINK_QUERY_ARGS, LINK_QUERY_ENTRY, 2)
LINK_RAW_KERNELS(source, LINK_SOURCE_ARGS, LINK_SOURCE_ENTRY, 4)

// Key sort of one pack: counting partition of the rows by the high 12 key bits (count, scan, scatter), then a
// per-bucket counting sort by the low bits that also writes the CSR offsets. Not stable: a key's cycles come out
// in arbitrary order, which the exact run sums below do not see.
#define LINK_SORT_THREADS 1024u
#define LINK_SORT_PER_THREAD 16u
#define LINK_SORT_TILE (LINK_SORT_THREADS * LINK_SORT_PER_THREAD)
#define LINK_SORT_BINS 4096u

struct LinkSortParams {
    uint entries;
    uint n;
    uint pack;
    uint low_bits;
};

inline void link_zero_bins(threadgroup atomic_uint* bins, uint tid) {
    for (uint b = tid; b < LINK_SORT_BINS; b += LINK_SORT_THREADS) {
        atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Exclusive prefix sum of the 4096 bins in place, four consecutive bins per thread.
inline void link_scan_bins(threadgroup atomic_uint* bins, threadgroup uint* simd_totals, uint tid, uint lane, uint simd) {
    uint v[4];
    uint local = 0u;
    for (uint k = 0u; k < 4u; k++) {
        v[k] = atomic_load_explicit(&bins[4u * tid + k], memory_order_relaxed);
        local += v[k];
    }
    uint prefix = simd_prefix_exclusive_sum(local);
    if (lane == 31u) {
        simd_totals[simd] = prefix + local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        simd_totals[lane] = simd_prefix_exclusive_sum(simd_totals[lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = simd_totals[simd] + prefix;
    for (uint k = 0u; k < 4u; k++) {
        atomic_store_explicit(&bins[4u * tid + k], offset, memory_order_relaxed);
        offset += v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline uint link_row_key(device const uchar* q, constant LinkColumns& columns, constant LinkSortParams& params, uint t) {
    uint pack = params.pack;
    return link_key(
        pack,
        link_column(q, columns, params.n, pack, 0u)[t],
        link_column(q, columns, params.n, pack, 1u)[t],
        link_column(q, columns, params.n, pack, 2u)[t]);
}

// Key 0 (every byte zero: every inactive RAM row) is counted apart into zeros[0] and scattered straight to the
// front of the sorted pairs: as one bucket it would leave a single threadgroup of byte_link_sort_buckets with
// most of the rows.
kernel void byte_link_sort_count(
    device const uchar* q [[buffer(0)]],
    device atomic_uint* bucket_counts [[buffer(1)]],
    constant LinkSortParams& params [[buffer(2)]],
    device atomic_uint* zeros [[buffer(3)]],
    constant LinkColumns& columns [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint bins[LINK_SORT_BINS];
    link_zero_bins(bins, tid);
    uint zero = 0u;
    for (uint i = 0u; i < LINK_SORT_PER_THREAD; i++) {
        uint t = group * LINK_SORT_TILE + i * LINK_SORT_THREADS + tid;
        if (t < params.entries) {
            uint key = link_row_key(q, columns, params, t);
            if (key == 0u) {
                zero++;
            } else {
                atomic_fetch_add_explicit(&bins[key >> params.low_bits], 1u, memory_order_relaxed);
            }
        }
    }
    zero = simd_sum(zero);
    if (lane == 0u && zero != 0u) {
        atomic_fetch_add_explicit(zeros, zero, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < LINK_SORT_BINS; b += LINK_SORT_THREADS) {
        uint count = atomic_load_explicit(&bins[b], memory_order_relaxed);
        if (count != 0u) {
            atomic_fetch_add_explicit(&bucket_counts[b], count, memory_order_relaxed);
        }
    }
}

// Bucket starts after the zeros[0] key-0 pairs; bucket_start[LINK_SORT_BINS] = entries.
kernel void byte_link_sort_scan(
    device const uint* bucket_counts [[buffer(0)]],
    device uint* bucket_start [[buffer(1)]],
    device uint* cursors [[buffer(2)]],
    constant uint& entries [[buffer(3)]],
    device const uint* zeros [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup atomic_uint bins[LINK_SORT_BINS];
    threadgroup uint simd_totals[32];
    for (uint b = tid; b < LINK_SORT_BINS; b += LINK_SORT_THREADS) {
        atomic_store_explicit(&bins[b], bucket_counts[b], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    link_scan_bins(bins, simd_totals, tid, lane, simd);
    for (uint b = tid; b < LINK_SORT_BINS; b += LINK_SORT_THREADS) {
        uint start = zeros[0] + atomic_load_explicit(&bins[b], memory_order_relaxed);
        bucket_start[b] = start;
        cursors[b] = start;
    }
    if (tid == 0u) {
        bucket_start[LINK_SORT_BINS] = entries;
    }
}

kernel void byte_link_sort_scatter(
    device const uchar* q [[buffer(0)]],
    device atomic_uint* cursors [[buffer(1)]],
    device uint2* staging [[buffer(2)]],
    constant LinkSortParams& params [[buffer(3)]],
    device atomic_uint* zero_cursor [[buffer(4)]],
    device uint2* sorted [[buffer(5)]],
    constant LinkColumns& columns [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint bins[LINK_SORT_BINS];
    threadgroup atomic_uint zero_rank;
    threadgroup uint zero_base;
    if (tid == 0u) {
        atomic_store_explicit(&zero_rank, 0u, memory_order_relaxed);
    }
    link_zero_bins(bins, tid);
    uint key[LINK_SORT_PER_THREAD];
    uint rank[LINK_SORT_PER_THREAD];
    for (uint i = 0u; i < LINK_SORT_PER_THREAD; i++) {
        uint t = group * LINK_SORT_TILE + i * LINK_SORT_THREADS + tid;
        key[i] = t < params.entries ? link_row_key(q, columns, params, t) : 1u;
        bool zero = t < params.entries && key[i] == 0u;
        uint before = simd_prefix_exclusive_sum(uint(zero));
        uint zeros = simd_sum(uint(zero));
        uint base = 0u;
        if (lane == 0u && zeros != 0u) {
            base = atomic_fetch_add_explicit(&zero_rank, zeros, memory_order_relaxed);
        }
        base = simd_broadcast(base, 0u);
        if (zero) {
            rank[i] = base + before;
        } else if (t < params.entries) {
            rank[i] = atomic_fetch_add_explicit(&bins[key[i] >> params.low_bits], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < LINK_SORT_BINS; b += LINK_SORT_THREADS) {
        uint count = atomic_load_explicit(&bins[b], memory_order_relaxed);
        if (count != 0u) {
            uint start = atomic_fetch_add_explicit(&cursors[b], count, memory_order_relaxed);
            atomic_store_explicit(&bins[b], start, memory_order_relaxed);
        }
    }
    if (tid == 0u) {
        uint count = atomic_load_explicit(&zero_rank, memory_order_relaxed);
        zero_base = count == 0u ? 0u : atomic_fetch_add_explicit(zero_cursor, count, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < LINK_SORT_PER_THREAD; i++) {
        uint t = group * LINK_SORT_TILE + i * LINK_SORT_THREADS + tid;
        if (t < params.entries) {
            if (key[i] == 0u) {
                sorted[zero_base + rank[i]] = uint2(0u, t);
            } else {
                uint start = atomic_load_explicit(&bins[key[i] >> params.low_bits], memory_order_relaxed);
                staging[start + rank[i]] = uint2(key[i], t);
            }
        }
    }
}

// One threadgroup per high bucket: counting sort by the low bits, CSR offsets and sorted (key, cycle). The
// entries first move to `scratch` grouped by the top half of the low bits, so the final scatter of the
// threadgroup lands within one of 64 groups (12 KB at 2^29) instead of anywhere in the bucket (787 KB): the
// whole-bucket scatter outgrew the GPU caches as buckets grew with the rows.
kernel void byte_link_sort_buckets(
    device const uint2* staging [[buffer(0)]],
    device const uint* bucket_start [[buffer(1)]],
    device uint* offsets [[buffer(2)]],
    device uint2* scratch [[buffer(3)]],
    constant LinkSortParams& params [[buffer(4)]],
    device uint2* sorted [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint bucket [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint bins[LINK_SORT_BINS];
    threadgroup atomic_uint groups[64];
    threadgroup uint simd_totals[32];
    uint low_mask = (1u << params.low_bits) - 1u;
    uint group_shift = params.low_bits / 2u;
    uint begin = bucket_start[bucket];
    uint end = bucket_start[bucket + 1u];
    link_zero_bins(bins, tid);
    for (uint i = begin + tid; i < end; i += LINK_SORT_THREADS) {
        atomic_fetch_add_explicit(&bins[staging[i].x & low_mask], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    link_scan_bins(bins, simd_totals, tid, lane, simd);
    for (uint b = tid; b <= low_mask; b += LINK_SORT_THREADS) {
        uint key = (bucket << params.low_bits) + b;
        offsets[key] = key == 0u ? 0u : begin + atomic_load_explicit(&bins[b], memory_order_relaxed);
    }
    if (bucket + 1u == LINK_SORT_BINS && tid == 0u) {
        offsets[LINK_SORT_BINS << params.low_bits] = params.entries;
    }
    if (tid <= (low_mask >> group_shift)) {
        atomic_store_explicit(&groups[tid], atomic_load_explicit(&bins[tid << group_shift], memory_order_relaxed), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = begin + tid; i < end; i += LINK_SORT_THREADS) {
        uint2 entry = staging[i];
        uint position = atomic_fetch_add_explicit(&groups[(entry.x & low_mask) >> group_shift], 1u, memory_order_relaxed);
        scratch[begin + position] = entry;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = begin + tid; i < end; i += LINK_SORT_THREADS) {
        uint2 entry = scratch[i];
        uint position = atomic_fetch_add_explicit(&bins[entry.x & low_mask], 1u, memory_order_relaxed);
        sorted[begin + position] = entry;
    }
}

// W from the key-sorted cycles in fixed chunks of sorted positions: a run whose key lies wholly inside the
// chunk is written directly; a run of a key crossing chunk boundaries is added exactly into that key's
// five-word integer accumulator, reduced by byte_link_weights_finish.
struct LinkWeightParams {
    uint entries;
    uint keys;
    uint chunk;
    uint lo_bits;
};

inline bool link_crossing(device const uint* offsets, uint chunk, uint key) {
    uint begin = offsets[key];
    uint end = offsets[key + 1u];
    return end > begin && begin / chunk != (end - 1u) / chunk;
}

kernel void byte_link_weights_prepare(
    device const uint* offsets [[buffer(0)]],
    device uint* accumulators [[buffer(1)]],
    constant LinkWeightParams& params [[buffer(2)]],
    uint key [[thread_position_in_grid]])
{
    if (key < params.keys && link_crossing(offsets, params.chunk, key)) {
        for (uint k = 0u; k < 5u; k++) {
            accumulators[5u * key + k] = 0u;
        }
    }
}

inline void link_flush_run(
    device const uint* offsets,
    device SolinasFp128* w,
    device atomic_uint* accumulators,
    uint start,
    uint stop,
    uint key,
    SolinasLazySum sum)
{
    if (offsets[key] >= start && offsets[key + 1u] <= stop) {
        w[key] = link_lazy_reduce(sum);
        return;
    }
    uint carry = 0u;
    for (uint limb = 0u; limb < 4u; limb++) {
        ulong addend = (ulong)sum.low.limb[limb] + (ulong)carry;
        uint low = (uint)addend;
        uint previous = atomic_fetch_add_explicit(&accumulators[5u * key + limb], low, memory_order_relaxed);
        carry = (uint)(addend >> 32) | (uint)(previous > 0xffffffffu - low);
    }
    if (carry + sum.overflow != 0u) {
        atomic_fetch_add_explicit(&accumulators[5u * key + 4u], carry + sum.overflow, memory_order_relaxed);
    }
}

kernel void byte_link_weights_runs(
    device const uint2* sorted [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const SolinasFp128* eq_lo [[buffer(2)]],
    device const SolinasFp128* eq_hi [[buffer(3)]],
    device SolinasFp128* w [[buffer(4)]],
    device atomic_uint* accumulators [[buffer(5)]],
    constant LinkWeightParams& params [[buffer(6)]],
    uint chunk [[thread_position_in_grid]])
{
    uint start = chunk * params.chunk;
    if (start >= params.entries) {
        return;
    }
    uint stop = min(start + params.chunk, params.entries);
    uint key = sorted[start].x;
    SolinasLazySum sum = solinas_lazy_zero();
    for (uint i = start; i < stop; i++) {
        uint2 entry = sorted[i];
        if (entry.x != key) {
            link_flush_run(offsets, w, accumulators, start, stop, key, sum);
            key = entry.x;
            sum = solinas_lazy_zero();
        }
        solinas_lazy_add(sum, link_split(eq_lo, eq_hi, params.lo_bits, entry.y));
    }
    link_flush_run(offsets, w, accumulators, start, stop, key, sum);
}

kernel void byte_link_weights_finish(
    device const uint* offsets [[buffer(0)]],
    device const uint* accumulators [[buffer(1)]],
    device SolinasFp128* w [[buffer(2)]],
    constant LinkWeightParams& params [[buffer(3)]],
    uint key [[thread_position_in_grid]])
{
    if (key >= params.keys) {
        return;
    }
    if (offsets[key + 1u] == offsets[key]) {
        w[key] = solinas_zero();
    } else if (link_crossing(offsets, params.chunk, key)) {
        SolinasLazySum sum;
        sum.low.limb = uint4(accumulators[5u * key], accumulators[5u * key + 1u], accumulators[5u * key + 2u], accumulators[5u * key + 3u]);
        sum.overflow = accumulators[5u * key + 4u];
        w[key] = link_lazy_reduce(sum);
    }
}

// partials[block][c] = eq_hi[block] * sum over the block's rows of eq_lo[row] * sigma(byte),
// sigma(byte) = (byte ^ 0x80) - 128 so each product is an unsigned 128 x 8-bit accumulation.
struct LinkMleParams {
    uint n;
    uint lo_bits;
    uint rows_per_thread;
    uint columns;
};

kernel void byte_link_column_mle(
    device const uchar* q [[buffer(0)]],
    device const SolinasFp128* eq_lo [[buffer(1)]],
    device const SolinasFp128* eq_hi [[buffer(2)]],
    device SolinasFp128* partials [[buffer(3)]],
    constant LinkMleParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 simd_sums[32u * LINK_SIMDS];
    uint first = block * (LINK_THREADS * params.rows_per_thread) + tid * params.rows_per_thread;
    uint lo_mask = (1u << params.lo_bits) - 1u;
    SolinasLazySum weights = solinas_lazy_zero();
    for (uint r = 0u; r < params.rows_per_thread; r++) {
        solinas_lazy_add(weights, eq_lo[(first + r) & lo_mask]);
    }
    SolinasFp128 weight_sum = link_lazy_reduce(weights);
    for (uint c = 0u; c < params.columns; c++) {
        device const uchar* column = q + (ulong)c * params.n;
        uint acc[5] = {0u, 0u, 0u, 0u, 0u};
        for (uint r = 0u; r < params.rows_per_thread; r++) {
            SolinasFp128 e = eq_lo[(first + r) & lo_mask];
            ulong u = (ulong)(column[first + r] ^ 0x80u);
            ulong carry = 0ul;
            for (uint k = 0u; k < 4u; k++) {
                ulong v = (ulong)e.limb[k] * u + (ulong)acc[k] + carry;
                acc[k] = (uint)v;
                carry = v >> 32;
            }
            acc[4] += (uint)carry;
        }
        SolinasLazySum sum;
        sum.low.limb = uint4(acc[0], acc[1], acc[2], acc[3]);
        sum.overflow = acc[4];
        SolinasFp128 value = solinas_sub(link_lazy_reduce(sum), solinas_half_width_mul_u64(weight_sum, 128ul));
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            simd_sums[c * LINK_SIMDS + simd] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < params.columns) {
        SolinasFp128 total = simd_sums[tid * LINK_SIMDS];
        for (uint s = 1u; s < LINK_SIMDS; s++) {
            total = solinas_add(total, simd_sums[tid * LINK_SIMDS + s]);
        }
        uint rows = LINK_THREADS * params.rows_per_thread;
        partials[block * params.columns + tid] = solinas_mul_wide(eq_hi[(block * rows) >> params.lo_bits], total);
    }
}

// sums[c] = sum over blocks of partials[block][c]; one threadgroup per column.
kernel void byte_link_sum_columns(
    device const SolinasFp128* partials [[buffer(0)]],
    device SolinasFp128* sums [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint column [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup SolinasFp128 simd_sums[LINK_SUM_THREADS / 32u];
    SolinasLazySum acc = solinas_lazy_zero();
    for (uint b = tid; b < params.x; b += LINK_SUM_THREADS) {
        solinas_lazy_add(acc, partials[b * params.y + column]);
    }
    SolinasFp128 total = solinas_simd_sum_32(link_lazy_reduce(acc));
    if (lane == 0u) {
        simd_sums[simd] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint s = 1u; s < LINK_SUM_THREADS / 32u; s++) {
            total = solinas_add(total, simd_sums[s]);
        }
        sums[column] = total;
    }
}
