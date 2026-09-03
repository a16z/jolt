// Lazy-RA device tier (stage 6b): the round-message mass of the
// booleanity-cycle and instruction-RA-virtualization kernels, riding the
// LazyRaDevice seam. Shares the packed InstructionCycleRow layout with
// instruction.metal (12 u32 words per row) and the slot-round helpers from
// kernels.metal (jk_round_pair, jk_tg_sum).
//
// Phases mirror optimized/lazy_ra.rs exactly:
// - Lazy rounds (widths 1/2/4): per-poly branch-table gathers off the rows,
//   summand lanes reduced to lane-major per-threadgroup partials.
// - Adoption round (width 8): the lanes of a lazy round fused with every
//   poly's dense gather at cycles/16 into ONE flat poly-major buffer the
//   dense rounds ping-pong.
// - Dense rounds: fold + lanes fused, compact strides len -> len/2.
//
// eq = e_out·e_in multiplies each row's lane sums (the CPU folds e_in into
// the block lanes and e_out at block end — equal by distributivity, exact).

// Max per-virtual batch width the RA-virtualization kernels hold in thread
// state (production n is 2 or 4).
#define JK_RAV_MAX_BATCH 8u

// The hot chunk index of polynomial family `kind` at unbound cycle `j`:
// family 0 chunks the 128-bit lookup index (always hot), family 1 the
// +1-sentinel mapped PC, family 2 the +1-sentinel remapped RAM address
// (0 = cold). Mirrors ColumnSelector::index / RaChunkSelector::chunk_u128.
inline bool jk_ra_hot_index(device const uint* rows, uint j, uint kind, uint shift,
                            uint mask, thread uint& idx)
{
    device const uint* r = rows + j * 12u;
    if (kind == 0u) {
        ulong lo = (ulong)r[0] | ((ulong)r[1] << 32);
        ulong hi = (ulong)r[2] | ((ulong)r[3] << 32);
        ulong word = (shift >= 64u)
            ? (hi >> (shift - 64u))
            : ((lo >> shift) | jk_shl64_unbounded(hi, 64u - shift));
        idx = (uint)(word & (ulong)mask);
        return true;
    }
    ulong plus_one = (kind == 1u) ? ((ulong)r[4] | ((ulong)r[5] << 32))
                                  : ((ulong)r[6] | ((ulong)r[7] << 32));
    if (plus_one == 0ul) {
        return false;
    }
    idx = (uint)(((plus_one - 1ul) >> shift) & (ulong)mask);
    return true;
}

// The eq-weighted branch gather (lazy_ra::gather): one table lookup and one
// add per hot branch of unbound width `width`; the eq weights are pre-scaled
// into the branch tables. `table` points at poly i's flat branch tables
// (offset-major, stride k_entries).
inline Fr256 jk_ra_gather(device const uint* rows, device const uint* table, uint width,
                          uint k_entries, uint kind, uint shift, uint mask, uint j)
{
    Fr256 sum = fr_zero();
    for (uint off = 0u; off < width; off++) {
        uint idx;
        if (jk_ra_hot_index(rows, j * width + off, kind, shift, mask, idx)) {
            sum = fr_add(sum, fr_load(table + off * k_entries * FR_LIMBS, idx));
        }
    }
    return sum;
}

struct BoolLazyParams {
    uint groups;   // gruen row-pair count = (cycles >> binds) / 2
    uint num_tgs;
    uint log_in;   // log2(e_in length)
    uint width;    // branch count (1, 2, or 4)
    uint num_polys;
    uint k_entries;
    uint mask;
};

// Booleanity-cycle lazy round: per row the inner-quadratic coefficient sums
// sum_i H0·(H0 − rho_i) and sum_i (H1 − H0)^2 over gathered pairs, weighted
// by eq — lane-major partials [q_constant, q_leading] the host feeds to
// gruen_poly_deg_3. Reads only; declining or failing costs nothing.
kernel void jk_bool_lazy_round(
    device const uint* rows [[buffer(0)]],
    device const uint* meta [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const uint* rho [[buffer(3)]],
    device const uint* e_in [[buffer(4)]],
    device const uint* e_out [[buffer(5)]],
    device uint* partials [[buffer(6)]],
    constant BoolLazyParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    uint per_poly = p.width * p.k_entries * FR_LIMBS;
    Fr256 constant_acc = fr_zero();
    Fr256 leading_acc = fr_zero();
    for (uint i = 0u; i < p.num_polys; i++) {
        if (!active) {
            break;
        }
        uint kind = meta[2u * i];
        uint shift = meta[2u * i + 1u];
        device const uint* table = tables + i * per_poly;
        Fr256 h0 = jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask, 2u * gid);
        Fr256 h1 =
            jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask, 2u * gid + 1u);
        Fr256 delta = fr_sub(h1, h0);
        constant_acc = fr_add(constant_acc, fr_mont_mul(h0, fr_sub(h0, fr_load(rho, i))));
        leading_acc = fr_add(leading_acc, fr_mont_mul(delta, delta));
    }
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, constant_acc), partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, leading_acc), partials, 1u, p.num_tgs);
}

struct BoolDenseParams {
    uint groups;   // post-bind per-poly pair count
    uint do_bind;
    uint num_tgs;
    uint log_in;
    uint num_polys;
    uint len;      // per-poly CURRENT (pre-bind) length = cur stride
    uint r[FR_LIMBS];
};

// Booleanity-cycle dense round over the flat poly-major tables: fold the
// pending challenge per poly and accumulate the same two lanes.
kernel void jk_bool_dense_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device const uint* rho [[buffer(2)]],
    device const uint* e_in [[buffer(3)]],
    device const uint* e_out [[buffer(4)]],
    device uint* partials [[buffer(5)]],
    constant BoolDenseParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0u;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    Fr256 constant_acc = fr_zero();
    Fr256 leading_acc = fr_zero();
    for (uint i = 0u; i < p.num_polys; i++) {
        Fr256 h0, h1;
        jk_round_pair(cur + i * p.len * FR_LIMBS, nxt + i * (p.len >> 1) * FR_LIMBS, bind, r,
                      gid, active, h0, h1);
        Fr256 delta = fr_sub(h1, h0);
        constant_acc = fr_add(constant_acc, fr_mont_mul(h0, fr_sub(h0, fr_load(rho, i))));
        leading_acc = fr_add(leading_acc, fr_mont_mul(delta, delta));
    }
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, constant_acc), partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, leading_acc), partials, 1u, p.num_tgs);
}

struct RavLazyParams {
    uint groups;
    uint num_tgs;
    uint log_in;
    uint width;
    uint num_polys;  // num_virtual · batch
    uint batch;      // committed per virtual (2 ≤ batch ≤ JK_RAV_MAX_BATCH)
    uint k_entries;
    uint mask;
};

// RA-virtualization lazy round: per row, per virtual batch, the product grid
// [q(1), …, q(batch−1), q(∞)] over the batch's gathered pairs, batches
// summed, eq-weighted — the lanes gruen_poly_from_evals consumes.
kernel void jk_rav_lazy_round(
    device const uint* rows [[buffer(0)]],
    device const uint* meta [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const uint* e_in [[buffer(3)]],
    device const uint* e_out [[buffer(4)]],
    device uint* partials [[buffer(5)]],
    constant RavLazyParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    uint batch = min(p.batch, JK_RAV_MAX_BATCH);
    uint per_poly = p.width * p.k_entries * FR_LIMBS;
    Fr256 evals[JK_RAV_MAX_BATCH];
    Fr256 steps[JK_RAV_MAX_BATCH];
    Fr256 acc[JK_RAV_MAX_BATCH];
    for (uint l = 0u; l < batch; l++) {
        acc[l] = fr_zero();
    }
    for (uint i = 0u; active && i < p.num_polys; i += batch) {
        for (uint f = 0u; f < batch; f++) {
            uint poly = i + f;
            uint kind = meta[2u * poly];
            uint shift = meta[2u * poly + 1u];
            device const uint* table = tables + poly * per_poly;
            Fr256 lo =
                jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask, 2u * gid);
            Fr256 hi = jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask,
                                    2u * gid + 1u);
            evals[f] = hi;
            steps[f] = fr_sub(hi, lo);
        }
        for (uint t = 1u; t < batch; t++) {
            Fr256 prod = evals[0];
            for (uint f = 1u; f < batch; f++) {
                prod = fr_mont_mul(prod, evals[f]);
            }
            acc[t - 1u] = fr_add(acc[t - 1u], prod);
            if (t + 1u < batch) {
                for (uint f = 0u; f < batch; f++) {
                    evals[f] = fr_add(evals[f], steps[f]);
                }
            }
        }
        Fr256 lead = steps[0];
        for (uint f = 1u; f < batch; f++) {
            lead = fr_mont_mul(lead, steps[f]);
        }
        acc[batch - 1u] = fr_add(acc[batch - 1u], lead);
    }
    for (uint l = 0u; l < batch; l++) {
        jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, acc[l]), partials, l, p.num_tgs);
    }
}

struct BoolAdoptParams {
    uint groups;     // new_len / 2 message pairs
    uint num_tgs;
    uint log_in;
    uint width;      // branch count (2 · lazy horizon, i.e. 16)
    uint num_polys;
    uint k_entries;
    uint mask;
    uint len;        // dense length = cycles / width (cur stride)
};

// Booleanity-cycle fused adoption round: thread `gid` gathers each
// polynomial's (h0, h1) dense pair at width `width` — the values
// lazy_ra::materialize would store — writes them into the flat poly-major
// `cur` (stride `len`), and accumulates the same two summand lanes as
// jk_bool_dense_round. One pass over the rows replaces the legacy
// materialize dispatch PLUS the message round's full re-read of `cur`.
kernel void jk_bool_adopt_round(
    device const uint* rows [[buffer(0)]],
    device const uint* meta [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device uint* cur [[buffer(3)]],
    device const uint* rho [[buffer(4)]],
    device const uint* e_in [[buffer(5)]],
    device const uint* e_out [[buffer(6)]],
    device uint* partials [[buffer(7)]],
    constant BoolAdoptParams& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    uint per_poly = p.width * p.k_entries * FR_LIMBS;
    Fr256 constant_acc = fr_zero();
    Fr256 leading_acc = fr_zero();
    for (uint i = 0u; i < p.num_polys; i++) {
        if (!active) {
            break;
        }
        uint kind = meta[2u * i];
        uint shift = meta[2u * i + 1u];
        device const uint* table = tables + i * per_poly;
        Fr256 h0 = jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask, 2u * gid);
        Fr256 h1 =
            jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask, 2u * gid + 1u);
        fr_store(cur, i * p.len + 2u * gid, h0);
        fr_store(cur, i * p.len + 2u * gid + 1u, h1);
        Fr256 delta = fr_sub(h1, h0);
        constant_acc = fr_add(constant_acc, fr_mont_mul(h0, fr_sub(h0, fr_load(rho, i))));
        leading_acc = fr_add(leading_acc, fr_mont_mul(delta, delta));
    }
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, constant_acc), partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, leading_acc), partials, 1u, p.num_tgs);
}

struct RavAdoptParams {
    uint groups;     // new_len / 2 message pairs
    uint num_tgs;
    uint log_in;
    uint width;      // branch count (2 · lazy horizon, i.e. 16)
    uint num_polys;
    uint batch;
    uint k_entries;
    uint mask;
    uint len;        // dense length = cycles / width (cur stride)
};

// RA-virtualization fused adoption round: thread `gid` gathers each
// polynomial's (lo, hi) dense pair at width `width` — the same values
// lazy_ra::materialize would store — writes them into the flat poly-major
// `cur` (stride `len`), and accumulates the same batched product grid as
// jk_rav_dense_round. One pass over the rows replaces the legacy
// materialize dispatch PLUS the message round's full re-read of `cur`.
kernel void jk_rav_adopt_round(
    device const uint* rows [[buffer(0)]],
    device const uint* meta [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device uint* cur [[buffer(3)]],
    device const uint* e_in [[buffer(4)]],
    device const uint* e_out [[buffer(5)]],
    device uint* partials [[buffer(6)]],
    constant RavAdoptParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    uint batch = min(p.batch, JK_RAV_MAX_BATCH);
    uint per_poly = p.width * p.k_entries * FR_LIMBS;
    Fr256 evals[JK_RAV_MAX_BATCH];
    Fr256 steps[JK_RAV_MAX_BATCH];
    Fr256 acc[JK_RAV_MAX_BATCH];
    for (uint l = 0u; l < batch; l++) {
        acc[l] = fr_zero();
    }
    for (uint i = 0u; active && i < p.num_polys; i += batch) {
        for (uint f = 0u; f < batch; f++) {
            uint poly = i + f;
            uint kind = meta[2u * poly];
            uint shift = meta[2u * poly + 1u];
            device const uint* table = tables + poly * per_poly;
            Fr256 lo =
                jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask, 2u * gid);
            Fr256 hi = jk_ra_gather(rows, table, p.width, p.k_entries, kind, shift, p.mask,
                                    2u * gid + 1u);
            fr_store(cur, poly * p.len + 2u * gid, lo);
            fr_store(cur, poly * p.len + 2u * gid + 1u, hi);
            evals[f] = hi;
            steps[f] = fr_sub(hi, lo);
        }
        for (uint t = 1u; t < batch; t++) {
            Fr256 prod = evals[0];
            for (uint f = 1u; f < batch; f++) {
                prod = fr_mont_mul(prod, evals[f]);
            }
            acc[t - 1u] = fr_add(acc[t - 1u], prod);
            if (t + 1u < batch) {
                for (uint f = 0u; f < batch; f++) {
                    evals[f] = fr_add(evals[f], steps[f]);
                }
            }
        }
        Fr256 lead = steps[0];
        for (uint f = 1u; f < batch; f++) {
            lead = fr_mont_mul(lead, steps[f]);
        }
        acc[batch - 1u] = fr_add(acc[batch - 1u], lead);
    }
    for (uint l = 0u; l < batch; l++) {
        jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, acc[l]), partials, l, p.num_tgs);
    }
}

struct RavDenseParams {
    uint groups;
    uint do_bind;
    uint num_tgs;
    uint log_in;
    uint num_polys;
    uint batch;
    uint len;
    uint r[FR_LIMBS];
};

// RA-virtualization dense round: fold + the same batched product grid.
kernel void jk_rav_dense_round(
    device const uint* cur [[buffer(0)]],
    device uint* nxt [[buffer(1)]],
    device const uint* e_in [[buffer(2)]],
    device const uint* e_out [[buffer(3)]],
    device uint* partials [[buffer(4)]],
    constant RavDenseParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    bool active = gid < p.groups;
    bool bind = p.do_bind != 0u;
    Fr256 r = fr_load_const(p.r, 0);
    Fr256 eq = fr_zero();
    if (active) {
        eq = fr_mont_mul(fr_load(e_out, gid >> p.log_in),
                         fr_load(e_in, gid & ((1u << p.log_in) - 1u)));
    }
    uint batch = min(p.batch, JK_RAV_MAX_BATCH);
    Fr256 evals[JK_RAV_MAX_BATCH];
    Fr256 steps[JK_RAV_MAX_BATCH];
    Fr256 acc[JK_RAV_MAX_BATCH];
    for (uint l = 0u; l < batch; l++) {
        acc[l] = fr_zero();
    }
    for (uint i = 0u; i < p.num_polys; i += batch) {
        for (uint f = 0u; f < batch; f++) {
            uint poly = i + f;
            Fr256 lo, hi;
            jk_round_pair(cur + poly * p.len * FR_LIMBS, nxt + poly * (p.len >> 1) * FR_LIMBS,
                          bind, r, gid, active, lo, hi);
            evals[f] = hi;
            steps[f] = fr_sub(hi, lo);
        }
        for (uint t = 1u; t < batch; t++) {
            Fr256 prod = evals[0];
            for (uint f = 1u; f < batch; f++) {
                prod = fr_mont_mul(prod, evals[f]);
            }
            acc[t - 1u] = fr_add(acc[t - 1u], prod);
            if (t + 1u < batch) {
                for (uint f = 0u; f < batch; f++) {
                    evals[f] = fr_add(evals[f], steps[f]);
                }
            }
        }
        Fr256 lead = steps[0];
        for (uint f = 1u; f < batch; f++) {
            lead = fr_mont_mul(lead, steps[f]);
        }
        acc[batch - 1u] = fr_add(acc[batch - 1u], lead);
    }
    for (uint l = 0u; l < batch; l++) {
        jk_tg_sum(scratch, lid, tg, fr_mont_mul(eq, acc[l]), partials, l, p.num_tgs);
    }
}
