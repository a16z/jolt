// RAM read/write-checking cycle rounds. Sparse rows are merged on device;
// the small collapsed matrix remains storageModeShared for the host tail.

struct JkRamRwEntry {
    Fr256 val;
    Fr256 ra;
    ulong prev_val;
    ulong next_val;
    uint col;
    uint pad;
};

inline Fr256 jk_ram_rw_val_term(Fr256 val, Fr256 inc, Fr256 gamma) {
    return fr_add(val, fr_mont_mul(gamma, fr_add(inc, val)));
}

struct JkRamRwMessageParams {
    uint pairs;
    uint num_tgs;
    uint eq_in_log;
    uint eq_in_len;
    uint gamma[FR_LIMBS];
};

kernel void jk_ram_rw_message(
    device const JkRamRwEntry* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* inc [[buffer(2)]],
    device const uint* eq_out [[buffer(3)]],
    device const uint* eq_in [[buffer(4)]],
    device uint* partials [[buffer(5)]],
    device uint* counts [[buffer(6)]],
    constant JkRamRwMessageParams& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup uint scratch[FR_LIMBS * JK_TG_SIZE];
    Fr256 q0 = fr_zero();
    Fr256 qinf = fr_zero();
    if (gid < p.pairs) {
        uint start = row_offsets[2u * gid];
        uint mid = row_offsets[2u * gid + 1u];
        uint end = row_offsets[2u * gid + 2u];
        Fr256 inc0 = fr_load(inc, 2u * gid);
        Fr256 inc_slope = fr_sub(fr_load(inc, 2u * gid + 1u), inc0);
        Fr256 gamma = fr_load_const(p.gamma, 0);
        uint i = start;
        uint j = mid;
        uint count = 0u;
        while (i < mid || j < end) {
            bool has_even = i < mid;
            bool has_odd = j < end;
            bool take_even = has_even;
            bool take_odd = has_odd;
            if (has_even && has_odd) {
                uint ce = entries[i].col;
                uint co = entries[j].col;
                take_even = ce <= co;
                take_odd = co <= ce;
            }
            if (take_even && take_odd) {
                Fr256 ra0 = entries[i].ra;
                Fr256 val0 = entries[i].val;
                q0 = fr_add(q0, fr_mont_mul(
                    ra0, jk_ram_rw_val_term(val0, inc0, gamma)));
                qinf = fr_add(qinf, fr_mont_mul(
                    fr_sub(entries[j].ra, ra0),
                    jk_ram_rw_val_term(fr_sub(entries[j].val, val0), inc_slope, gamma)));
                i++;
                j++;
            } else if (take_even) {
                Fr256 ra0 = entries[i].ra;
                Fr256 val0 = entries[i].val;
                q0 = fr_add(q0, fr_mont_mul(
                    ra0, jk_ram_rw_val_term(val0, inc0, gamma)));
                qinf = fr_add(qinf, fr_mont_mul(
                    fr_sub(fr_zero(), ra0),
                    jk_ram_rw_val_term(
                        fr_sub(jk_fr_mont_from_u64(entries[i].next_val), val0),
                        inc_slope,
                        gamma)));
                i++;
            } else {
                Fr256 even_val = jk_fr_mont_from_u64(entries[j].prev_val);
                qinf = fr_add(qinf, fr_mont_mul(
                    entries[j].ra,
                    jk_ram_rw_val_term(
                        fr_sub(entries[j].val, even_val), inc_slope, gamma)));
                j++;
            }
            count++;
        }
        counts[gid] = count;
        Fr256 scale = fr_load(eq_out, gid >> p.eq_in_log);
        if (p.eq_in_len > 1u) {
            scale = fr_mont_mul(scale, fr_load(eq_in, gid & (p.eq_in_len - 1u)));
        }
        q0 = fr_mont_mul(scale, q0);
        qinf = fr_mont_mul(scale, qinf);
    }
    jk_tg_sum(scratch, lid, tg, q0, partials, 0u, p.num_tgs);
    jk_tg_sum(scratch, lid, tg, qinf, partials, 1u, p.num_tgs);
}

struct JkRamRwBindParams {
    uint pairs;
    uint r[FR_LIMBS];
};

kernel void jk_ram_rw_bind(
    device const JkRamRwEntry* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* out_offsets [[buffer(2)]],
    device JkRamRwEntry* out [[buffer(3)]],
    device const uint* inc [[buffer(4)]],
    device uint* out_inc [[buffer(5)]],
    constant JkRamRwBindParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.pairs) {
        return;
    }
    uint start = row_offsets[2u * gid];
    uint mid = row_offsets[2u * gid + 1u];
    uint end = row_offsets[2u * gid + 2u];
    Fr256 r = fr_load_const(p.r, 0);
    uint i = start;
    uint j = mid;
    uint dst = out_offsets[gid];
    while (i < mid || j < end) {
        bool has_even = i < mid;
        bool has_odd = j < end;
        bool take_even = has_even;
        bool take_odd = has_odd;
        if (has_even && has_odd) {
            uint ce = entries[i].col;
            uint co = entries[j].col;
            take_even = ce <= co;
            take_odd = co <= ce;
        }
        JkRamRwEntry bound;
        bound.pad = 0u;
        if (take_even && take_odd) {
            JkRamRwEntry even = entries[i];
            JkRamRwEntry odd = entries[j];
            bound.col = even.col;
            bound.ra = fr_add(even.ra, fr_mont_mul(r, fr_sub(odd.ra, even.ra)));
            bound.val = fr_add(even.val, fr_mont_mul(r, fr_sub(odd.val, even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = odd.next_val;
            i++;
            j++;
        } else if (take_even) {
            JkRamRwEntry even = entries[i];
            Fr256 odd_val = jk_fr_mont_from_u64(even.next_val);
            bound.col = even.col;
            bound.ra = fr_mont_mul(fr_sub(jk_fr_mont_from_u64(1ul), r), even.ra);
            bound.val = fr_add(even.val, fr_mont_mul(r, fr_sub(odd_val, even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = even.next_val;
            i++;
        } else {
            JkRamRwEntry odd = entries[j];
            Fr256 even_val = jk_fr_mont_from_u64(odd.prev_val);
            bound.col = odd.col;
            bound.ra = fr_mont_mul(r, odd.ra);
            bound.val = fr_add(even_val, fr_mont_mul(r, fr_sub(odd.val, even_val)));
            bound.prev_val = odd.prev_val;
            bound.next_val = odd.next_val;
            j++;
        }
        out[dst++] = bound;
    }
    Fr256 inc0 = fr_load(inc, 2u * gid);
    fr_store(out_inc, gid, fr_add(
        inc0, fr_mont_mul(r, fr_sub(fr_load(inc, 2u * gid + 1u), inc0))));
}
