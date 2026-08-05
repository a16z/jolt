// Registers read/write-checking sparse cycle rounds.

struct JkRegRwEntryIdx {
    Fr256 val;
    ulong prev_val;
    ulong next_val;
    ushort ra;
    ushort wa;
    uchar col;
    uchar pad[3];
};

struct JkRegRwEntryF {
    Fr256 val;
    Fr256 ra;
    Fr256 wa;
    ulong prev_val;
    ulong next_val;
    uchar col;
    uchar pad[7];
};

struct JkRegRwBuildParams {
    uint rows;
};

kernel void jk_reg_rw_build(
    device const uchar* rs1_index [[buffer(0)]],
    device const uchar* rs2_index [[buffer(1)]],
    device const uchar* rd_index [[buffer(2)]],
    device const ulong* rs1_value [[buffer(3)]],
    device const ulong* rs2_value [[buffer(4)]],
    device const ulong* rd_pre_value [[buffer(5)]],
    device const ulong* rd_post_value [[buffer(6)]],
    device const uint* row_offsets [[buffer(7)]],
    device JkRegRwEntryIdx* entries [[buffer(8)]],
    constant JkRegRwBuildParams& p [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.rows) return;
    JkRegRwEntryIdx row[3];
    uint len = 0u;
    uchar rs1 = rs1_index[gid];
    uchar rs2 = rs2_index[gid];
    uchar rd = rd_index[gid];
    if (rs1 != 0xff) {
        JkRegRwEntryIdx entry = {};
        entry.val = jk_fr_mont_from_u64(rs1_value[gid]);
        entry.prev_val = rs1_value[gid];
        entry.next_val = rs1_value[gid];
        entry.ra = 1u;
        entry.col = rs1;
        row[len++] = entry;
    }
    if (rs2 != 0xff) {
        int found = -1;
        for (uint i = 0u; i < len; ++i) {
            if (row[i].col == rs2) found = int(i);
        }
        if (found >= 0) {
            row[found].ra = 3u;
        } else {
            JkRegRwEntryIdx entry = {};
            entry.val = jk_fr_mont_from_u64(rs2_value[gid]);
            entry.prev_val = rs2_value[gid];
            entry.next_val = rs2_value[gid];
            entry.ra = 2u;
            entry.col = rs2;
            row[len++] = entry;
        }
    }
    if (rd != 0xff) {
        int found = -1;
        for (uint i = 0u; i < len; ++i) {
            if (row[i].col == rd) found = int(i);
        }
        if (found >= 0) {
            row[found].wa = 1u;
            row[found].next_val = rd_post_value[gid];
        } else {
            JkRegRwEntryIdx entry = {};
            entry.val = jk_fr_mont_from_u64(rd_pre_value[gid]);
            entry.prev_val = rd_pre_value[gid];
            entry.next_val = rd_post_value[gid];
            entry.wa = 1u;
            entry.col = rd;
            row[len++] = entry;
        }
    }
    for (uint i = 1u; i < len; ++i) {
        JkRegRwEntryIdx entry = row[i];
        uint j = i;
        while (j > 0u && row[j - 1u].col > entry.col) {
            row[j] = row[j - 1u];
            --j;
        }
        row[j] = entry;
    }
    uint dst = row_offsets[gid];
    for (uint i = 0u; i < len; ++i) entries[dst + i] = row[i];
}

inline void jk_reg_rw_accumulate(
    Fr256 ra0, Fr256 ra1, Fr256 wa0, Fr256 wa1,
    Fr256 val0, Fr256 val1, Fr256 inc0, Fr256 inc1,
    thread Fr256& q0, thread Fr256& qinf)
{
    q0 = fr_add(q0, fr_add(
        fr_mont_mul(ra0, val0), fr_mont_mul(wa0, fr_add(val0, inc0))));
    qinf = fr_add(qinf, fr_add(
        fr_mont_mul(ra1, val1), fr_mont_mul(wa1, fr_add(val1, inc1))));
}

struct JkRegRwMessageParams {
    uint pairs;
    uint num_tgs;
    uint eq_in_log;
    uint eq_in_len;
};

kernel void jk_reg_rw_message_idx(
    device const JkRegRwEntryIdx* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* ra_table [[buffer(2)]],
    device const uint* wa_table [[buffer(3)]],
    device const uint* inc [[buffer(4)]],
    device const uint* eq_out [[buffer(5)]],
    device const uint* eq_in [[buffer(6)]],
    device uint* partials [[buffer(7)]],
    device uint* counts [[buffer(8)]],
    constant JkRegRwMessageParams& p [[buffer(9)]],
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
        Fr256 inc1 = fr_sub(fr_load(inc, 2u * gid + 1u), inc0);
        uint i = start;
        uint j = mid;
        uint count = 0u;
        while (i < mid || j < end) {
            bool has_even = i < mid;
            bool has_odd = j < end;
            bool take_even = has_even;
            bool take_odd = has_odd;
            if (has_even && has_odd) {
                take_even = entries[i].col <= entries[j].col;
                take_odd = entries[j].col <= entries[i].col;
            }
            Fr256 ra0 = fr_zero();
            Fr256 ra1;
            Fr256 wa0 = fr_zero();
            Fr256 wa1;
            Fr256 val0;
            Fr256 val1;
            if (take_even && take_odd) {
                ra0 = fr_load(ra_table, entries[i].ra);
                ra1 = fr_sub(fr_load(ra_table, entries[j].ra), ra0);
                wa0 = fr_load(wa_table, entries[i].wa);
                wa1 = fr_sub(fr_load(wa_table, entries[j].wa), wa0);
                val0 = entries[i].val;
                val1 = fr_sub(entries[j].val, val0);
                i++;
                j++;
            } else if (take_even) {
                ra0 = fr_load(ra_table, entries[i].ra);
                ra1 = fr_sub(fr_zero(), ra0);
                wa0 = fr_load(wa_table, entries[i].wa);
                wa1 = fr_sub(fr_zero(), wa0);
                val0 = entries[i].val;
                val1 = fr_sub(jk_fr_mont_from_u64(entries[i].next_val), val0);
                i++;
            } else {
                ra1 = fr_load(ra_table, entries[j].ra);
                wa1 = fr_load(wa_table, entries[j].wa);
                val0 = fr_zero();
                val1 = fr_sub(entries[j].val, jk_fr_mont_from_u64(entries[j].prev_val));
                j++;
            }
            jk_reg_rw_accumulate(ra0, ra1, wa0, wa1, val0, val1, inc0, inc1, q0, qinf);
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

kernel void jk_reg_rw_message_f(
    device const JkRegRwEntryF* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* inc [[buffer(2)]],
    device const uint* eq_out [[buffer(3)]],
    device const uint* eq_in [[buffer(4)]],
    device uint* partials [[buffer(5)]],
    device uint* counts [[buffer(6)]],
    constant JkRegRwMessageParams& p [[buffer(7)]],
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
        Fr256 inc1 = fr_sub(fr_load(inc, 2u * gid + 1u), inc0);
        uint i = start;
        uint j = mid;
        uint count = 0u;
        while (i < mid || j < end) {
            bool has_even = i < mid;
            bool has_odd = j < end;
            bool take_even = has_even;
            bool take_odd = has_odd;
            if (has_even && has_odd) {
                take_even = entries[i].col <= entries[j].col;
                take_odd = entries[j].col <= entries[i].col;
            }
            Fr256 ra0 = fr_zero();
            Fr256 ra1;
            Fr256 wa0 = fr_zero();
            Fr256 wa1;
            Fr256 val0;
            Fr256 val1;
            if (take_even && take_odd) {
                ra0 = entries[i].ra;
                ra1 = fr_sub(entries[j].ra, ra0);
                wa0 = entries[i].wa;
                wa1 = fr_sub(entries[j].wa, wa0);
                val0 = entries[i].val;
                val1 = fr_sub(entries[j].val, val0);
                i++;
                j++;
            } else if (take_even) {
                ra0 = entries[i].ra;
                ra1 = fr_sub(fr_zero(), ra0);
                wa0 = entries[i].wa;
                wa1 = fr_sub(fr_zero(), wa0);
                val0 = entries[i].val;
                val1 = fr_sub(jk_fr_mont_from_u64(entries[i].next_val), val0);
                i++;
            } else {
                ra1 = entries[j].ra;
                wa1 = entries[j].wa;
                val0 = fr_zero();
                val1 = fr_sub(entries[j].val, jk_fr_mont_from_u64(entries[j].prev_val));
                j++;
            }
            jk_reg_rw_accumulate(ra0, ra1, wa0, wa1, val0, val1, inc0, inc1, q0, qinf);
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

struct JkRegRwBindIdxParams {
    uint pairs;
    uint ra_bits;
    uint wa_bits;
    uint r[FR_LIMBS];
};

kernel void jk_reg_rw_bind_idx(
    device const JkRegRwEntryIdx* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* out_offsets [[buffer(2)]],
    device JkRegRwEntryIdx* out [[buffer(3)]],
    constant JkRegRwBindIdxParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.pairs) return;
    uint i = row_offsets[2u * gid];
    uint mid = row_offsets[2u * gid + 1u];
    uint j = mid;
    uint end = row_offsets[2u * gid + 2u];
    uint dst = out_offsets[gid];
    Fr256 r = fr_load_const(p.r, 0);
    while (i < mid || j < end) {
        bool has_even = i < mid;
        bool has_odd = j < end;
        bool take_even = has_even;
        bool take_odd = has_odd;
        if (has_even && has_odd) {
            take_even = entries[i].col <= entries[j].col;
            take_odd = entries[j].col <= entries[i].col;
        }
        JkRegRwEntryIdx bound;
        if (take_even && take_odd) {
            JkRegRwEntryIdx even = entries[i];
            JkRegRwEntryIdx odd = entries[j];
            bound.col = even.col;
            bound.ra = (odd.ra << p.ra_bits) | even.ra;
            bound.wa = (odd.wa << p.wa_bits) | even.wa;
            bound.val = fr_add(even.val, fr_mont_mul(r, fr_sub(odd.val, even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = odd.next_val;
            i++;
            j++;
        } else if (take_even) {
            JkRegRwEntryIdx even = entries[i];
            bound.col = even.col;
            bound.ra = even.ra;
            bound.wa = even.wa;
            bound.val = fr_add(even.val, fr_mont_mul(
                r, fr_sub(jk_fr_mont_from_u64(even.next_val), even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = even.next_val;
            i++;
        } else {
            JkRegRwEntryIdx odd = entries[j];
            Fr256 even_val = jk_fr_mont_from_u64(odd.prev_val);
            bound.col = odd.col;
            bound.ra = odd.ra << p.ra_bits;
            bound.wa = odd.wa << p.wa_bits;
            bound.val = fr_add(even_val, fr_mont_mul(r, fr_sub(odd.val, even_val)));
            bound.prev_val = odd.prev_val;
            bound.next_val = odd.next_val;
            j++;
        }
        out[dst++] = bound;
    }
}

struct JkRegRwBindParams {
    uint pairs;
    uint r[FR_LIMBS];
};

kernel void jk_reg_rw_bind_idx_to_f(
    device const JkRegRwEntryIdx* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* out_offsets [[buffer(2)]],
    device JkRegRwEntryF* out [[buffer(3)]],
    device const uint* ra_table [[buffer(4)]],
    device const uint* wa_table [[buffer(5)]],
    constant JkRegRwBindParams& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.pairs) return;
    uint i = row_offsets[2u * gid];
    uint mid = row_offsets[2u * gid + 1u];
    uint j = mid;
    uint end = row_offsets[2u * gid + 2u];
    uint dst = out_offsets[gid];
    Fr256 r = fr_load_const(p.r, 0);
    while (i < mid || j < end) {
        bool has_even = i < mid;
        bool has_odd = j < end;
        bool take_even = has_even;
        bool take_odd = has_odd;
        if (has_even && has_odd) {
            take_even = entries[i].col <= entries[j].col;
            take_odd = entries[j].col <= entries[i].col;
        }
        JkRegRwEntryF bound;
        if (take_even && take_odd) {
            JkRegRwEntryIdx even = entries[i];
            JkRegRwEntryIdx odd = entries[j];
            Fr256 ra0 = fr_load(ra_table, even.ra);
            Fr256 wa0 = fr_load(wa_table, even.wa);
            bound.col = even.col;
            bound.ra = fr_add(ra0, fr_mont_mul(r, fr_sub(fr_load(ra_table, odd.ra), ra0)));
            bound.wa = fr_add(wa0, fr_mont_mul(r, fr_sub(fr_load(wa_table, odd.wa), wa0)));
            bound.val = fr_add(even.val, fr_mont_mul(r, fr_sub(odd.val, even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = odd.next_val;
            i++;
            j++;
        } else if (take_even) {
            JkRegRwEntryIdx even = entries[i];
            Fr256 one_minus_r = fr_sub(jk_fr_mont_from_u64(1ul), r);
            bound.col = even.col;
            bound.ra = fr_mont_mul(one_minus_r, fr_load(ra_table, even.ra));
            bound.wa = fr_mont_mul(one_minus_r, fr_load(wa_table, even.wa));
            bound.val = fr_add(even.val, fr_mont_mul(
                r, fr_sub(jk_fr_mont_from_u64(even.next_val), even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = even.next_val;
            i++;
        } else {
            JkRegRwEntryIdx odd = entries[j];
            Fr256 even_val = jk_fr_mont_from_u64(odd.prev_val);
            bound.col = odd.col;
            bound.ra = fr_mont_mul(r, fr_load(ra_table, odd.ra));
            bound.wa = fr_mont_mul(r, fr_load(wa_table, odd.wa));
            bound.val = fr_add(even_val, fr_mont_mul(r, fr_sub(odd.val, even_val)));
            bound.prev_val = odd.prev_val;
            bound.next_val = odd.next_val;
            j++;
        }
        out[dst++] = bound;
    }
}

kernel void jk_reg_rw_bind_f(
    device const JkRegRwEntryF* entries [[buffer(0)]],
    device const uint* row_offsets [[buffer(1)]],
    device const uint* out_offsets [[buffer(2)]],
    device JkRegRwEntryF* out [[buffer(3)]],
    constant JkRegRwBindParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.pairs) return;
    uint i = row_offsets[2u * gid];
    uint mid = row_offsets[2u * gid + 1u];
    uint j = mid;
    uint end = row_offsets[2u * gid + 2u];
    uint dst = out_offsets[gid];
    Fr256 r = fr_load_const(p.r, 0);
    while (i < mid || j < end) {
        bool has_even = i < mid;
        bool has_odd = j < end;
        bool take_even = has_even;
        bool take_odd = has_odd;
        if (has_even && has_odd) {
            take_even = entries[i].col <= entries[j].col;
            take_odd = entries[j].col <= entries[i].col;
        }
        JkRegRwEntryF bound;
        if (take_even && take_odd) {
            JkRegRwEntryF even = entries[i];
            JkRegRwEntryF odd = entries[j];
            bound.col = even.col;
            bound.ra = fr_add(even.ra, fr_mont_mul(r, fr_sub(odd.ra, even.ra)));
            bound.wa = fr_add(even.wa, fr_mont_mul(r, fr_sub(odd.wa, even.wa)));
            bound.val = fr_add(even.val, fr_mont_mul(r, fr_sub(odd.val, even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = odd.next_val;
            i++;
            j++;
        } else if (take_even) {
            JkRegRwEntryF even = entries[i];
            Fr256 one_minus_r = fr_sub(jk_fr_mont_from_u64(1ul), r);
            bound.col = even.col;
            bound.ra = fr_mont_mul(one_minus_r, even.ra);
            bound.wa = fr_mont_mul(one_minus_r, even.wa);
            bound.val = fr_add(even.val, fr_mont_mul(
                r, fr_sub(jk_fr_mont_from_u64(even.next_val), even.val)));
            bound.prev_val = even.prev_val;
            bound.next_val = even.next_val;
            i++;
        } else {
            JkRegRwEntryF odd = entries[j];
            Fr256 even_val = jk_fr_mont_from_u64(odd.prev_val);
            bound.col = odd.col;
            bound.ra = fr_mont_mul(r, odd.ra);
            bound.wa = fr_mont_mul(r, odd.wa);
            bound.val = fr_add(even_val, fr_mont_mul(r, fr_sub(odd.val, even_val)));
            bound.prev_val = odd.prev_val;
            bound.next_val = odd.next_val;
            j++;
        }
        out[dst++] = bound;
    }
}
